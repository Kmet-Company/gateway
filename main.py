"""
AI gateway: reads camera `video_url` from PostgREST, downloads clips, and
forwards them to the violence-classification service in `ai/apis/deepseek_api.py`.

When AI_VISION_URL is unset, analysis returns a lightweight mock so the UI
can run without the `ai-vision` container. Docker Compose defaults it to
http://ai-vision:8000 (see frontend/docker-compose.yml).
"""

from __future__ import annotations

import asyncio
import math
import os
import subprocess
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware

POSTGREST_URL = os.environ.get("POSTGREST_URL", "http://api:3000").rstrip("/")
AI_VISION_URL = os.environ.get("AI_VISION_URL", "").rstrip("/")
AI_FIRE_URL = os.environ.get("AI_FIRE_URL", "").rstrip("/")
# When video_url in DB is "/foo.mp4", prepend this so the gateway can fetch it (e.g. http://web in Docker).
STATIC_ASSET_BASE = os.environ.get("STATIC_ASSET_BASE", "").rstrip("/")
ANALYSIS_INTERVAL_SEC = int(os.environ.get("ANALYSIS_INTERVAL_SEC", "0"))
MAX_DOWNLOAD_BYTES = int(os.environ.get("MAX_DOWNLOAD_BYTES", str(15 * 1024 * 1024)))

last_results: dict[str, dict[str, Any]] = {}


async def get_video_duration(path: str) -> float:
    """Get video duration in seconds using ffprobe."""
    result = await asyncio.to_thread(
        subprocess.run,
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    return float(result.stdout.strip())


async def extract_video_clip(input_path: str, output_path: str, start: float, duration: float) -> None:
    """Extract a clip from video using ffmpeg."""
    await asyncio.to_thread(
        subprocess.run,
        ["ffmpeg", "-i", input_path, "-ss", str(start), "-t", str(duration), "-c", "copy", "-y", output_path],
        check=True,
        capture_output=True,
    )


def _json_safe(value: Any) -> Any:
    """Ensure values are JSON-serializable (NaN/Inf break stdlib json → HTTP 500)."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    return value


def absolute_video_url(url: str) -> str:
    u = (url or "").strip()
    if u.startswith("/"):
        if not STATIC_ASSET_BASE:
            raise ValueError(
                "video_url is a site-relative path; set STATIC_ASSET_BASE "
                "(Docker: http://web). For local npm + gateway on host use e.g. "
                "http://host.docker.internal:4200 or http://localhost:4200."
            )
        return f"{STATIC_ASSET_BASE}{u}"
    return u


async def fetch_cameras() -> list[dict[str, Any]]:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(
            f"{POSTGREST_URL}/camera",
            params={"select": "code,video_url"},
        )
        r.raise_for_status()
        return r.json()


async def download_video(url: str) -> str:
    fetch_url = absolute_video_url(url)
    async with httpx.AsyncClient(
        timeout=120,
        follow_redirects=True,
        limits=httpx.Limits(max_keepalive_connections=5),
    ) as client:
        async with client.stream("GET", fetch_url) as resp:
            resp.raise_for_status()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            try:
                total = 0
                async for chunk in resp.aiter_bytes(65536):
                    total += len(chunk)
                    if total > MAX_DOWNLOAD_BYTES:
                        raise ValueError(
                            f"Video exceeds MAX_DOWNLOAD_BYTES ({MAX_DOWNLOAD_BYTES})",
                        )
                    tmp.write(chunk)
                tmp.flush()
                return tmp.name
            finally:
                tmp.close()


def _mime_for_path(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".webm"):
        return "video/webm"
    if lower.endswith(".mp4"):
        return "video/mp4"
    return "application/octet-stream"


async def analyze_video_file(path: str) -> dict[str, Any]:
    if not AI_VISION_URL:
        return {
            "mock": True,
            "filename": "demo",
            "results": [
                {
                    "start_time": 0.0,
                    "end_time": 3.0,
                    "violent_probability": 0.12,
                    "non_violent_probability": 0.88,
                    "prediction": "Non-Violent",
                }
            ],
        }

    url = f"{AI_VISION_URL}/predict-video/"
    try:
        async with httpx.AsyncClient(timeout=600) as client:
            with open(path, "rb") as handle:
                name = os.path.basename(path)
                files = {"file": (name, handle, _mime_for_path(path))}
                r = await client.post(url, files=files)
            r.raise_for_status()
            return r.json()
    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Cannot reach vision service at {url!r}: {exc!s}. "
            "If the gateway runs on the host (not Docker), set AI_VISION_URL to a "
            "host-reachable URL (e.g. http://127.0.0.1:8001 when compose publishes "
            "ai-vision on port 8001).",
        ) from exc


async def analyze_fire_file(path: str) -> dict[str, Any]:
    """Forward a clip to the YOLO fire service (`ai/apis/fire_detection.py`)."""
    if not AI_FIRE_URL:
        return {
            "mock": True,
            "filename": "demo",
            "results": [
                {
                    "start_time": 0.0,
                    "end_time": 3.0,
                    "fire_detected": False,
                    "max_fire_confidence": 0.0,
                    "frames_scored": 0,
                },
            ],
        }

    url = f"{AI_FIRE_URL}/predict-fire/"
    try:
        async with httpx.AsyncClient(timeout=600) as client:
            with open(path, "rb") as handle:
                name = os.path.basename(path)
                files = {"file": (name, handle, _mime_for_path(path))}
                r = await client.post(
                    url,
                    files=files,
                    params={"frame_skip": 5, "min_confidence": 0.25},
                )
            r.raise_for_status()
            return r.json()
    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Cannot reach fire service at {url!r}: {exc!s}. "
            "Set AI_FIRE_URL (e.g. http://ai-fire:8010) when the fire container is up.",
        ) from exc


async def run_one_camera(row: dict[str, Any]) -> None:
    code = row.get("code")
    url = row.get("video_url")
    if not code or not url:
        return
    path: str | None = None
    try:
        path = await download_video(url)
        duration = await get_video_duration(path)
        
        # Choose analyze function based on camera code
        if code in ["cam-main", "black_and_white"]:
            analyze_func = analyze_video_file
            result_key = "violence"
        elif code == "kocani":
            analyze_func = analyze_fire_file
            result_key = "fire"
        else:
            # Default to violence
            analyze_func = analyze_video_file
            result_key = "violence"
        
        results = []
        for start in range(0, int(duration), 3):
            end = min(start + 3, duration)
            clip_duration = end - start
            clip_path = tempfile.mktemp(suffix=".mp4")
            try:
                await extract_video_clip(path, clip_path, start, clip_duration)
                res = await analyze_func(clip_path)
                print(f"Camera {code}, interval {start:.1f}-{end:.1f}s: {res}")
                results.append({"start_time": start, "end_time": end, **res})
            finally:
                if os.path.exists(clip_path):
                    os.unlink(clip_path)
        
        last_results[code] = {
            "camera_code": code,
            "at": datetime.now(timezone.utc).isoformat(),
            result_key: {"results": results},
        }
    except Exception as exc:  # noqa: BLE001 — surface per-camera errors
        last_results[code] = {
            "camera_code": code,
            "at": datetime.now(timezone.utc).isoformat(),
            "error": str(exc),
        }
    finally:
        if path and os.path.isfile(path):
            os.unlink(path)


async def analysis_loop() -> None:
    while True:
        await asyncio.sleep(max(ANALYSIS_INTERVAL_SEC, 1))
        try:
            rows = await fetch_cameras()
            for row in rows:
                await run_one_camera(row)
        except Exception:
            # Log-free background loop; operators use /detections for state.
            pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    task: asyncio.Task | None = None
    if ANALYSIS_INTERVAL_SEC > 0:
        task = asyncio.create_task(analysis_loop())
    yield
    if task:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Vigilant AI Gateway", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, Any]:
    ai_ping: str | int | None = None
    if AI_VISION_URL:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{AI_VISION_URL}/health")
                ai_ping = r.status_code
        except Exception as exc:  # noqa: BLE001
            ai_ping = f"error: {exc!s}"

    fire_ping: str | int | None = None
    if AI_FIRE_URL:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{AI_FIRE_URL}/health")
                fire_ping = r.status_code
        except Exception as exc:  # noqa: BLE001
            fire_ping = f"error: {exc!s}"

    return {
        "ok": True,
        "postgrest": POSTGREST_URL,
        "ai_vision_configured": bool(AI_VISION_URL),
        "ai_vision_url": AI_VISION_URL or None,
        "ai_vision_ping": ai_ping,
        "ai_fire_configured": bool(AI_FIRE_URL),
        "ai_fire_url": AI_FIRE_URL or None,
        "ai_fire_ping": fire_ping,
        "static_asset_base": STATIC_ASSET_BASE or None,
        "analysis_interval_sec": ANALYSIS_INTERVAL_SEC,
    }


@app.get("/capabilities")
async def capabilities() -> dict[str, bool]:
    return {
        "violence": True,
        "fire": bool(AI_FIRE_URL),
    }


@app.get("/detections")
async def detections() -> dict[str, Any]:
    return _json_safe({"cameras": last_results})


@app.post("/analyze/{camera_code}")
async def analyze_camera(camera_code: str) -> dict[str, Any]:
    rows = await fetch_cameras()
    row = next((x for x in rows if x.get("code") == camera_code), None)
    if not row:
        raise HTTPException(404, "Unknown camera code")
    if not row.get("video_url"):
        raise HTTPException(400, "camera has no video_url set")
    await run_one_camera(row)
    return _json_safe(last_results.get(camera_code, {}))


@app.post("/analyze-all")
async def analyze_all() -> dict[str, Any]:
    rows = await fetch_cameras()
    for row in rows:
        await run_one_camera(row)
    return _json_safe({"cameras": last_results})


@app.post("/predict-upload")
async def predict_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """Forward a browser-uploaded clip (e.g. 3s WebM chunks) to the vision model."""
    suffix = os.path.splitext(file.filename or "clip")[1] or ".bin"
    path: str | None = None
    try:
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as handle:
            body = await file.read()
            if not body:
                raise HTTPException(status_code=400, detail="empty file upload")
            handle.write(body)
        assert path is not None
        data = await analyze_video_file(path)
        return _json_safe(data)
    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        resp = exc.response
        body = resp.text[:4000] if resp is not None else ""
        status = resp.status_code if resp is not None else 502
        if status < 100 or status > 599:
            status = 502
        # Pass through ai-vision status (500 was previously remapped to 502).
        raise HTTPException(
            status_code=status,
            detail={"msg": "ai-vision error", "upstream": body},
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "msg": "cannot reach ai-vision",
                "ai_vision_url": AI_VISION_URL,
                "error": str(exc),
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422,
            detail={"msg": "predict-upload failed", "error": str(exc)},
        ) from exc
    finally:
        if path and os.path.isfile(path):
            os.unlink(path)


@app.post("/predict-fire-upload")
async def predict_fire_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    """Forward a browser-uploaded clip to the YOLO fire detector (3s WebM chunks)."""
    suffix = os.path.splitext(file.filename or "clip")[1] or ".bin"
    path: str | None = None
    try:
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as handle:
            body = await file.read()
            if not body:
                raise HTTPException(status_code=400, detail="empty file upload")
            handle.write(body)
        assert path is not None
        data = await analyze_fire_file(path)
        return _json_safe(data)
    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        resp = exc.response
        body = resp.text[:4000] if resp is not None else ""
        status = resp.status_code if resp is not None else 502
        if status < 100 or status > 599:
            status = 502
        raise HTTPException(
            status_code=status,
            detail={"msg": "ai-fire error", "upstream": body},
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "msg": "cannot reach ai-fire",
                "ai_fire_url": AI_FIRE_URL,
                "error": str(exc),
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=422,
            detail={"msg": "predict-fire-upload failed", "error": str(exc)},
        ) from exc
    finally:
        if path and os.path.isfile(path):
            os.unlink(path)


@app.websocket("/analyze-realtime/{camera_code}")
async def analyze_realtime(websocket: WebSocket, camera_code: str):
    """Real-time analysis WebSocket: receives video chunks and returns immediate results."""
    await websocket.accept()
    
    try:
        # Get camera config
        rows = await fetch_cameras()
        row = next((x for x in rows if x.get("code") == camera_code), None)
        if not row:
            await websocket.send_json({"error": "Unknown camera code"})
            await websocket.close()
            return
        
        # Determine which analysis function to use
        if camera_code in ["cam-main", "black_and_white"]:
            analyze_func = analyze_video_file
            result_type = "violence"
        elif camera_code == "kocani":
            analyze_func = analyze_fire_file
            result_type = "fire"
        else:
            analyze_func = analyze_video_file
            result_type = "violence"
        
        await websocket.send_json({"status": "ready", "camera": camera_code, "type": result_type})
        
        while True:
            # Receive video chunk (expecting binary data)
            data = await websocket.receive_bytes()
            
            if not data:
                continue
                
            # Save chunk to temp file
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                temp_file.write(data)
                temp_path = temp_file.name
            
            try:
                # Analyze the chunk immediately
                result = await analyze_func(temp_path)
                
                # Send result back immediately
                response = {
                    "camera_code": camera_code,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "chunk_size": len(data),
                    "result": result
                }
                
                print(f"Real-time {camera_code}: {result}")
                await websocket.send_json(response)
                
            except Exception as exc:
                await websocket.send_json({
                    "error": str(exc),
                    "camera_code": camera_code,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
    except Exception as exc:
        try:
            await websocket.send_json({"error": f"WebSocket error: {str(exc)}"})
        except:
            pass
    finally:
        try:
            await websocket.close()
        except:
            pass
