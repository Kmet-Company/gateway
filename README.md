# ViReAl — Gateway

FastAPI orchestrator that fronts the AI vision model and exposes a stable
REST + WebSocket contract to the Angular frontend.

## Architecture

```
browser ──► nginx (frontend/web) ──► /gateway/* ──► gateway:8000
                                                        │
                                  ┌─── POSTGREST_URL ───┤
                                  ▼                     ▼
                              api:3000              ai-vision:8000
                              (frontend)            (ai)
```

All three stacks share the docker network `vireal-net`.

## One-time host setup

```bash
docker network create vireal-net
```

## Build & run

```bash
# Full mesh: frontend + ai + gateway
(cd ../frontend && docker compose up -d)
(cd ../ai       && docker compose up -d)
docker compose up --build -d

# Gateway alone (mock AI, no ../ai needed):
AI_VISION_URL= docker compose up --build -d
```

## Environment

| var                     | default                    | notes |
|-------------------------|----------------------------|-------|
| `POSTGREST_URL`         | `http://api:3000`          | from frontend stack |
| `STATIC_ASSET_BASE`     | `http://web`               | where `/cam-main.mp4` etc. live |
| `AI_VISION_URL`         | `http://ai-vision:8000`    | empty = mock mode |
| `AI_FIRE_URL`           | `http://ai-vision:8000`    | fire classifier (if separate) |
| `ANALYSIS_INTERVAL_SEC` | `0`                        | 0 = on-demand only |

Put overrides in a `.env` next to `docker-compose.yml`:

```env
AI_VISION_URL=
ANALYSIS_INTERVAL_SEC=30
```

## Endpoints

- `GET  /health`                      — gateway + ai-vision ping
- `GET  /detections`                  — last detection per camera
- `POST /analyze/{camera_code}`       — run vision pipeline on one camera
- `POST /analyze-all`                 — fan out over all cameras
- `POST /predict-upload`              — multipart upload, returns raw model output
- `WS   /analyze-realtime/{code}`     — stream chunks, receive per-chunk results

## Accessing from the host

```bash
curl http://127.0.0.1:8000/health
```
