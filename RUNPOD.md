# RunPod (Serverless) — HY-Motion API

This branch targets **[RunPod](https://www.runpod.io/)**: GPU **Serverless** workers with a **network volume** for checkpoints (and optional S3-compatible sync to seed it). There is **no** Google Cloud Build or Cloud Run in this branch.

## Layout

| File | Role |
|------|------|
| `Dockerfile.base` | CUDA + PyTorch + `hymotion` + entrypoint; **does not** bake weights — use a volume |
| `Dockerfile` | API image: FastAPI on top of `HYMOTION_BASE_IMAGE` |

## Build & push (Docker Hub or any registry)

From repo root, with context **`HY-Motion-1.0/`** (required in a monorepo so `COPY` works):

```bash
cd HY-Motion-1.0
docker build -f Dockerfile.base -t YOUR_USER/hymotion-base:latest .
docker push YOUR_USER/hymotion-base:latest

docker build -f Dockerfile -t YOUR_USER/hymotion-api:latest .
docker push YOUR_USER/hymotion-api:latest
```

Override the base when building the API:

```bash
export HYMOTION_BASE_IMAGE=YOUR_USER/hymotion-base:latest
docker build --build-arg HYMOTION_BASE_IMAGE="$HYMOTION_BASE_IMAGE" -f Dockerfile -t YOUR_USER/hymotion-api:latest .
```

In **RunPod → Git / Docker build**, set the same **build argument** name: **`HYMOTION_BASE_IMAGE`**.

## RunPod Serverless

1. **Seed** the network volume (S3 sync or one-time download) so it contains `tencent/HY-Motion-1.0-Lite/`, `Qwen3-8B/`, `clip-vit-large-patch14/` under your chosen prefix (e.g. `ckpts/`). See `ckpts/README.md`. The worker **exits on startup** if checkpoints are missing.
2. Attach that **network volume** to the endpoint (same datacenter/region as the volume).
3. **Environment** (example when the mount is `/runpod-volume` and data is under `ckpts/`):

   | Variable | Example |
   |----------|---------|
   | `CKPTS_ROOT` | `/runpod-volume/ckpts` |
   | `MODEL_PATH` | `/runpod-volume/ckpts/tencent/HY-Motion-1.0-Lite` |
   | `USE_HF_MODELS` | `0` |

4. Point the worker image at your pushed **`hymotion-api`** image.

**HTTP probes:** `GET /ping` and `GET /health` both return **200** with `{"status":"ok"}` (`/ping` is for RunPod-style checks; neither route loads the model per request).

Private registry: add pull credentials in RunPod if needed.

## References

- API contract: `API_README.md`
- Checkpoint tree: `ckpts/README.md`
