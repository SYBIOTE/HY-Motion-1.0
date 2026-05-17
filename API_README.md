# HY-Motion API (microservice)

JSON-only HTTP API for text-to-motion. No Gradio, no FBX; returns motion data for use by Next.js or other clients.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness; returns `{"status":"ok"}` |
| POST | `/v1/motion` | Generate motion from text; returns `{ motion, meta }` |

## POST /v1/motion

**Request body (JSON):**
```json
{
  "text": "A person walks forward and waves",
  "duration": 3.0,
  "seed": 42,
  "cfg_scale": 5.0
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| text | string | required | Motion prompt |
| duration | float | 3.0 | Length in seconds (0.5–30) |
| seed | int | 42 | Random seed |
| cfg_scale | float | 5.0 | Guidance scale (1–20) |

**Response 200:**
```json
{
  "motion": {
    "keypoints3d": [[[x,y,z], ...], ...],
    "rot6d": [[[6d], ...], ...],
    "transl": [[tx,ty,tz], ...],
    "root_rotations_mat": [[[3x3], ...], ...],
    "num_frames": 60,
    "fps": 20
  },
  "meta": {
    "text": "A person walks forward and waves",
    "duration": 3.0,
    "seed": 42
  }
}
```

Shapes (single sample): `keypoints3d` [num_frames, num_joints, 3], `rot6d` [num_frames, num_joints, 6], `transl` [num_frames, 3], `root_rotations_mat` [num_frames, 3, 3].

## Run locally

```bash
# From repo root; ensure ckpts are present (see ckpts/README.md)
export MODEL_PATH=ckpts/tencent/HY-Motion-1.0-Lite
export QWEN_QUANTIZATION=int4
export DISABLE_PROMPT_ENGINEERING=True

python -m uvicorn api:app --host 0.0.0.0 --port 8080
```

Then: `curl -X POST http://localhost:8080/v1/motion -H "Content-Type: application/json" -d '{"text":"A person waves"}'`

## Docker

```bash
# Build base + API (from HY-Motion-1.0 root)
docker build --build-arg BUNDLE_CKPTS=1 -f Dockerfile.base -t hymotion-base .
docker build --build-arg BASE_IMAGE=hymotion-base:latest -t hymotion-api .

# Slim image + named volume — first boot downloads once into the volume (HF/token as needed).
docker build --build-arg BUNDLE_CKPTS=0 -f Dockerfile.base -t hymotion-base .
docker build --build-arg BASE_IMAGE=hymotion-base:latest -t hymotion-api .
docker run --gpus all -p 8080:8080 -v hymotion-ckpts:/app/ckpts hymotion-api

# Host bind-mount (reuse local ckpts tree)
docker run --gpus all -p 8080:8080 \
  -v "$(pwd)/ckpts:/app/ckpts" \
  -e MODEL_PATH=/app/ckpts/tencent/HY-Motion-1.0-Lite \
  -e CKPTS_ROOT=/app/ckpts \
  -e AUTO_DOWNLOAD_CKPTS=0 \
  hymotion-api

# Cloud Build (`cloudbuild.yaml`) sets BUNDLE_CKPTS=1 by default — weights ship in the image.
```

For **RunPod** (or any Git-connected build), the API image defaults to **`docker.io/sybiote/hymotion-base:latest`**. Change the `BASE_IMAGE` default in `Dockerfile` / `Dockerfile.gradio` if your Hub namespace differs, or pass a build-arg `BASE_IMAGE=yourname/hymotion-base:tag`. Use **build context** `HY-Motion-1.0` (monorepo subdir) so `COPY` paths resolve.

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| CKPTS_ROOT | /app/ckpts (Docker) | Root for motion + local Qwen + CLIP trees |
| MODEL_PATH | (Docker) `/app/ckpts/tencent/HY-Motion-1.0-Lite`; must mirror `$CKPTS_ROOT`/tencent/…‑Lite | Derived from CKPTS_ROOT; do not point at another tree unless you sync CKPTS_ROOT |
| AUTO_DOWNLOAD_CKPTS | 1 | On startup download missing checkpoints (set 0 when volume already filled) |
| SKIP_CHECKPOINT_PREP | 0 | Set 1 to bypass entrypoint prefetch (dangerous unless layout is guaranteed) |
| QWEN_QUANTIZATION | int4 | int4 / int8 / none |
| DISABLE_PROMPT_ENGINEERING | True | Disable LLM rewriter (saves VRAM) |
| USE_HF_MODELS | 1 (outside Docker) / 0 in image | HF hub IDs vs local dirs under CKPTS_ROOT |

## Next.js integration

Set `HY_MOTION_API_URL` to the API base URL (e.g. `http://localhost:8080`). The Next.js app calls `POST {HY_MOTION_API_URL}/v1/motion` and uses the returned `motion` for cleanup and retargeting.
