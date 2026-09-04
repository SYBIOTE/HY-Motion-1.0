# HY-Motion job API (RunPod queue)

Text-to-motion as a **RunPod Serverless (queue) endpoint**. No HTTP service: the
worker pulls jobs off the queue, so clients submit a job and poll for the result
rather than holding a request open.

**Deployment:** see [RUNPOD.md](RUNPOD.md).

## Calling the endpoint

Submit with `POST https://api.runpod.ai/v2/{ENDPOINT_ID}/run`, then poll
`GET https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{JOB_ID}` until `status` is
`COMPLETED` or `FAILED`. Both take `Authorization: Bearer $RUNPOD_API_KEY`.

Use `/runsync` instead of `/run` for short clips to get the result in one call;
it holds the connection open and is subject to RunPod's sync timeout.

**Job input:**
```json
{
  "input": {
    "op": "motion",
    "text": "A person walks forward and waves",
    "duration": 3.0,
    "seed": 42,
    "cfg_scale": 5.0
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| op | string | `"motion"` | Job discriminator; only `motion` is supported |
| text | string | required | Motion prompt |
| duration | float | 3.0 | Length in seconds (0.5–30) |
| seed | int | 42 | Random seed |
| cfg_scale | float | 5.0 | Guidance scale (1–20) |

**Completed job:** the payload below is the `output` field of the `/status` response.
```json
{
  "motion": {
    "keypoints3d": [[[0.0, 0.0, 0.0]]],
    "rot6d": [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
    "transl": [[0.0, 0.0, 0.0]],
    "root_rotations_mat": [[[0.0, 0.0, 0.0]]],
    "num_frames": 90,
    "fps": 30
  },
  "meta": {
    "text": "A person walks forward and waves",
    "duration": 3.0,
    "seed": 42
  }
}
```

Shapes (single sample): `keypoints3d` [num_frames, num_joints, 3], `rot6d`
[num_frames, num_joints, 6], `transl` [num_frames, 3], `root_rotations_mat`
[num_frames, 3, 3].

**Failed job:** the handler returns an error object as `output` (the job itself
still reports `COMPLETED`, since the worker did not crash):

```json
{ "error": "Model not available: ...", "code": "unavailable" }
```

| `code` | Meaning |
|--------|---------|
| `bad_op` | Unsupported `op` |
| `bad_input` | Input failed validation; `error` is the Pydantic report |
| `unavailable` | Runtime could not load, or generation failed |

### Response size

A completed job's `output` is returned inline and is bounded by RunPod's
`/status` body limit (~2 MB). At 30 fps the payload runs roughly 0.4 MB for 3 s
and 1.6 MB for 12 s (the 360-frame training limit), so the full supported range
fits, but long clips leave little headroom. Anything longer needs the motion
written to the network volume or object storage with a reference in `output`.

## Run locally

The worker needs a queue to pull from, so RunPod's SDK serves a local test
server when no queue is configured:

```bash
export MODEL_PATH=ckpts/tencent/HY-Motion-1.0-Lite
export QWEN_QUANTIZATION=int4
export DISABLE_PROMPT_ENGINEERING=True

python -u handler.py --rp_serve_api
```

Then: `curl -X POST http://localhost:8000/runsync -H "Content-Type: application/json" -d '{"input":{"text":"A person waves"}}'`

Or drive the handler directly, without the SDK:

```bash
python -c 'from api import MotionRequest, run_motion; print(run_motion(MotionRequest(text="A person waves"))["motion"]["num_frames"])'
```

## Docker (local GPU)

From the `HY-Motion-1.0/` directory:

```bash
docker build -f Dockerfile.base -t hymotion-base .
docker build --build-arg HYMOTION_BASE_IMAGE=hymotion-base:latest -f Dockerfile -t hymotion-queue .

docker run --gpus all \
  -v "$(pwd)/ckpts:/app/ckpts" \
  -e MODEL_PATH=/app/ckpts/tencent/HY-Motion-1.0-Lite \
  -e CKPTS_ROOT=/app/ckpts \
  -e RUNPOD_API_KEY="$RUNPOD_API_KEY" \
  hymotion-queue
```

No port is published: the container reaches out to the queue rather than
accepting connections.

**RunPod (Git build):** set build arg **`HYMOTION_BASE_IMAGE`** to your pushed
base image if not using the Dockerfile default. Build context must be
**`HY-Motion-1.0/`** in a monorepo. Details: [RUNPOD.md](RUNPOD.md).

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| CKPTS_ROOT | /app/ckpts (Docker) | Root for motion + local Qwen + CLIP trees; must be populated (volume mount) |
| MODEL_PATH | `/app/ckpts/tencent/HY-Motion-1.0-Lite` in image | Must match `{CKPTS_ROOT}/tencent/HY-Motion-1.0-Lite` |
| SKIP_CHECKPOINT_PREP | 0 | 1 = skip entrypoint checkpoint step |
| QWEN_QUANTIZATION | int4 | int4 / int8 / none |
| DISABLE_PROMPT_ENGINEERING | True | Disable LLM rewriter (saves VRAM) |
| USE_HF_MODELS | 1 (bare local) / 0 in image | HF hub IDs vs dirs under CKPTS_ROOT |
| QWEN_INT4_PATH | `{CKPTS_ROOT}/Qwen3-8B-int4` | Pre-quantized encoder; used when present (see below) |

## Cold start

A worker loads the model once at start, so the cost is per worker rather than
per job. Of a ~70s start, ~48s is the Qwen3-8B text encoder: 16GB of fp16
shards read off the network volume and quantized to int4 on every boot, for a
result that is identical each time.

Write that result once instead:

```bash
CKPTS_ROOT=/runpod-volume/ckpts python scripts/prequantize_qwen.py
```

That saves a ~5GB int4 copy to `{CKPTS_ROOT}/Qwen3-8B-int4`, which workers then
load directly — roughly 48s down to 12-15s. Needs a GPU (bitsandbytes quantizes
on-device); run it on a pod with the volume attached. The step is optional: with
no such directory the encoder quantizes at load exactly as before, and
`QWEN_QUANTIZATION=none` or `int8` bypasses it entirely.

## Next.js integration

Set `HY_MOTION_ENDPOINT_ID` to the RunPod endpoint ID and `RUNPOD_API_KEY` for
auth. The app submits to `/run` and polls `/status/{id}`, then uses the returned
`motion` for cleanup and retargeting.
