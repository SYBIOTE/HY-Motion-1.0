"""
RunPod queue-endpoint handler for HY-Motion.

Wraps the same runtime the FastAPI app uses (api.run_motion), so local Docker
and the load-balancer deployment stay unchanged; this module only swaps the
transport from HTTP to the RunPod job queue.

    Job input:  {"op": "motion", "text": ..., "duration": ..., "seed": ..., "cfg_scale": ...}
    Job output: {"motion": {...}, "meta": {...}}
    On failure: {"error": "...", "code": "..."}

Env:
    See api.py. The model is loaded once at import and reused across jobs on a
    warm worker, so cold start is paid per worker rather than per job.
"""

import runpod
from pydantic import ValidationError

from api import MotionRequest, MotionUnavailable, get_runtime, run_motion

# Supported job ops. The queue collapses every HTTP route onto one endpoint, so
# the payload carries the discriminator the URL path used to.
OP_MOTION = "motion"


def _error(code: str, message: str) -> dict:
    return {"error": message, "code": code}


def handler(job: dict) -> dict:
    payload = job.get("input") or {}

    op = payload.get("op", OP_MOTION)
    if op != OP_MOTION:
        return _error("bad_op", f"Unsupported op {op!r}; expected {OP_MOTION!r}")

    try:
        req = MotionRequest(**{k: v for k, v in payload.items() if k != "op"})
    except ValidationError as e:
        return _error("bad_input", e.json())

    try:
        return run_motion(req)
    except MotionUnavailable as e:
        return _error("unavailable", str(e))


# Warm the model at worker start rather than on the first job, so the first job
# out of the queue is not the one that pays for the checkpoint load.
try:
    get_runtime()
except Exception as e:  # noqa: BLE001 - mirrors api.lifespan; jobs report the real error
    print(f">>> [WARNING] Runtime not loaded at startup: {e}")

runpod.serverless.start({"handler": handler})
