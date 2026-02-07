"""
HY-Motion microservice: JSON-only API for text-to-motion.
Exposes POST /v1/motion and GET /health for use by Next.js or other clients.

Usage:
    MODEL_PATH=ckpts/tencent/HY-Motion-1.0-Lite python -m uvicorn api:app --host 0.0.0.0 --port 8080

Env:
    MODEL_PATH: Directory containing config.yml and latest.ckpt (default: ckpts/tencent/HY-Motion-1.0-Lite)
    DISABLE_PROMPT_ENGINEERING: Set to True to disable LLM prompt rewriter (saves VRAM)
    QWEN_QUANTIZATION: int4 | int8 | none (default: int4 for low VRAM)
    DISABLE_WOODEN_MESH: Set to 1 to skip loading wooden body model (keypoints3d=zeros, transl unadjusted; app does ground alignment)
    MMGP_PROFILE: 0=off, 1=max offload (lowest VRAM, slowest), 3=balanced (default: 3)
    MMGP_VERBOSE: Logging level for MMGP offloading (default: 1)
"""

import os
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Set env before importing runtime (for quantization, etc.)
os.environ.setdefault("QWEN_QUANTIZATION", "int4")
os.environ.setdefault("DISABLE_PROMPT_ENGINEERING", "True")
if "USE_HF_MODELS" not in os.environ:
    os.environ["USE_HF_MODELS"] = "1"

from hymotion.utils.t2m_runtime import T2MRuntime

MODEL_PATH = os.environ.get("MODEL_PATH", "ckpts/tencent/HY-Motion-1.0-Lite")
CONFIG_PATH = os.path.join(MODEL_PATH, "config.yml")
CKPT_PATH = os.path.join(MODEL_PATH, "latest.ckpt")
# Matches pipeline default output_mesh_fps (30). TODO: pass fps dynamically (e.g. from pipeline or request).
MOTION_FPS = 30

_runtime: T2MRuntime | None = None


def _apply_mmgp(runtime: T2MRuntime):
    """Apply MMGP memory offloading to trade speed for lower VRAM."""
    profile = int(os.environ.get("MMGP_PROFILE", "1"))
    if profile <= 0:
        return
    try:
        from mmgp import offload

        pipe = runtime.extract_models_for_mmgp()
        if not pipe:
            print(">>> [WARNING] No models extracted for MMGP offloading. Skipping.")
            return

        kwargs = {}
        if profile != 1 and profile != 3:
            kwargs["budgets"] = {"*": 4000}  # 4 GB budget for active components

        verbose = int(os.environ.get("MMGP_VERBOSE", "1"))
        print(f">>> Applying MMGP offload profile {profile}...")
        offload.profile(pipe, profile_no=profile, verboseLevel=verbose, **kwargs)
        print(f">>> MMGP offloading active (profile {profile}).")
    except ImportError:
        print(">>> [INFO] mmgp not installed. Skipping dynamic offloading.")
    except Exception as e:
        print(f">>> [ERROR] Failed to apply MMGP offloading: {e}")


def get_runtime() -> T2MRuntime:
    global _runtime
    if _runtime is None:
        if not os.path.exists(CONFIG_PATH):
            raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
        skip_loading = not os.path.exists(CKPT_PATH)
        _runtime = T2MRuntime(
            config_path=CONFIG_PATH,
            ckpt_name=CKPT_PATH,
            skip_text=False,
            device_ids=None,
            skip_model_loading=skip_loading,
            disable_prompt_engineering=os.environ.get("DISABLE_PROMPT_ENGINEERING", "true").lower() == "true",
            prompt_engineering_host=None,
            prompt_engineering_model_path=None,
        )
        _apply_mmgp(_runtime)
    return _runtime


def _tensor_to_list(x):
    """Convert tensor to nested list; take batch index 0 so shape is (num_frames, ...)."""
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    # Batch dim is 0; we use a single seed so take [0]
    if hasattr(x, "shape") and len(x.shape) >= 1:
        x = x[0]
    return x.tolist()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Optional: preload runtime on startup (can remove to lazy-load on first request)
    try:
        get_runtime()
    except Exception as e:
        print(f">>> [WARNING] Runtime not loaded at startup: {e}")
    yield
    # Shutdown: nothing to close for now


app = FastAPI(title="HY-Motion API", version="1.0", lifespan=lifespan)


class MotionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    duration: float = Field(default=3.0, ge=0.5, le=30.0)
    seed: int = Field(default=42, ge=0)
    cfg_scale: float = Field(default=5.0, ge=1.0, le=20.0)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/v1/motion")
def generate_motion(req: MotionRequest):
    try:
        runtime = get_runtime()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=f"Model not available: {e}")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unavailable: {e}")

    with tempfile.TemporaryDirectory(prefix="hymotion_") as tmpdir:
        try:
            _, _, model_output = runtime.generate_motion(
                text=req.text,
                seeds_csv=str(req.seed),
                duration=req.duration,
                cfg_scale=req.cfg_scale,
                output_format="dict",
                output_dir=tmpdir,
                original_text=req.text,
            )
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Generation failed: {e}")

    # model_output: dict with keypoints3d, rot6d, transl, root_rotations_mat, text (tensors, batch dim first)
    keypoints3d = model_output["keypoints3d"]
    rot6d = model_output["rot6d"]
    transl = model_output["transl"]
    root_rotations_mat = model_output["root_rotations_mat"]

    num_frames = keypoints3d.shape[1]

    motion = {
        "keypoints3d": _tensor_to_list(keypoints3d),
        "rot6d": _tensor_to_list(rot6d),
        "transl": _tensor_to_list(transl),
        "root_rotations_mat": _tensor_to_list(root_rotations_mat),
        "num_frames": int(num_frames),
        "fps": MOTION_FPS,
    }
    meta = {
        "text": req.text,
        "duration": req.duration,
        "seed": req.seed,
    }
    return {"motion": motion, "meta": meta}
