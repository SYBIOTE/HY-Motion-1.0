#!/usr/bin/env python3
"""Verify or populate HY-Motion checkpoints under CKPTS_ROOT.

Invoked:
  • At container start (via docker-entrypoint.sh) — verify only; exit 1 if missing.
  • At image build — pass --bundled-build to download into image layers.

Env:
  CKPTS_ROOT     Root for tencent/, Qwen3-8B/, clip-vit-large-patch14/ (default: ckpts).
  MODEL_PATH     Lite model dir; must match {CKPTS_ROOT}/tencent/HY-Motion-1.0-Lite.
  USE_HF_MODELS  If truthy ("1"), skip local Qwen + CLIP checks.
  HF_HOME        Override the download staging dir (default: {CKPTS_ROOT}/.hf-cache).
"""

from __future__ import annotations

import argparse
import os
import sys


def _truthy(val: str | None, default: bool = False) -> bool:
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _motion_lite_dir(ckpts_root: str) -> str:
    return os.path.join(ckpts_root, "tencent", "HY-Motion-1.0-Lite")


def _motion_ready(lite_dir: str) -> bool:
    ckpt = os.path.join(lite_dir, "latest.ckpt")
    cfg = os.path.join(lite_dir, "config.yml")
    return os.path.isfile(ckpt) and os.path.isfile(cfg)


def _sidecar_ready(kind: str, root: str) -> bool:
    cfg = os.path.join(root, "config.json")
    if not os.path.isfile(cfg):
        return False
    if kind == "qwen":
        index = os.path.join(root, "model.safetensors.index.json")
        single = os.path.join(root, "model.safetensors")
        return os.path.isfile(index) or os.path.isfile(single)
    # CLIP ViT-L
    return (
        os.path.isfile(os.path.join(root, "model.safetensors"))
        or os.path.isfile(os.path.join(root, "pytorch_model.bin"))
    )


def _cache_dir(ckpts_root: str) -> str:
    """
    Where huggingface_hub stages downloads before moving them into place.

    It defaults to ~/.cache/huggingface — the container disk, which on a RunPod
    pod is far smaller than the volume being seeded. Qwen3-8B alone is 16GB in
    the cache plus 16GB at the destination, which overruns a 40GB disk with
    "Disk quota exceeded". Stage beside the target instead, on the volume.
    """
    return os.environ.get("HF_HOME") or os.path.join(ckpts_root, ".hf-cache")


def _download_motion(lite_root_parent: str, cache_dir: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="tencent/HY-Motion-1.0",
        allow_patterns="HY-Motion-1.0-Lite/*",
        local_dir=lite_root_parent,
        cache_dir=cache_dir,
    )


def _download_qwen(target: str, cache_dir: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id="Qwen/Qwen3-8B", local_dir=target, cache_dir=cache_dir)


def _download_clip(target: str, cache_dir: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="openai/clip-vit-large-patch14", local_dir=target, cache_dir=cache_dir
    )


def _verify(
    ckpts_root: str,
    lite_dir: str,
    use_hf: bool,
) -> int:
    if not _motion_ready(lite_dir):
        print(
            f"ERROR: Motion checkpoint missing under {lite_dir}. "
            f"Mount a populated volume at CKPTS_ROOT ({ckpts_root}). "
            "Expected config.yml and latest.ckpt.",
            file=sys.stderr,
        )
        return 1

    if use_hf:
        print(">>> Checkpoint layout ready (USE_HF_MODELS — Qwen/CLIP from Hugging Face).")
        return 0

    qwen_root = os.path.join(ckpts_root, "Qwen3-8B")
    clip_root = os.path.join(ckpts_root, "clip-vit-large-patch14")
    missing = []
    if not _sidecar_ready("qwen", qwen_root):
        missing.append(f"Qwen3-8B ({qwen_root})")
    if not _sidecar_ready("clip", clip_root):
        missing.append(f"clip-vit-large-patch14 ({clip_root})")
    if missing:
        print(
            "ERROR: USE_HF_MODELS=0 but text encoder weights are missing: "
            + ", ".join(missing)
            + f". Mount them under {ckpts_root} or set USE_HF_MODELS=1.",
            file=sys.stderr,
        )
        return 1

    print(">>> Checkpoint layout ready.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify or populate CKPTS_ROOT for HY-Motion.")
    parser.add_argument(
        "--bundled-build",
        action="store_true",
        help="Docker build phase: download weights into image layers.",
    )
    args = parser.parse_args()

    ckpts_root = os.environ.get("CKPTS_ROOT", "ckpts")
    ckpts_root = os.path.abspath(ckpts_root)

    inference_lite = os.path.abspath(_motion_lite_dir(ckpts_root))
    mp = os.environ.get("MODEL_PATH")
    if mp:
        requested = os.path.abspath(mp)
        if requested != inference_lite:
            print(
                f"ERROR: MODEL_PATH ({requested}) must match {inference_lite} "
                f"(derived from CKPTS_ROOT={ckpts_root}). Adjust CKPTS_ROOT or MODEL_PATH.",
                file=sys.stderr,
            )
            return 1

    use_hf = _truthy(os.environ.get("USE_HF_MODELS"), False)

    if not args.bundled_build:
        return _verify(ckpts_root, inference_lite, use_hf)

    os.makedirs(ckpts_root, exist_ok=True)
    tencent_parent = os.path.join(ckpts_root, "tencent")
    os.makedirs(tencent_parent, exist_ok=True)

    cache_dir = _cache_dir(ckpts_root)
    os.makedirs(cache_dir, exist_ok=True)
    print(f">>> Staging downloads via {cache_dir}")

    if not _motion_ready(inference_lite):
        print(f">>> Ensuring HY-Motion-1.0-Lite weights under {tencent_parent} ...")
        _download_motion(tencent_parent, cache_dir)

    if use_hf:
        print(">>> USE_HF_MODELS enabled — skipping local Qwen/CLIP download.")
        return _verify(ckpts_root, inference_lite, use_hf)

    qwen_root = os.path.join(ckpts_root, "Qwen3-8B")
    clip_root = os.path.join(ckpts_root, "clip-vit-large-patch14")

    if not _sidecar_ready("qwen", qwen_root):
        print(f">>> Ensuring Qwen3-8B under {qwen_root} ...")
        _download_qwen(qwen_root, cache_dir)

    if not _sidecar_ready("clip", clip_root):
        print(f">>> Ensuring CLIP ViT-L under {clip_root} ...")
        _download_clip(clip_root, cache_dir)

    return _verify(ckpts_root, inference_lite, use_hf)


if __name__ == "__main__":
    raise SystemExit(main())
