#!/usr/bin/env python3
"""Download HY-Motion checkpoints into CKPTS_ROOT (volume or bundled image).

Invoked:
  • At container start (via docker-entrypoint.sh) — idempotent skips if present.
  • At image build — pass --bundled-build to always populate layers (Cloud Build).

Env:
  CKPTS_ROOT        Root for tencent/, Qwen3-8B/, clip-vit-large-patch14/ (default: ckpts).
  MODEL_PATH        Lite model dir; used to infer required motion files (optional).
  USE_HF_MODELS     If truthy ("1"), skip Qwen + CLIP local downloads.
  AUTO_DOWNLOAD_CKPTS   If falsy ("0"), fail at runtime if checkpoints missing (--bundled-build ignores).
"""

from __future__ import annotations

import argparse
import os
import sys


def _truthy(val: str | None, default: bool = True) -> bool:
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


def _download_motion(lite_root_parent: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="tencent/HY-Motion-1.0",
        allow_patterns="HY-Motion-1.0-Lite/*",
        local_dir=lite_root_parent,
    )


def _download_qwen(target: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id="Qwen/Qwen3-8B", local_dir=target)


def _download_clip(target: str) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id="openai/clip-vit-large-patch14", local_dir=target)


def main() -> int:
    parser = argparse.ArgumentParser(description="Populate CKPTS_ROOT for HY-Motion.")
    parser.add_argument(
        "--bundled-build",
        action="store_true",
        help="Docker build phase: download regardless of AUTO_DOWNLOAD_CKPTS.",
    )
    args = parser.parse_args()

    ckpts_root = os.environ.get("CKPTS_ROOT", "ckpts")
    ckpts_root = os.path.abspath(ckpts_root)
    os.makedirs(ckpts_root, exist_ok=True)

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

    lite_dir = inference_lite

    bundled = args.bundled_build
    auto_dl = bundled or _truthy(os.environ.get("AUTO_DOWNLOAD_CKPTS"), True)
    use_hf = _truthy(os.environ.get("USE_HF_MODELS"), False)

    if not bundled and not auto_dl:
        if not _motion_ready(lite_dir):
            print(
                f"ERROR: Motion checkpoint missing under {lite_dir} "
                "(set AUTO_DOWNLOAD_CKPTS=1 or mount a populated volume)",
                file=sys.stderr,
            )
            return 1
        qwen_root = os.path.join(ckpts_root, "Qwen3-8B")
        clip_root = os.path.join(ckpts_root, "clip-vit-large-patch14")
        if not use_hf and (
            not _sidecar_ready("qwen", qwen_root) or not _sidecar_ready("clip", clip_root)
        ):
            print(
                "ERROR: USE_HF_MODELS=0 but Qwen and/or CLIP weights are missing "
                f"under {ckpts_root}. Enable AUTO_DOWNLOAD_CKPTS or mount weights.",
                file=sys.stderr,
            )
            return 1
        return 0

    tencent_parent = os.path.join(ckpts_root, "tencent")
    os.makedirs(tencent_parent, exist_ok=True)

    if bundled or not _motion_ready(inference_lite):
        print(f">>> Ensuring HY-Motion-1.0-Lite weights under {tencent_parent} ...")
        _download_motion(tencent_parent)

    if use_hf:
        print(">>> USE_HF_MODELS enabled — skipping local Qwen/CLIP download.")
        return 0

    qwen_root = os.path.join(ckpts_root, "Qwen3-8B")
    clip_root = os.path.join(ckpts_root, "clip-vit-large-patch14")

    if bundled or not _sidecar_ready("qwen", qwen_root):
        print(f">>> Ensuring Qwen3-8B under {qwen_root} ...")
        _download_qwen(qwen_root)

    if bundled or not _sidecar_ready("clip", clip_root):
        print(f">>> Ensuring CLIP ViT-L under {clip_root} ...")
        _download_clip(clip_root)

    if not _motion_ready(inference_lite):
        print(f"ERROR: Motion bundle incomplete at {inference_lite}", file=sys.stderr)
        return 1
    if not _sidecar_ready("qwen", qwen_root) or not _sidecar_ready("clip", clip_root):
        print(
            "ERROR: Text encoder checkpoints still incomplete "
            f"(USE_HF_MODELS={os.environ.get('USE_HF_MODELS')!r})",
            file=sys.stderr,
        )
        return 1

    print(">>> Checkpoint layout ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
