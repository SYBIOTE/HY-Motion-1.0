#!/usr/bin/env python3
"""
Save an int4 copy of the Qwen3-8B text encoder.

Quantizing at worker start costs ~48s of a ~70s cold start: 16GB of fp16 shards
are read off the network volume and quantized with bitsandbytes on every boot,
producing the same result each time. Writing that result once (~5GB) lets a
worker load it directly.

Run once against the volume, then the loader picks it up automatically
(text_encoder._prequantized_qwen_dir).

    CKPTS_ROOT=/runpod-volume/ckpts python scripts/prequantize_qwen.py

Needs a GPU: bitsandbytes quantizes on-device.
"""

import argparse
import os
import shutil
import sys
import time

# Read weights from CKPTS_ROOT, not the HF hub.
os.environ.setdefault("USE_HF_MODELS", "0")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src",
        help="Source fp16 model (default: {CKPTS_ROOT}/Qwen3-8B)",
    )
    ap.add_argument(
        "--dst",
        help="Destination for the int4 copy (default: {CKPTS_ROOT}/Qwen3-8B-int4)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the destination if it already exists",
    )
    args = ap.parse_args()

    ckpts_root = os.environ.get("CKPTS_ROOT", "ckpts")
    src = args.src or os.path.join(ckpts_root, "Qwen3-8B")
    dst = args.dst or os.path.join(ckpts_root, "Qwen3-8B-int4")

    if not os.path.isdir(src):
        print(f"ERROR: source model not found: {src}", file=sys.stderr)
        return 1

    if os.path.exists(dst):
        if not args.force:
            print(f"ERROR: {dst} exists. Pass --force to overwrite.", file=sys.stderr)
            return 1
        print(f">>> Removing existing {dst}")
        shutil.rmtree(dst)

    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen2ForCausalLM

    if not torch.cuda.is_available():
        print("ERROR: a GPU is required (bitsandbytes quantizes on-device).", file=sys.stderr)
        return 1

    # Must match the load-time config in text_encoder.py, or the saved weights
    # will not match what the encoder expects.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print(f">>> Loading + quantizing {src} (this is the slow part, once)")
    t0 = time.time()
    model = Qwen2ForCausalLM.from_pretrained(
        src,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f">>> Quantized in {time.time() - t0:.1f}s")

    print(f">>> Saving to {dst}")
    t1 = time.time()
    model.save_pretrained(dst, safe_serialization=True)

    # The tokenizer is loaded from the original path, but keep a copy beside the
    # weights so the directory stands alone.
    try:
        AutoTokenizer.from_pretrained(
            src, padding_side="right", use_fast=False, trust_remote_code=True
        ).save_pretrained(dst)
    except Exception as e:  # noqa: BLE001 - tokenizer copy is a convenience
        print(f">>> [WARNING] Could not copy tokenizer: {e}")

    print(f">>> Saved in {time.time() - t1:.1f}s")

    def _du(path: str) -> float:
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total / 1e9

    print(f">>> Done. {src}: {_du(src):.1f}GB -> {dst}: {_du(dst):.1f}GB")
    print(">>> Workers will now load the int4 copy automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
