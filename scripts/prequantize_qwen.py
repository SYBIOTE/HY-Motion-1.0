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
import json
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
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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

    # AutoModelForCausalLM, not Qwen2ForCausalLM: Qwen3 has QK-norm and no
    # attention biases, so forcing the Qwen2 class silently drops q_norm/k_norm
    # and randomly initializes q/k/v biases — a corrupted encoder that still
    # saves and loads. text_encoder.py registers qwen3->Qwen2Config only as a
    # fallback for transformers too old to know Qwen3; where the real class
    # exists it must win.
    print(f">>> Loading + quantizing {src} (this is the slow part, once)")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        src,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f">>> Loaded as {type(model).__name__}")
    if type(model).__name__.startswith("Qwen2"):
        print(
            "ERROR: loaded the Qwen2 architecture for Qwen3 weights; the saved "
            "encoder would be corrupt (dropped QK-norms, random biases). "
            "Upgrade transformers so Qwen3ForCausalLM is available.",
            file=sys.stderr,
        )
        return 1
    print(f">>> Quantized in {time.time() - t0:.1f}s")

    free_gb = shutil.disk_usage(os.path.dirname(os.path.abspath(dst))).free / 1e9
    if free_gb < 10:
        print(
            f"ERROR: only {free_gb:.1f}GB free at {dst}; the int4 copy needs ~7GB "
            "plus room for shard staging. Free space and re-run.",
            file=sys.stderr,
        )
        return 1

    print(f">>> Saving to {dst} ({free_gb:.1f}GB free)")
    t1 = time.time()
    model.save_pretrained(dst, safe_serialization=True)

    # Keep a tokenizer beside the weights so the directory stands alone once the
    # fp16 source is deleted.
    try:
        AutoTokenizer.from_pretrained(
            src, padding_side="right", use_fast=False, trust_remote_code=True
        ).save_pretrained(dst)
    except Exception as e:  # noqa: BLE001 - tokenizer copy is a convenience
        print(f">>> [WARNING] Could not copy tokenizer: {e}")

    # Round-tripping the tokenizer through save_pretrained writes
    # extra_special_tokens as a list, while transformers expects a mapping and
    # does `extra_special_tokens.keys()` on load ("'list' object has no
    # attribute 'keys'"). Normalise it rather than shipping a config that only
    # fails at worker start.
    tok_cfg_path = os.path.join(dst, "tokenizer_config.json")
    if os.path.exists(tok_cfg_path):
        with open(tok_cfg_path) as fh:
            tok_cfg = json.load(fh)
        extra = tok_cfg.get("extra_special_tokens")
        if isinstance(extra, list):
            tok_cfg["extra_special_tokens"] = {}
            tok_cfg.setdefault("additional_special_tokens", extra)
            with open(tok_cfg_path, "w") as fh:
                json.dump(tok_cfg, fh, indent=1, ensure_ascii=False)
            print(">>> Normalised extra_special_tokens (list -> mapping)")

    # Fail here rather than at worker start if the saved tokenizer cannot load.
    try:
        AutoTokenizer.from_pretrained(
            dst, padding_side="right", use_fast=True, trust_remote_code=True
        )
        print(">>> Verified: the saved tokenizer loads")
    except Exception as e:  # noqa: BLE001 - surfaced as a hard failure below
        print(f"ERROR: saved tokenizer does not load: {e}", file=sys.stderr)
        return 1

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
