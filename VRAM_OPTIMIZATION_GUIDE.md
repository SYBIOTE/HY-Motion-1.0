# HY-Motion-1.0 VRAM Optimization Guide

## Overview

This guide covers every technique available to reduce GPU VRAM usage when running HY-Motion-1.0-Lite. All entry points (`api.py`, `gradio_app.py`, `local_infer.py`) default to the most memory-efficient settings out of the box.

### Where VRAM goes

| Component | Model | Unoptimized VRAM | With INT4 |
| :--- | :--- | :--- | :--- |
| **Qwen3-8B text encoder** | 8B params | ~16 GB (BF16) | ~4-5 GB |
| **CLIP-ViT-L/14** | ~430M params (FP16) | ~0.8 GB | ~0.8 GB |
| **DiT denoiser (Lite)** | 0.46B params | ~1-2 GB | ~1-2 GB |
| **Activations / KV cache** | Per-request temporary | ~1-2 GB | ~1-2 GB |
| **CUDA context + PyTorch** | Fixed overhead | ~0.5-1 GB | ~0.5-1 GB |

### Configuration presets

| Preset | VRAM | Speed | Best for |
| :--- | :--- | :--- | :--- |
| INT4 + MMGP 1 **(default)** | ~5-6 GB peak | Slowest | 8-16 GB GPUs, Docker/Cloud Run |
| INT4 + MMGP 3 | ~8-10 GB peak | Moderate | 12-16 GB GPUs |
| INT4 + no MMGP | ~10-12 GB | Fastest | 16-24 GB GPUs |
| INT8 + no MMGP | ~14-16 GB | Fast | 24 GB GPUs |
| No quantization | ~22-24 GB | Baseline | 24+ GB GPUs |

---

## Technique 1: Quantization (biggest single win)

Qwen3-8B is the largest component. INT4 quantization shrinks it from ~16 GB to ~4-5 GB with minimal quality loss.

Controlled by `QWEN_QUANTIZATION`. Defaults to `int4` in all entry points.

```bash
# INT4 (default, recommended)
QWEN_QUANTIZATION=int4 python gradio_app.py

# INT8 (better quality, more VRAM)
QWEN_QUANTIZATION=int8 python gradio_app.py

# No quantization (requires >24GB GPU)
QWEN_QUANTIZATION=none python gradio_app.py
```

### Requirements

```bash
pip install bitsandbytes>=0.41.0 accelerate>=0.20.0
```

Verify: `python -c "import bitsandbytes; print(bitsandbytes.__version__)"`

---

## Technique 2: MMGP Memory Offloading (complements quantization)

MMGP dynamically moves model components between GPU VRAM and CPU RAM. Only the component currently doing a forward pass stays in VRAM, so peak usage equals the single largest component plus overhead -- not the sum of all components.

Controlled by `MMGP_PROFILE`. Defaults to `1` (most aggressive) in the Docker container and API.

| Profile | Behavior | Peak VRAM (with INT4) | Speed |
| :--- | :--- | :--- | :--- |
| `0` | Disabled -- everything stays in VRAM | ~10-12 GB | Fastest |
| `3` | Balanced -- offloads idle components | ~8-10 GB | ~1.5-2x slower |
| `1` **(default)** | Maximum offload -- most aggressive swapping | ~5-6 GB | ~3-4x slower |

```bash
# Maximum VRAM savings (default in Docker)
MMGP_PROFILE=1 python gradio_app.py

# Balanced (good for 16GB GPUs with faster inference)
MMGP_PROFILE=1 python gradio_app.py --profile 3

# Disable offloading (plenty of VRAM available)
MMGP_PROFILE=0 python gradio_app.py --profile 0
```

The API (`api.py`) applies MMGP automatically after loading the runtime -- no CLI flags needed. Control via the `MMGP_PROFILE` environment variable.

```bash
# Override at runtime in Docker
docker run --gpus all -p 8080:8080 -e MMGP_PROFILE=3 hymotion-api
```

---

## Technique 3: Disable Prompt Engineering (saves ~4 GB)

The optional LLM prompt rewriter loads a second copy of Qwen for text rewriting and duration estimation. Disabling it saves ~4 GB and is the default in all entry points.

Controlled by `DISABLE_PROMPT_ENGINEERING`. Defaults to `True`.

```bash
# Disabled (default -- saves ~4GB)
DISABLE_PROMPT_ENGINEERING=True python gradio_app.py

# Enabled (requires extra ~4GB VRAM)
DISABLE_PROMPT_ENGINEERING=False python gradio_app.py
```

If you enable prompt engineering but need to save VRAM, use a smaller rewriter model:

```bash
# Qwen2-1.5B rewriter: ~1.5GB instead of ~4GB
DISABLE_PROMPT_ENGINEERING=False PROMPT_MODEL_PATH="Qwen/Qwen2-1.5B-Instruct" python gradio_app.py

# Run rewriter on CPU (0 GPU VRAM, slower, needs ~30GB RAM)
DISABLE_PROMPT_ENGINEERING=False PROMPT_CPU_MODE=true python gradio_app.py
```

---

## Technique 4: Disable Wooden Body Model (API only)

The API container sets `DISABLE_WOODEN_MESH=1` to skip loading the wooden body mesh model. The API returns raw motion data (`rot6d`, `transl`, `keypoints3d`) and the downstream app handles retargeting and visualization.

This is set automatically in the API Dockerfile and has no effect on motion quality.

---

## Technique 5: Constrain Generation Parameters

Smaller inputs = less activation memory during inference.

| Constraint | Effect | Set by |
| :--- | :--- | :--- |
| Max 30 words in prompt | Fewer tokens, less KV cache | Enforced in `gradio_app.py` |
| Short duration (< 5s) | Fewer frames, less diffusion memory | User / API caller |
| Single seed | 1 sample instead of multiple | Default in API |

---

## Why a smaller Qwen model won't work

The DiT denoiser has a hardcoded projection layer:

```python
# hymotion/network/hymotion_mmdit.py line 355
self.ctxt_encoder = nn.Linear(in_features=4096, out_features=512)
```

This `4096` matches the `hidden_size` of **Qwen3-8B specifically**. Smaller Qwen models have different hidden sizes:

| Model | `hidden_size` | Compatible? |
| :--- | :--- | :--- |
| Qwen3-8B | 4096 | Yes (trained with this) |
| Qwen3-4B | 2560 | No -- shape mismatch |
| Qwen3-1.7B | 2048 | No -- shape mismatch |
| Qwen3-0.6B | 1024 | No -- shape mismatch |

Swapping in a smaller Qwen would require retraining the DiT or adding a learned projection layer. Quantization and CPU offloading are the correct ways to reduce Qwen's footprint without retraining.

---

## Environment Variables Reference

### Core Settings (defaults applied automatically)

| Variable | Values | Default | Description |
| :--- | :--- | :--- | :--- |
| `QWEN_QUANTIZATION` | `int4`, `int8`, `none` | `int4` | Qwen3-8B quantization level |
| `DISABLE_PROMPT_ENGINEERING` | `True`, `False` | `True` | Disable LLM prompt rewriter (saves ~4GB) |
| `MMGP_PROFILE` | `0`, `1`, `3` | `1` | MMGP offloading aggressiveness |
| `MMGP_VERBOSE` | `0`, `1` | `1` | Logging level for MMGP |
| `USE_HF_MODELS` | `1`, `0` | `1` | Download models from Hugging Face |
| `DISABLE_WOODEN_MESH` | `1`, `0` | `1` (API) | Skip wooden body model loading |

### Prompt Engineering Settings (only when enabled)

| Variable | Values | Default | Description |
| :--- | :--- | :--- | :--- |
| `PROMPT_MODEL_PATH` | Model path/ID | `Qwen/Qwen3-8B` | Hugging Face model ID for prompt rewriter |
| `PROMPT_CPU_MODE` | `true`, `false` | `false` | Run prompt rewriter on CPU (~30GB RAM) |

### Docker / Cloud Run

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MODEL_PATH` | `/app/ckpts/tencent/HY-Motion-1.0-Lite` | Path to model checkpoint directory |
| `PORT` | `8080` | HTTP server port |

---

## Testing

```bash
cd /path/to/HY-Motion-1.0

# Test INT4 quantization (default)
python test_quantization.py

# Test INT8 quantization
QWEN_QUANTIZATION=int8 python test_quantization.py
```

Expected output (INT4 on RTX 4090, no MMGP):
```
Summary:
  - Quantization mode: int4
  - Model loaded successfully
  - Motion generation working
>>> Final GPU state:
============================================================
GPU 0 (NVIDIA GeForce RTX 4090):
  Allocated: 10.45GB / 24.00GB (43.5%)
  Reserved:  11.23GB / 24.00GB (46.8%)
============================================================
SUCCESS: Peak memory usage (10.45GB) is under 16GB!
```

With MMGP Profile 1, peak usage drops to ~5-6 GB.

---

## Troubleshooting

### "CUDA out of memory"

1. Verify `QWEN_QUANTIZATION=int4` is set (check container ENV or startup logs)
2. Enable MMGP: `MMGP_PROFILE=1`
3. Reduce duration and prompt length
4. Disable prompt engineering: `DISABLE_PROMPT_ENGINEERING=True`

### "BitsAndBytes not found"

```bash
pip uninstall bitsandbytes -y
pip install bitsandbytes>=0.41.0 --force-reinstall
```

### Slow inference

Expected with MMGP offloading. Profile 1 is ~3-4x slower than no offloading. Use Profile 3 for a balance, or Profile 0 if you have enough VRAM.

### Quality degradation with INT4

INT4 may reduce text encoding accuracy by ~5% for complex prompts. Solutions:
- Use INT8 if you have the VRAM headroom
- Use simpler, more direct prompts
- Increase `cfg_scale` (e.g., from 5.0 to 7.0)
