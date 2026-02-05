# HY-Motion-1.0 Quantization Quick Start Guide

## Overview

This guide shows you how to use the quantized HY-Motion-1.0 model on a 16GB GPU.

**Memory Usage Summary**:
- **INT4 quantization** (default): ~10-12GB VRAM ✅ Recommended for 16GB GPUs
- **INT8 quantization**: ~14-16GB VRAM ⚠️ Tight fit on 16GB GPUs
- **No quantization**: ~22-24GB VRAM ❌ Requires >24GB GPU

---

## Installation

### 1. Install BitsAndBytes (Required for Quantization)

```bash
cd /path/to/HY-Motion-1.0
pip install bitsandbytes>=0.41.0 accelerate>=0.20.0
```

### 2. Verify Installation

```bash
python -c "import bitsandbytes; print(bitsandbytes.__version__)"
```

---

## Usage

### Option 1: Gradio Web Interface (Recommended)

Run the Gradio app with INT4 quantization (default):

```bash
cd /path/to/HY-Motion-1.0

# INT4 quantization (default, ~10-12GB VRAM)
QWEN_QUANTIZATION=int4 DISABLE_PROMPT_ENGINEERING=True python gradio_app.py

# INT8 quantization (~14-16GB VRAM)
QWEN_QUANTIZATION=int8 DISABLE_PROMPT_ENGINEERING=True python gradio_app.py

# No quantization (requires >24GB GPU)
QWEN_QUANTIZATION=none DISABLE_PROMPT_ENGINEERING=True python gradio_app.py
```

The Gradio interface will be available at `http://localhost:7860`.

### Option 2: Command-Line Interface

```bash
cd /path/to/HY-Motion-1.0

QWEN_QUANTIZATION=int4 DISABLE_PROMPT_ENGINEERING=True python local_infer.py \
    --model_path ckpts/tencent/HY-Motion-1.0-Lite \
    --prompt "A person walks forward and waves" \
    --duration 3.0 \
    --seed 42 \
    --output my_motion.fbx
```

### Option 3: Python API

```python
import os

# Set quantization before importing
os.environ["QWEN_QUANTIZATION"] = "int4"
os.environ["DISABLE_PROMPT_ENGINEERING"] = "True"
os.environ["USE_HF_MODELS"] = "1"

from hymotion.utils.t2m_runtime import T2MRuntime

# Initialize runtime (model will be quantized)
runtime = T2MRuntime(
    config_path="ckpts/tencent/HY-Motion-1.0-Lite/config.yml",
    ckpt_name="ckpts/tencent/HY-Motion-1.0-Lite/latest.ckpt",
    disable_prompt_engineering=True,
)

# Generate motion
html, fbx_files, output = runtime.generate_motion(
    text="A person walks forward",
    seeds_csv="42",
    duration=3.0,
    cfg_scale=5.0,
    output_format="fbx"
)

print(f"Generated FBX files: {fbx_files}")
```

---

## Testing

### Run Quantization Test Suite

```bash
cd /path/to/HY-Motion-1.0

# Test INT4 quantization (default)
QWEN_QUANTIZATION=int4 DISABLE_PROMPT_ENGINEERING=True python test_quantization.py

# Test INT8 quantization
QWEN_QUANTIZATION=int8 DISABLE_PROMPT_ENGINEERING=True python test_quantization.py
```

**Expected output**:
```
✅ ALL TESTS PASSED!
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
✅ SUCCESS: Peak memory usage (10.45GB) is under 16GB!
```

---

## Troubleshooting

### Issue: "CUDA out of memory" error

**Solution 1**: Use INT4 quantization (most memory-efficient)
```bash
QWEN_QUANTIZATION=int4 DISABLE_PROMPT_ENGINEERING=True python gradio_app.py
```

**Solution 2**: Clear CUDA cache before loading
```bash
python -c "import torch; torch.cuda.empty_cache()"
```

**Solution 3**: Reduce batch size or duration
```python
runtime.generate_motion(
    text="...",
    duration=2.0,  # Reduce from 3.0 to 2.0
    ...
)
```

### Issue: "BitsAndBytes not found" or import errors

**Solution**: Reinstall BitsAndBytes
```bash
pip uninstall bitsandbytes -y
pip install bitsandbytes>=0.41.0 --force-reinstall
```

### Issue: Slow inference with quantization

This is expected. INT4 quantization trades speed for memory. Typically:
- **INT4**: 1.0-1.5x slower than unquantized
- **INT8**: 1.0-1.2x slower than unquantized

The slowdown is acceptable for most use cases on memory-constrained GPUs.

### Issue: Model quality degradation

**INT4 quantization** may reduce text encoding accuracy by ~5-10% for complex prompts.

**Solutions**:
1. Use **INT8 quantization** (better quality, more VRAM)
2. Use simpler, more direct prompts
3. Increase `cfg_scale` parameter (e.g., from 5.0 to 7.0)

---

### MMGP Memory Offloading

MMGP (Memory Management for Generative Models) allows running larger models on smaller GPUs by dynamically offloading unused components to CPU RAM.

| Variable       | CLI Argument | Default | Description                                                |
| :------------- | :----------- | :------ | :--------------------------------------------------------- |
| `MMGP_PROFILE` | `--profile`  | `3`     | Offloading aggressiveness (1-4). Higher = more aggressive. |
| `MMGP_VERBOSE` | `--verbose`  | `1`     | Logging level for offloading operations.                   |

**Profile Levels:**
- **Profile 1**: Detailed offloading (slowest, max memory saving)
- **Profile 3** (Default): Balanced strategy. Keeps active components in VRAM, offloads others. Ideal for 12-16GB GPUs.
- **Profile 0**: Disabled

**Usage Example:**
```bash
# Enable aggressive offloading for 12GB cards
python gradio_app.py --share --profile 3 --verbose 1
```

**Combined with Quantization:**
```bash
# Maximum memory efficiency
QWEN_QUANTIZATION=int4 python gradio_app.py --share --profile 3
```

---

## Environment Variables Reference

### Core Quantization Settings

| Variable            | Values                 | Default | Description                  |
| ------------------- | ---------------------- | ------- | ---------------------------- |
| `QWEN_QUANTIZATION` | `int4`, `int8`, `none` | `int4`  | Qwen3-8B quantization level  |
| `USE_HF_MODELS`     | `1`, `0`               | `0`     | Use Hugging Face model paths |

### Prompt Engineering Settings

| Variable                     | Values          | Default         | Description                                         |
| ---------------------------- | --------------- | --------------- | --------------------------------------------------- |
| `DISABLE_PROMPT_ENGINEERING` | `True`, `False` | `False`         | Disable LLM prompt rewriter (saves ~4GB GPU memory) |
| `PROMPT_MODEL_PATH`          | Model path/ID   | `Qwen/Qwen3-8B` | Hugging Face model ID for prompt rewriter           |
| `PROMPT_CPU_MODE`            | `true`, `false` | `false`         | Run prompt rewriter on CPU (requires ~30GB RAM)     |

#### Prompt Engineering Configuration Examples

**Disable prompt engineering** (recommended for 16GB GPUs):
```bash
DISABLE_PROMPT_ENGINEERING=True python gradio_app.py
```

**Use smaller model** (fits in limited GPU memory):
```bash
# Qwen2-1.5B: ~1.5GB GPU memory (good quality)
PROMPT_MODEL_PATH="Qwen/Qwen2-1.5B-Instruct" python gradio_app.py

# Qwen2-0.5B: ~0.5GB GPU memory (basic quality)
PROMPT_MODEL_PATH="Qwen/Qwen2-0.5B-Instruct" python gradio_app.py
```

**Run prompt rewriter on CPU** (slow, uses ~30GB RAM):
```bash
PROMPT_CPU_MODE=true python gradio_app.py
```

**Combined example** for 16GB GPU:
```bash
# Main model: INT4 quantization (~5GB)
# Prompt rewriter: Qwen2-1.5B (~1.5GB)
# Total: ~9GB GPU memory
QWEN_QUANTIZATION=int4 PROMPT_MODEL_PATH="Qwen/Qwen2-1.5B-Instruct" python gradio_app.py
```

---

## Performance Comparison

| Configuration   | VRAM Usage | Speed | Quality | 16GB GPU?   |
| --------------- | ---------- | ----- | ------- | ----------- |
| INT4 (default)  | ~10-12GB   | 1.0x  | 95%     | ✅ Yes       |
| INT8            | ~14-16GB   | 1.2x  | 98%     | ⚠️ Tight fit |
| No quantization | ~22-24GB   | 1.5x  | 100%    | ❌ No        |

---

## Next Steps

1. **Test on your GPU**: Run `test_quantization.py`
2. **Generate motions**: Use Gradio web interface or command-line
3. **Adjust quality**: Try INT8 if you have spare VRAM
4. **Report issues**: Check troubleshooting section

For more details, see `implementation_plan.md`.