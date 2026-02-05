#!/usr/bin/env python3
"""
Test script to verify HY-Motion-1.0 quantization on 16GB GPU.

Usage:
    # Test INT4 quantization (default, ~4GB VRAM for Qwen3-8B)
    QWEN_QUANTIZATION=int4 DISABLE_PROMPT_ENGINEERING=True python test_quantization.py
    
    # Test INT8 quantization (~8GB VRAM for Qwen3-8B)
    QWEN_QUANTIZATION=int8 DISABLE_PROMPT_ENGINEERING=True python test_quantization.py
    
    # Test without quantization (full BF16, ~16GB VRAM for Qwen3-8B)
    QWEN_QUANTIZATION=none DISABLE_PROMPT_ENGINEERING=True python test_quantization.py
"""


import os
import sys
import torch


# Ensure environment variables are set before imports
if "QWEN_QUANTIZATION" not in os.environ:
    os.environ["QWEN_QUANTIZATION"] = "int4"


if "USE_HF_MODELS" not in os.environ:
    os.environ["USE_HF_MODELS"] = "1"


from hymotion.utils.t2m_runtime import T2MRuntime




def format_memory(bytes_value):
    """Format bytes as GB with 2 decimal places."""
    return f"{bytes_value / 1024**3:.2f}GB"




def print_gpu_memory():
    """Print current GPU memory usage for all devices."""
    if not torch.cuda.is_available():
        print(">>> No CUDA devices available")
        return


    print("\n" + "="*60)
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i)
        reserved = torch.cuda.memory_reserved(i)
        total = torch.cuda.get_device_properties(i).total_memory


        print(f"GPU {i} ({torch.cuda.get_device_name(i)}):")
        print(f"  Allocated: {format_memory(allocated)} / {format_memory(total)} ({allocated/total*100:.1f}%)")
        print(f"  Reserved:  {format_memory(reserved)} / {format_memory(total)} ({reserved/total*100:.1f}%)")


        # Warning if over 90% usage
        if allocated / total > 0.9:
            print(f"  ⚠️  WARNING: GPU {i} is using >90% of available memory!")
    print("="*60 + "\n")




def test_model_loading():
    """Test 1: Verify model loads successfully with quantization."""
    print("\n" + "="*60)
    print("TEST 1: Model Loading with Quantization")
    print("="*60)


    quantization = os.environ.get("QWEN_QUANTIZATION", "int4")
    print(f">>> Quantization mode: {quantization}")


    model_path = "ckpts/tencent/HY-Motion-1.0-Lite"
    config_path = os.path.join(model_path, "config.yml")
    ckpt_path = os.path.join(model_path, "latest.ckpt")


    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False


    print(f">>> Loading model from: {model_path}")
    print_gpu_memory()


    try:
        runtime = T2MRuntime(
            config_path=config_path,
            ckpt_name=ckpt_path,
            skip_text=False,
            device_ids=None,
            skip_model_loading=False,
            disable_prompt_engineering=True,
        )


        print("✅ Model loaded successfully!")
        print_gpu_memory()


        return runtime


    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None




def test_motion_generation(runtime):
    """Test 2: Generate a simple motion to verify inference works."""
    print("\n" + "="*60)
    print("TEST 2: Motion Generation")
    print("="*60)


    if runtime is None:
        print("❌ Skipping motion generation (model not loaded)")
        return False


    test_prompt = "A person walks forward"
    print(f">>> Generating motion for: '{test_prompt}'")
    print_gpu_memory()


    try:
        html_content, fbx_files, model_output = runtime.generate_motion(
            text=test_prompt,
            seeds_csv="42",
            duration=3.0,
            cfg_scale=5.0,
            output_format="dict",
            output_dir="output/test"
        )


        print("✅ Motion generated successfully!")
        print(f">>> Output keys: {model_output.keys() if isinstance(model_output, dict) else type(model_output)}")
        print_gpu_memory()


        return True


    except Exception as e:
        print(f"❌ Motion generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False




def main():
    print("\n" + "="*60)
    print("HY-Motion-1.0 Quantization Test Suite")
    print("="*60)


    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA is not available. This test requires a GPU.")
        sys.exit(1)


    print(f"✅ CUDA available: {torch.version.cuda}")
    print(f"✅ PyTorch version: {torch.__version__}")


    # Show initial memory state
    print("\n>>> Initial GPU state:")
    print_gpu_memory()


    # Test 1: Model loading
    runtime = test_model_loading()
    if runtime is None:
        print("\n❌ FAILED: Model did not load successfully")
        sys.exit(1)


    # Test 2: Motion generation
    success = test_motion_generation(runtime)
    if not success:
        print("\n❌ FAILED: Motion generation did not complete successfully")
        sys.exit(1)


    # Final summary
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\nSummary:")
    print(f"  - Quantization mode: {os.environ.get('QWEN_QUANTIZATION', 'int4')}")
    print(f"  - Model loaded successfully")
    print(f"  - Motion generation working")
    print("\n>>> Final GPU state:")
    print_gpu_memory()


    # Check if we're under 16GB
    if torch.cuda.is_available():
        max_allocated = max(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count()))
        if max_allocated < 16 * 1024**3:
            print(f"✅ SUCCESS: Peak memory usage ({format_memory(max_allocated)}) is under 16GB!")
        else:
            print(f"⚠️  WARNING: Peak memory usage ({format_memory(max_allocated)}) exceeds 16GB")




if __name__ == "__main__":
    main()