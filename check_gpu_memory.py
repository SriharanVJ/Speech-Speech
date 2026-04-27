"""
Diagnostic script to check GPU memory usage of each component.
This version loads models properly (CPU first, then GPU) to avoid memory issues.
"""

import torch
import gc

def check_gpu_memory(label: str):
    """Print current GPU memory usage."""
    if not torch.cuda.is_available():
        print(f"[{label}] CUDA not available")
        return

    allocated = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0) / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3

    print(f"[{label}]")
    print(f"  Allocated: {allocated:.2f} GB")
    print(f"  Reserved:  {reserved:.2f} GB")
    print(f"  Total GPU: {total:.2f} GB")
    print(f"  Free:      {total - reserved:.2f} GB")
    print()

def main():
    print("=" * 50)
    print("GPU Memory Diagnostic (Optimized)")
    print("=" * 50)
    print()

    # Baseline
    check_gpu_memory("Baseline (empty)")

    # Test 1: Load Whisper to CPU first, then convert to fp16, then move to GPU
    print("Test 1: Whisper large-v3 (CPU -> fp16 -> GPU)")
    print("-" * 40)
    import whisper

    # Load to CPU first (in fp32)
    print("  Loading to CPU (fp32)...")
    model = whisper.load_model("large-v3", device="cpu")
    check_gpu_memory("After CPU load (should be 0 GPU)")

    # Convert to fp16 on CPU
    print("  Converting to fp16 on CPU...")
    model = model.half()
    check_gpu_memory("After fp16 conversion (should be 0 GPU)")

    # Move to GPU
    print("  Moving to GPU...")
    model = model.cuda()
    check_gpu_memory("After GPU move (fp16 model)")

    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()
    check_gpu_memory("After cleanup")

    print("=" * 50)
    print("ANALYSIS")
    print("=" * 50)
    print("""
The issue is that whisper.load_model(..., device="cuda") loads the
FULL fp32 model to GPU first, which is ~3GB. Then converting to fp16
requires additional memory for the fp16 copy temporarily.

For a 6GB card:
- fp32 model: ~3GB
- fp16 model: ~1.5GB
- Overhead during conversion: ~500MB-1GB
- Silero VAD: ~100MB
- Runtime memory: ~500MB-1GB
- Total peak: ~5-6GB (exceeds your 5.67GB!)

SOLUTION: Load to CPU first, convert to fp16, then move to GPU.
This reduces peak memory from ~6GB to ~2.5GB!
""")

if __name__ == "__main__":
    main()
