#!/usr/bin/env python3
"""
Download VAD and Turn Detection Models

This script downloads the required models:
1. Silero VAD v4 ONNX model
2. Smart Turn v3 ONNX model

Usage:
    python download_models.py

Requirements:
    pip install huggingface_hub
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from huggingface_hub import hf_hub_download, login
except ImportError:
    print("ERROR: huggingface_hub is not installed")
    print("Install it with: pip install huggingface_hub")
    sys.exit(1)


def download_silero_vad():
    """Download Silero VAD v4 ONNX model."""
    models_dir = Path(__file__).parent / "vad" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "silero_vad_v4.onnx"

    print("=" * 60)
    print("Downloading Silero VAD v4...")
    print("=" * 60)

    try:
        # Try alternative repository that has the ONNX model
        repo_id = "snakers4/silero-vad"

        # Download the ONNX model
        hf_hub_download(
            repo_id=repo_id,
            filename="models/silero_vad_v4.onnx",
            local_dir=str(models_dir.parent),
            local_dir_use_symlinks=False
        )

        print(f"✓ Silero VAD v4 downloaded to: {model_path}")
        print(f"  Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
        return True
    except Exception as e:
        print(f"✗ Failed to download Silero VAD: {e}")
        print("\nAlternative: Download manually from:")
        print("https://huggingface.co/snakers4/silero-vad/tree/main")
        print("Save as: vad/models/silero_vad_v4.onnx")
        return False


def download_smart_turn(device="cpu"):
    """Download Smart Turn v3 ONNX model."""
    models_dir = Path(__file__).parent / "vad" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    filename = f"smart-turn-v3.2-{device}.onnx"
    model_path = models_dir / filename

    print("=" * 60)
    print(f"Downloading Smart Turn v3 ({device})...")
    print("=" * 60)

    try:
        hf_hub_download(
            repo_id="pipecat-ai/smart-turn-v3",
            filename=filename,
            local_dir=str(models_dir),
            local_dir_use_symlinks=False
        )

        print(f"✓ Smart Turn v3 downloaded to: {model_path}")
        print(f"  Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
        return True
    except Exception as e:
        print(f"✗ Failed to download Smart Turn: {e}")
        print("\nAlternative: Download manually from:")
        print("https://huggingface.co/pipecat-ai/smart-turn-v3/tree/main")
        print(f"Save as: vad/models/{filename}")
        return False


def main():
    """Download all models."""
    print("\n" + "=" * 60)
    print("VAD and Turn Detection Model Downloader")
    print("=" * 60 + "\n")

    # Check device preference
    device = os.environ.get("SMART_TURN_DEVICE", "cpu").lower()
    if device not in ["cpu", "gpu"]:
        device = "cpu"

    results = []

    # Download Silero VAD
    results.append(("Silero VAD v4", download_silero_vad()))
    print()

    # Download Smart Turn
    results.append((f"Smart Turn v3 ({device})", download_smart_turn(device)))
    print()

    # Summary
    print("=" * 60)
    print("Download Summary")
    print("=" * 60)
    for name, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  {name}: {status}")
    print("=" * 60)

    all_success = all(r[1] for r in results)
    if all_success:
        print("\n✓ All models downloaded successfully!")
        print("\nYou can now use the VAD module:")
        print("  from vad import TurnDetector, create_turn_detector")
        print("  detector = create_turn_detector(threshold=0.70)")
        return 0
    else:
        print("\n✗ Some downloads failed. Please see above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
