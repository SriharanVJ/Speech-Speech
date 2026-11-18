"""
Configuration constants for VAD turn-taking system.
- Uses Silero VAD v4 and Smart Turn v3 from pipecat-ai
"""

from pathlib import Path
import os

_VAD_DIR = Path(__file__).parent
MODELS_DIR = _VAD_DIR / "models"

# Layer 1: Silero VAD v4
VAD_CONFIG = {
    "sample_rate": 16000,
    "threshold": 0.6,
    "energy_gate_rms": 0.015,
    "speech_trigger_ms": 200,
    "chunk_size": 512,
    "silero_model_path": MODELS_DIR / "silero_vad_v4.onnx",
}

# Layer 2: Smart-Turn v3 — semantic turn completion
TURN_DETECTION_CONFIG = {
    "vad_silence_trigger_ms": 280,
    "smart_turn_threshold": 0.70,
    "min_speech_duration_ms": 250,
    "buffer_duration_ms": 8000,
    "smart_turn_cooldown_ms": 250,
    "max_silence_timeout_ms": 1500,
    "smart_turn_num_threads": 4,
    "smart_turn_device": os.environ.get("SMART_TURN_DEVICE", "cpu").lower(),
    # Model path selected based on device
    "smart_turn_model_path": MODELS_DIR / f"smart-turn-v3.2-{os.environ.get('SMART_TURN_DEVICE', 'cpu').lower()}.onnx",
}
