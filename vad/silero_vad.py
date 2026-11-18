"""
Silero VAD v4 wrapper using ONNX Runtime.

This is a drop-in replacement for the torch.hub Silero VAD.
Uses the ONNX model for faster inference and better compatibility.
"""

import logging
import time
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Optional

from .config import VAD_CONFIG

logger = logging.getLogger(__name__)


class SileroVAD:
    """Silero VAD v4 wrapper for speech activity detection using ONNX."""

    MODEL_RESET_STATES_TIME = 5.0

    def __init__(self, model_path: Optional[str] = None, threshold: float = 0.5):
        """
        Initialize Silero VAD with ONNX model.

        Args:
            model_path: Path to silero_vad_v4.onnx model file
            threshold: Speech probability threshold (default 0.5)
        """
        if model_path is None:
            model_path = str(VAD_CONFIG["silero_model_path"])

        self.threshold = threshold
        self.model_path = Path(model_path)
        self._last_reset_time = time.time()

        # Initialize ONNX session
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
            sess_options=opts
        )

        self._init_states()

        logger.info(f"[SileroVAD] Loaded ONNX model from {self.model_path}")

    def _init_states(self):
        """Initialize VAD state buffers for v4."""
        self._h = np.zeros((2, 1, 64), dtype=np.float32)
        self._c = np.zeros((2, 1, 64), dtype=np.float32)

    def maybe_reset(self):
        """Reset VAD state if too much time has passed."""
        if (time.time() - self._last_reset_time) >= self.MODEL_RESET_STATES_TIME:
            self._init_states()
            self._last_reset_time = time.time()

    def prob(self, chunk_f32: np.ndarray) -> float:
        """
        Get speech probability for audio chunk.

        Args:
            chunk_f32: float32 audio chunk, 16kHz, expected to be 512 samples

        Returns:
            float: Speech probability (0.0 to 1.0)
        """
        x = np.reshape(chunk_f32, (1, -1))
        if x.shape[1] != 512:
            return 0.0

        ort_inputs = {
            "input": x.astype(np.float32),
            "sr": np.array([16000], dtype=np.int64),
            "h": self._h,
            "c": self._c,
        }

        try:
            out, hn, cn = self.session.run(None, ort_inputs)
            self._h = hn
            self._c = cn
        except Exception as e:
            logger.error(f"[SileroVAD] ONNX inference failed: {e}")
            return 0.0

        self.maybe_reset()
        return float(out[0][0])

    def is_speech(self, audio: np.ndarray) -> bool:
        """
        Detect if audio frame contains speech.

        Args:
            audio: Audio frame (int16 or float32)

        Returns:
            bool: True if speech detected
        """
        # Convert to float32 if needed
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Normalize if needed
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / 32768.0

        # Pad or trim to 512 samples
        if len(audio) < 512:
            audio = np.pad(audio, (0, 512 - len(audio)), mode='constant')
        elif len(audio) > 512:
            # Take the first 512 samples
            audio = audio[:512]

        # Get speech probability
        prob = self.prob(audio)
        return prob > self.threshold

    def reset(self) -> None:
        """Reset VAD state."""
        self._init_states()
        self._last_reset_time = time.time()


def download_silero_vad_model():
    """Download Silero VAD v4 ONNX model from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub")
        return False

    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "silero_vad_v4.onnx"

    print(f"Downloading Silero VAD v4 model to {model_path}...")

    try:
        # Download from snakers4/silero-vad repository
        hf_hub_download(
            repo_id="snakers4/silero-vad",
            filename="models/silero_vad_v4.onnx",
            local_dir=str(models_dir.parent),
            local_dir_use_symlinks=False
        )
        print(f"Model downloaded successfully to {model_path}")
        return True
    except Exception as e:
        print(f"Failed to download model: {e}")
        print("\nAlternative: Download manually from:")
        print("https://huggingface.co/snakers4/silero-vad/tree/main")
        return False


if __name__ == "__main__":
    # Test download
    print("Testing Silero VAD v4 model download...")
    if download_silero_vad_model():
        print("Download successful!")
        # Test loading
        vad = SileroVAD()
        print(f"VAD loaded successfully")
    else:
        print("Download failed!")
