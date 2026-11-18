"""
Smart-Turn v3 ONNX model for semantic turn completion detection.

Replaces the heuristic turn detector with pipecat-ai/smart-turn-v3,
a Whisper Tiny encoder + linear classifier (~8M params) that predicts
whether a speaker has finished their turn.

Reference: https://github.com/pipecat-ai/smart-turn
Model:     https://huggingface.co/pipecat-ai/smart-turn-v3

Input:  Up to 8s of 16kHz mono PCM -> Whisper mel spectrogram (1, 80, 800)
Output: Probability 0.0-1.0 that the speaker is done speaking.
"""

import logging
import time
import math
import numpy as np
import os
from pathlib import Path
from typing import Optional

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("[WARNING] onnxruntime not installed. Install with: pip install onnxruntime")

from .config import TURN_DETECTION_CONFIG

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_LENGTH = 8  # seconds


def truncate_or_pad_audio(audio: np.ndarray, n_seconds: int = CHUNK_LENGTH) -> np.ndarray:
    """
    Truncate to last n_seconds or zero-pad at beginning.

    Per smart-turn docs: "If the input data is shorter than 8 seconds,
    insert padding at the beginning so audio data is at the end."

    Args:
        audio: float32 array of 16kHz audio samples

    Returns:
        float32 array of exactly n_seconds * SAMPLE_RATE samples
    """
    max_samples = n_seconds * SAMPLE_RATE

    if len(audio) > max_samples:
        # Keep the LAST 8 seconds (most recent audio)
        return audio[-max_samples:]
    elif len(audio) < max_samples:
        # Zero-pad at the BEGINNING
        padding = max_samples - len(audio)
        return np.pad(audio, (padding, 0), mode='constant', constant_values=0)

    return audio


class SmartTurnModel:
    """
    Smart-Turn v3 ONNX model for semantic turn completion detection.

    Uses the ONNX model with Whisper-style mel spectrograms.
    Runs ONNX inference (~50ms) on CPU or GPU.

    Usage:
        model = SmartTurnModel()
        probability = model.predict(audio_pcm_16k)
        # probability > 0.5 -> speaker has finished their turn
    """

    def __init__(self, num_threads: int = 4):
        if not ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime is not installed. Install with: pip install onnxruntime")

        self.num_threads = num_threads
        self._session: Optional[ort.InferenceSession] = None
        self.feature_extractor = None

        # Load or download model
        self._load_model()

    def _load_model(self) -> None:
        """Load the ONNX model."""
        model_path = Path(TURN_DETECTION_CONFIG["smart_turn_model_path"])
        model_path.parent.mkdir(parents=True, exist_ok=True)

        if not model_path.exists():
            # Try to download from HuggingFace
            raise FileNotFoundError(
                f"Smart-Turn v3 model not found at {model_path}.\n"
                f"Please download it from:\n"
                f"  https://huggingface.co/pipecat-ai/smart-turn-v3\n\n"
                f"Download instructions:\n"
                f"1. Install huggingface_hub: pip install huggingface_hub\n"
                f"2. Run: python -c \"from huggingface_hub import hf_hub_download; "
                f"hf_hub_download('pipecat-ai/smart-turn-v3', 'smart-turn-v3.2-cpu.onnx', local_dir='./vad/models')\"\n"
                f"3. Or manually download from https://huggingface.co/pipecat-ai/smart-turn-v3/tree/main"
            )

        try:
            so = ort.SessionOptions()
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.inter_op_num_threads = 1
            so.intra_op_num_threads = self.num_threads
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Select provider based on device setting
            device = TURN_DETECTION_CONFIG.get("smart_turn_device", "cpu")
            if device == "gpu":
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=so,
                providers=providers,
            )

            # Try to load Whisper feature extractor
            try:
                from transformers import WhisperFeatureExtractor
                self.feature_extractor = WhisperFeatureExtractor(chunk_length=8)
            except ImportError:
                logger.warning("transformers not installed. Using fallback feature extraction.")
                self.feature_extractor = None

            active_provider = self._session.get_providers()[0]
            logger.info(
                f"[SmartTurn] Model loaded from {model_path} "
                f"(requested={device}, active={active_provider})"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Smart-Turn v3 model: {e}")

    def _extract_features_fallback(self, audio_pcm_16k: np.ndarray) -> np.ndarray:
        """
        Fallback feature extraction when transformers is not available.
        Computes Whisper-style mel spectrogram using numpy/scipy.
        """
        try:
            from scipy import signal
        except ImportError:
            raise ImportError("Either transformers or scipy is required for feature extraction")

        # Parameters for Whisper mel spectrogram
        n_fft = 400
        hop_length = 160
        n_mels = 80
        sample_rate = 16000

        # Compute STFT
        frequencies, times, spectrogram = signal.spectrogram(
            audio_pcm_16k,
            fs=sample_rate,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            mode='magnitude'
        )

        # Convert to mel scale
        mel_filter = signal.mel_frequencies(n_mels=n_mels, fmin=0, fmax=8000)
        # Simple approximation - in production you'd use librosa or transformers
        mel_spec = spectrogram[:n_mels, :]

        # Log scale
        mel_spec = np.log10(mel_spec + 1e-10)

        # Pad/truncate to expected shape (1, 80, 800)
        target_len = 800
        if mel_spec.shape[1] < target_len:
            pad_width = target_len - mel_spec.shape[1]
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec = mel_spec[:, :target_len]

        return mel_spec[np.newaxis, :, :].astype(np.float32)

    def predict(self, audio_pcm_16k: np.ndarray) -> float:
        """
        Predict turn completion probability from raw 16kHz PCM audio.

        Args:
            audio_pcm_16k: float32 array of 16kHz mono PCM audio.
                          Minimum 0.5 seconds (8000 samples) recommended.
                          Will be truncated to last 8 seconds if longer.

        Returns:
            Probability 0.0-1.0 that the speaker has finished their turn.
        """
        if self._session is None:
            raise RuntimeError("Smart-Turn model not loaded")

        t_start = time.time()

        # 1. Ensure float32
        if audio_pcm_16k.dtype != np.float32:
            audio_pcm_16k = audio_pcm_16k.astype(np.float32)
            # Normalize int16 range to [-1, 1] if needed
            if np.abs(audio_pcm_16k).max() > 2.0:
                audio_pcm_16k = audio_pcm_16k / 32768.0

        # 2. Truncate to last 8s or pad at beginning
        audio_padded = truncate_or_pad_audio(audio_pcm_16k)

        # 3. Compute Whisper features
        if self.feature_extractor is not None:
            try:
                inputs = self.feature_extractor(
                    audio_padded,
                    sampling_rate=16000,
                    return_tensors="np",
                    padding="max_length",
                    max_length=8 * 16000,
                    truncation=True,
                    do_normalize=True,
                )
                input_features = inputs.input_features.astype(np.float32)
            except Exception as e:
                logger.warning(f"Feature extraction failed: {e}, using fallback")
                input_features = self._extract_features_fallback(audio_padded)
        else:
            input_features = self._extract_features_fallback(audio_padded)

        # 4. ONNX inference (outputs raw logits)
        outputs = self._session.run(None, {"input_features": input_features})
        raw_score = float(outputs[0][0].item())

        # 5. Apply Sigmoid to convert to probability (0.0-1.0)
        probability = 1.0 / (1.0 + math.exp(-raw_score))

        elapsed_ms = (time.time() - t_start) * 1000
        logger.debug(f"[SmartTurn] inference={elapsed_ms:.1f}ms, logit={raw_score:.3f}, prob={probability:.3f}")

        return probability

    def get_stats(self) -> dict:
        """Get model status info."""
        return {
            "model_loaded": self._session is not None,
            "num_threads": self.num_threads,
        }


# Standalone function to download the model
def download_smart_turn_model():
    """Download Smart-Turn v3 model from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub")
        return False

    models_dir = Path(__file__).parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    device = os.environ.get("SMART_TURN_DEVICE", "cpu").lower()
    filename = f"smart-turn-v3.2-{device}.onnx"
    model_path = models_dir / filename

    print(f"Downloading Smart-Turn v3 model to {model_path}...")

    try:
        hf_hub_download(
            repo_id="pipecat-ai/smart-turn-v3",
            filename=filename,
            local_dir=str(models_dir),
            local_dir_use_symlinks=False
        )
        print(f"Model downloaded successfully to {model_path}")
        return True
    except Exception as e:
        print(f"Failed to download model: {e}")
        return False


if __name__ == "__main__":
    # Test download
    print("Testing Smart-Turn model download...")
    if download_smart_turn_model():
        print("Download successful!")
        # Test loading
        model = SmartTurnModel()
        print(f"Model loaded: {model.get_stats()}")
    else:
        print("Download failed!")
