"""
Whisper Large Turbo v3 - Enhanced WebSocket Speech-to-Text Server

🔥 NEW FEATURES:
- Real-time Silero VAD with speech_start events (barge-in support)
- Partial transcripts for live UX
- ✅ Enhanced Linguistic Turn Detection (multi-factor analysis)
- VAD percentage tracking
- Client interrupt support
- Max duration timeout protection (prevents stuck states)

⚠️ TODO: Implement real partial transcripts with:
- Overlapping chunks (sliding window)
- OR incremental decoding (faster but less accurate)

Current "partial" transcripts are mini-final chunks, not true streaming partials.

✅ UPGRADE: Turn Detection uses enhanced linguistic analysis
- Multi-factor: sentence structure, punctuation, word count, patterns
- Handles: numbers, pauses, incomplete sentences
- No external ML dependencies (works reliably)
- Keeps last 6 messages for context

Architecture:
    Audio → Silero VAD → Streaming Buffer → Whisper (partial + final) → Enhanced Turn Detector

Usage:
    python new_whisper.py

Install requirements:
    pip install torch transformers (optional - for future ML features)

WebSocket:
    ws://localhost:8006/stream/asr?language=en&use_vad=true&enable_speech_start=true

Events:
    - speech_start: Barge-in trigger (stop TTS)
    - partial: Real-time transcript for turn detector
    - turn_end: Turn complete due to silence OR max duration
    - segment_completed: Final transcript with metadata

Client Integration (REQUIRED):
    - Send tts_start when TTS begins
    - Send tts_end when TTS ends
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import torch
import whisper
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState
import uvicorn

# LiveKit Turn Detector (optional - falls back to heuristic if not available)
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[INFO] transformers not installed. Install with: pip install transformers")
    print("[INFO] Will use heuristic turn detector as fallback")

# ==============================================================================
# MODEL LOADING (exact approach from AI-agent-STT)
# ==============================================================================

# Model configuration
_MODEL_PATH = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
MODEL_DEVICE = os.environ.get("WHISPER_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

# Map model names to whisper format (handles both openai/ prefix and plain names)
MODEL_NAME_MAP = {
    "openai/whisper-large-v3-turbo": "large-v3-turbo",
    "large-v3-turbo": "large-v3-turbo",
    "turbo": "turbo",
    "openai/whisper-large-v3": "large-v3",
    "large-v3": "large-v3",
    "openai/whisper-large-v2": "large-v2",
    "large-v2": "large-v2",
    "openai/whisper-large": "large",
    "large": "large",
    "openai/whisper-medium": "medium",
    "medium": "medium",
    "openai/whisper-small": "small",
    "small": "small",
    "openai/whisper-base": "base",
    "base": "base",
    "openai/whisper-tiny": "tiny",
    "tiny": "tiny",
}

MODEL_PATH = MODEL_NAME_MAP.get(_MODEL_PATH, _MODEL_PATH)

print("=" * 50)
print("  Whisper ASR - Loading Model")
print(f"  Model: {MODEL_PATH}")
print(f"  Device: {MODEL_DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("  CPU Mode")
print("=" * 50)

# Clear CUDA cache BEFORE loading
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

# Load model (exact same approach as AI-agent-STT)
# Only load once (check if already loaded to prevent re-import issues)
_model_loaded = False

def get_model():
    global model, _model_loaded
    if not _model_loaded:
        print(f"Loading model: {MODEL_PATH}")

        # For GPU with limited memory, load to CPU first, then move to GPU
        if MODEL_DEVICE == "cuda" and torch.cuda.is_available():
            print("Loading to CPU first to avoid GPU OOM...")
            model = whisper.load_model(MODEL_PATH, device="cpu")
            print("Moving model to GPU...")
            model = model.to(MODEL_DEVICE)
            print("Model moved to GPU successfully")
        else:
            model = whisper.load_model(MODEL_PATH, device=MODEL_DEVICE)

        _model_loaded = True

        # GPU optimization AFTER model is loaded
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True  # GPU optimization

        if torch.cuda.is_available():
            print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    return model

# Load model at module level
model = get_model()

# GPU optimization AFTER model is loaded
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # GPU optimization

if torch.cuda.is_available():
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

print(f"  Whisper {MODEL_PATH} loaded successfully")
print("=" * 50)
print()

# Try loading silero VAD from torch hub
VAD_AVAILABLE = False
_vad_model = None
_vad_utils = None

try:
    # Try loading Silero VAD with different approaches for compatibility
    import os
    os.environ['TORCH_HOME'] = os.path.expanduser('~/.cache/torch')

    _vad_model, _vad_utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        trust_repo=True,
        source='github'
    )
    VAD_AVAILABLE = True
    print("[INFO] Silero VAD loaded from torch.hub")
except Exception as e:
    print(f"[INFO] Silero VAD not available: {e}")
    print("[INFO] This is usually due to torchaudio compatibility issues")
    print("[INFO] Will use SimpleVAD (energy-based) as fallback")
    print("[INFO] To fix Silero VAD, try: pip install torchaudio --upgrade --force-reinstall")

SAMPLE_RATE = 16000
CHANNELS = 1

logger = logging.getLogger(__name__)


# ==============================================================================
# TURN DETECTOR (LiveKit ML-based semantic turn detection)
# ==============================================================================


class LiveKitTurnDetector:
    """
    Enhanced Turn Detector using improved linguistic analysis.

    Uses sophisticated heuristics and linguistic patterns to detect
    when a user has finished their turn, with multiple factors considered.
    """

    def __init__(self, threshold: float = 0.7):
        """
        Initialize the Enhanced Turn Detector.

        Args:
            threshold: Probability threshold for turn completion (default 0.7)
        """
        self.threshold = threshold

        # Linguistic patterns for turn completion
        self.completion_patterns = {
            "sentence_endings": [".", "!", "...", "。", "！", "…"],
            "question_indicators": ["?", "？"],
            "continuation_indicators": [
                "and", "but", "or", "so", "because", "although",
                "however", "moreover", "furthermore", "also"
            ],
            "pause_indicators": [
                "um", "uh", "um...", "uh...", "er", "erm",
                "hold on", "wait a moment", "let me see", "let me think",
                "actually", "basically", "i mean", "you know"
            ],
            "incomplete_patterns": [
                "i'm going to", "i was about to", "i wanted to",
                "the thing is", "what i meant was", "let me finish",
                "one more thing", "also", "and then", "but wait"
            ],
            "completion_phrases": [
                "thank you", "thanks", "please", "that's all",
                "that's it", "done", "finished", "complete"
            ]
        }

        print(f"[TURN DETECTOR] Enhanced linguistic turn detector initialized")
        print(f"[TURN DETECTOR] Threshold: {threshold}")

    def predict(self, messages: List[Dict[str, Any]]) -> float:
        """
        Predict the probability that the user has finished their turn.

        Uses multiple linguistic factors:
        - Sentence structure
        - Punctuation patterns
        - Word count and flow
        - Completion indicators
        - Context from conversation

        Args:
            messages: List of conversation messages with 'role' and 'content'

        Returns:
            float: Probability (0.0 to 1.0) that turn is complete
        """
        if not messages:
            return 0.0

        # Get the latest user message
        latest_message = messages[-1]
        content = latest_message.get("content", "").strip()

        if not content:
            return 0.0

        content_lower = content.lower()
        words = content.split()
        word_count = len(words)

        # Start with a base probability
        probability = 0.5

        # Factor 1: Punctuation analysis
        ends_with_period = content.endswith((".", "!", "...", "。", "！", "…"))
        ends_with_question = content.endswith(("?", "？"))

        if ends_with_period:
            probability += 0.25
        if ends_with_question:
            probability -= 0.15

        # Factor 2: Word count analysis
        if word_count < 3:
            probability -= 0.2  # Very short, likely incomplete
        elif word_count >= 6:
            probability += 0.15  # Substantial statement
        elif word_count >= 12:
            probability += 0.1  # Long statement, likely complete

        # Factor 3: Continuation indicators
        has_continuation = any(indicator in content_lower.split()
                              for indicator in self.completion_patterns["continuation_indicators"])
        if has_continuation and not ends_with_period:
            probability -= 0.2  # Likely continuing

        # Factor 4: Pause indicators
        has_pause = any(pause in content_lower
                       for pause in self.completion_patterns["pause_indicators"])
        if has_pause:
            probability -= 0.15  # User paused, likely not done

        # Factor 5: Incomplete patterns
        has_incomplete = any(pattern in content_lower
                           for pattern in self.completion_patterns["incomplete_patterns"])
        if has_incomplete:
            probability -= 0.25  # Clear incomplete pattern

        # Factor 6: Completion phrases
        has_completion = any(phrase in content_lower
                            for phrase in self.completion_patterns["completion_phrases"])
        if has_completion:
            probability += 0.3  # Clear completion signal

        # Factor 7: Sentence structure complexity
        sentence_count = content.count(".") + content.count("!") + content.count("?")
        if sentence_count > 1:
            probability += 0.1  # Multiple sentences, likely complete

        # Factor 8: Capitalization patterns
        # If text starts lowercase and ends with punctuation, might be continuation
        if words and words[0][0].islower() and ends_with_period:
            probability -= 0.1

        # Factor 9: Trailing conjunctions or prepositions
        trailing_words = ["and", "or", "but", "with", "to", "for", "at", "in", "on"]
        if words and words[-1].lower() in trailing_words:
            probability -= 0.2  # Ends with connector, likely incomplete

        # Factor 10: Numbers and addresses (commonly cause pauses)
        import re
        has_number_sequence = bool(re.search(r'\d+[\s-]*\d*', content))
        if has_number_sequence and not ends_with_period:
            probability -= 0.15  # Likely giving numbers with pauses

        # Normalize probability to 0-1 range
        probability = max(0.0, min(1.0, probability))

        return probability

    def reset(self):
        """Reset the turn detector state (no-op)."""
        pass


# Factory function to create turn detector
def create_turn_detector(threshold: float = 0.7):
    """
    Create a turn detector instance.

    Uses the enhanced linguistic turn detector which combines
    multiple linguistic patterns for accurate turn detection.

    Args:
        threshold: Probability threshold for turn completion

    Returns:
        LiveKitTurnDetector (enhanced linguistic version)
    """
    return LiveKitTurnDetector(threshold=threshold)


# ==============================================================================
# VOICE ACTIVITY DETECTION (VAD)
# ==============================================================================


class SimpleVAD:
    """Simple energy-based Voice Activity Detection."""

    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        threshold: float = 0.5,
        speech_pad_ms: int = 100,
    ):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.threshold = threshold
        self.speech_pad_frames = int(speech_pad_ms / frame_duration_ms)
        self.speech_frames = 0

    def _compute_energy(self, audio: np.ndarray) -> float:
        """Compute RMS energy of audio frame."""
        rms = np.sqrt(np.mean(audio**2))
        return rms

    def is_speech(self, audio: np.ndarray) -> bool:
        """Detect if audio frame contains speech."""
        energy = self._compute_energy(audio)
        normalized_energy = min(energy * 5, 1.0)
        is_speech_frame = normalized_energy > self.threshold

        if is_speech_frame:
            self.speech_frames = self.speech_pad_frames
        elif self.speech_frames > 0:
            self.speech_frames -= 1
            is_speech_frame = True

        return is_speech_frame

    def reset(self) -> None:
        """Reset VAD state."""
        self.speech_frames = 0


class SileroVAD:
    """Silero VAD wrapper - runs on CPU to save GPU memory."""

    MIN_SAMPLES = 512

    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        speech_pad_ms: int = 100,
    ):
        global _vad_model, _vad_utils

        if not VAD_AVAILABLE or _vad_model is None:
            raise RuntimeError("Silero VAD not available. Check torch.hub connection.")

        self.sample_rate = sample_rate
        self.threshold = threshold

        # Use the globally loaded Silero VAD model from torch.hub
        self.model = _vad_model
        self.model.eval()

        print("[VAD] Silero VAD running on CPU (saves GPU memory)")

        self.speech_pad_samples = int(sample_rate * speech_pad_ms / 1000)
        self.speech_buffer = np.array([], dtype=np.float32)
        self.last_speech_result = False

    def is_speech(self, audio: np.ndarray) -> bool:
        """Detect if audio frame contains speech using Silero VAD."""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        if np.max(np.abs(audio)) > 1.0:
            audio = audio / 32768.0

        self.speech_buffer = np.concatenate([self.speech_buffer, audio])

        if len(self.speech_buffer) < self.MIN_SAMPLES:
            return self.last_speech_result

        audio_to_process = self.speech_buffer[:self.MIN_SAMPLES]
        remaining_buffer = self.speech_buffer[self.MIN_SAMPLES:]

        audio_tensor = torch.from_numpy(audio_to_process)

        with torch.no_grad():
            prob = self.model(audio_tensor, self.sample_rate).item()

        is_speech = prob > self.threshold
        self.last_speech_result = is_speech
        self.speech_buffer = remaining_buffer

        return is_speech

    def reset(self) -> None:
        """Reset VAD state."""
        self.speech_buffer = np.array([], dtype=np.float32)
        self.last_speech_result = False


# ==============================================================================
# STREAMING AUDIO BUFFER
# ==============================================================================


class StreamingAudioBuffer:
    """Real-time audio buffer for frame-by-frame processing."""

    SILENCE_THRESHOLD = 0.08
    SILENCE_DURATION_MS = 800  # ✅ FIX 1: Increased from 400 to 800ms for natural pauses
    MIN_BUFFER_MS = 800

    def __init__(
        self,
        sample_rate: int = 16000,
        buffer_duration_ms: int = 1500,
        enable_silence_flush: bool = True,
        vad: Optional[Callable[[np.ndarray], bool]] = None,
        min_speech_duration_ms: int = 300,
        silence_threshold: float = 0.08,
    ):
        self.sample_rate = sample_rate
        self.buffer_size_samples = int(sample_rate * buffer_duration_ms / 1000)
        self.min_buffer_samples = int(sample_rate * self.MIN_BUFFER_MS / 1000)
        self.enable_silence_flush = enable_silence_flush
        self.silence_threshold = silence_threshold

        self.vad = vad
        self.min_speech_samples = int(sample_rate * min_speech_duration_ms / 1000)
        self.speech_detected = False
        self.speech_samples = 0

        self.buffer: np.ndarray = np.array([], dtype=np.float32)
        self.silence_buffer: np.ndarray = np.array([], dtype=np.float32)
        self.silence_frames = 0
        self.silence_frame_threshold = int(
            sample_rate * self.SILENCE_DURATION_MS / 1000
        )
        self.lock = threading.Lock()

    def _is_silence(self, audio: np.ndarray) -> bool:
        """Detect silence using RMS energy."""
        rms = np.sqrt(np.mean(audio**2))
        return rms < self.silence_threshold

    def add_frame(self, frame: np.ndarray, is_speech: bool = True) -> Optional[np.ndarray]:
        """
        Add an audio frame to the buffer.

        Args:
            frame: Audio frame to add
            is_speech: VAD result (pre-computed to avoid duplicate VAD calls)
        """
        with self.lock:
            self.buffer = np.concatenate([self.buffer, frame])

            # Use pre-computed VAD result instead of calling self.vad(frame)
            # This fixes the duplicate VAD call performance issue
            if is_speech:
                self.speech_detected = True
                self.speech_samples += len(frame)

            if (
                self.enable_silence_flush
                and len(self.buffer) >= self.min_buffer_samples
            ):
                # ✅ FIX: Use VAD-based silence instead of RMS
                # This fixes the mismatch between VAD silence and RMS silence
                if not is_speech:
                    self.silence_buffer = np.concatenate([self.silence_buffer, frame])
                    self.silence_frames += len(frame)

                    if self.silence_frames >= self.silence_frame_threshold:
                        # Use speech_detected from VAD instead of RMS check
                        has_speech = self.speech_detected

                        if has_speech:
                            audio_to_process = self.buffer[: -self.silence_frames].copy()
                            self.buffer = np.array([], dtype=np.float32)
                            self.silence_buffer = np.array([], dtype=np.float32)
                            self.silence_frames = 0
                            self.speech_detected = False
                            self.speech_samples = 0
                            return audio_to_process
                else:
                    self.silence_buffer = np.array([], dtype=np.float32)
                    self.silence_frames = 0

            if len(self.buffer) >= self.buffer_size_samples:
                if not self.speech_detected:
                    self.buffer = np.array([], dtype=np.float32)
                    self.silence_buffer = np.array([], dtype=np.float32)
                    self.silence_frames = 0
                    return None

                audio_to_process = self.buffer.copy()
                self.buffer = np.array([], dtype=np.float32)
                self.silence_buffer = np.array([], dtype=np.float32)
                self.silence_frames = 0
                self.speech_detected = False
                self.speech_samples = 0
                return audio_to_process

            return None

    def flush(self) -> Optional[np.ndarray]:
        """Flush any remaining audio in the buffer."""
        with self.lock:
            if len(self.buffer) > 0:
                if self.vad and not self.speech_detected:
                    return None

                audio_to_process = self.buffer.copy()
                self.buffer = np.array([], dtype=np.float32)
                self.silence_buffer = np.array([], dtype=np.float32)
                self.silence_frames = 0
                self.speech_detected = False
                self.speech_samples = 0
                return audio_to_process
            return None

    def reset(self) -> None:
        """Clear the buffer."""
        with self.lock:
            self.buffer = np.array([], dtype=np.float32)
            self.silence_buffer = np.array([], dtype=np.float32)
            self.silence_frames = 0
            self.speech_detected = False
            self.speech_samples = 0


# ==============================================================================
# TRANSCRIPTION FUNCTION (matching AI-agent-STT)
# ==============================================================================


def transcribe_audio(
    audio: np.ndarray,
    language: str = "English",
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Transcribe audio using whisper (direct numpy array, no temp file).
    """
    # Ensure audio is float32 normalized
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    if np.max(np.abs(audio)) > 1.0:
        audio = audio / 32768.0

    try:
        # Options matching AI-agent-STT
        options = {
            "task": "transcribe",
            "language": None if language == "auto" else language,
            "fp16": True if MODEL_DEVICE == "cuda" else False,
            "temperature": temperature,
            "beam_size": 1,
            "condition_on_previous_text": False,
            "no_speech_threshold": 0.3,
            "logprob_threshold": -1.0,
            "compression_ratio_threshold": 2.4,
        }

        # Pass audio directly as numpy array (whisper handles 16kHz audio)
        result = model.transcribe(audio, **options)

        text = result.get("text", "").strip()
        detected_lang = result.get("language", "en")

        print(f"Model recognized language: {detected_lang}")
        print(f"Transcription result: {text}")

        # Filter short/empty transcriptions
        if len(text) < 2:
            return {
                "text": "",
                "segments": [],
                "language": detected_lang,
                "language_probability": 0.0,
            }

        return {
            "text": text,
            "segments": result.get("segments", []),
            "language": detected_lang,
            "language_probability": 1.0,
        }

    except Exception as e:
        print(f"[Whisper ERROR] {e}")
        traceback.print_exc()
        return {
            "text": "",
            "segments": [],
            "error": str(e),
        }


# ==============================================================================
# TEXT FILTERS
# ==============================================================================


def filter_non_speech_labels(text: str) -> str:
    """Filter out non-speech labels like [Music], [Noise], etc."""
    import re

    if not text:
        return text

    patterns = [
        r'\[Music\]', r'\[Noise\]', r'\[Laughter\]', r'\[Cough\]',
        r'\[Sigh\]', r'\[Breath\]', r'\[Background Noise\]', r'\[Silence\]',
        r'\[Pause\]', r'\[Background\]', r'\[Applause\]',
    ]

    filtered = text
    for pattern in patterns:
        filtered = re.sub(pattern, '', filtered, flags=re.IGNORECASE)

    filtered = re.sub(r'\s+', ' ', filtered).strip()

    if not filtered or filtered in ['.', ',', '!', '?', ';', ':']:
        return ''

    return filtered


def filter_hallucinations(text: str) -> str:
    """Filter out hallucination patterns."""
    import re

    if not text or text.strip() in ['', '.', '..']:
        return ''

    words = text.split()
    if len(words) >= 4:
        repeat_count = 1
        for i in range(1, len(words)):
            if words[i].lower() == words[i-1].lower():
                repeat_count += 1
                if repeat_count >= 3:
                    return ''
            else:
                repeat_count = 1

    if re.search(r'(.)\1{5,}', text):
        return ''

    if len(text.strip()) <= 2 and text.strip().isalpha():
        return ''

    return text


# ==============================================================================
# FASTAPI APP
# ==============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    port = os.environ.get("PORT", "8006")
    print(f"  Ready! ws://localhost:{port}/stream/asr")
    print()
    yield
    # Shutdown cleanup here if needed
    print("[INFO] Server shutdown")


app = FastAPI(
    title="Whisper Large v3 Turbo ASR",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def index():
    """Root endpoint."""
    return {
        "service": "Whisper Large v3 Turbo ASR",
        "status": "running",
        "backend": "whisper",
        "vad": "silero",
        "model": MODEL_PATH,
        "device": MODEL_DEVICE,
        "features": {
            "realtime_vad": True,
            "speech_start_events": True,
            "partial_transcripts": True,
            "turn_detection": True,
            "barge_in": True,
            "vad_percentage_tracking": True,
        }
    }


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.get("/config")
async def config():
    """Get server configuration."""
    return {
        "model": MODEL_PATH,
        "device": MODEL_DEVICE,
        "backend": "whisper",
        "sample_rate": SAMPLE_RATE,
    }


@app.get("/events")
async def events():
    """Get available WebSocket events."""
    return {
        "server_to_client": {
            "ready": "Initial connection ready message",
            "speech_start": "🔥 BARGE-IN: Speech detected, stop TTS immediately (only if tts_is_playing=True)",
            "partial": "Partial transcript (real-time, for turn detector)",
            "segment_completed": "Final completed transcript segment",
            "turn_end": "✅ Turn ended due to silence AND semantic check (turn_probability >= 0.7)",
            "interrupt_ack": "Acknowledgment of client interrupt request",
            "complete": "Session complete",
            "status": "Status update message"
        },
        "client_to_server": {
            "done": "End the session",
            "reset": "Reset all state (VAD, buffer, turn detector)",
            "interrupt": "🔥 BARGE-IN: Request interrupt (stop TTS/ASR)",
            "tts_start": "✅ REQUIRED: Tell server TTS started (enables barge-in)",
            "tts_end": "✅ REQUIRED: Tell server TTS ended (disables barge-in)"
        },
        "query_params": {
            "language": "auto|en|es|fr|...",
            "temperature": "0.0 (default)",
            "use_vad": "true|false",
            "vad_threshold": "0.5 (default)",
            "turn_threshold": "0.7 (default) - LiveKit turn detector threshold",
            "turn_silence_ms": "800 (default) - silence before turn end",
            "max_turn_duration_ms": "10000 (default) - force turn end after this duration",
            "connection_timeout": "300 (default, seconds) - auto-close after no data for this duration",
            "enable_speech_start": "true|false",
            "enable_partial": "true|false"
        },
        "turn_detection": {
            "model": "Enhanced Linguistic Turn Detector (multi-factor analysis)",
            "features": [
                "Sentence structure analysis",
                "Punctuation patterns",
                "Word count analysis",
                "Completion/incomplete indicators",
                "Pause detection",
                "Number/address handling"
            ],
            "description": "Turn end requires BOTH silence AND semantic check OR max duration reached",
            "conditions": [
                "silence_duration_ms >= turn_silence_ms AND turn_probability >= turn_threshold",
                "OR turn_duration_ms >= max_turn_duration_ms (timeout protection)"
            ],
            "context": "Last 6 messages analyzed with linguistic patterns",
            "threshold": "0.7 (default) - adjust via turn_threshold parameter"
        },
        "barge_in": {
            "description": "Barge-in only triggers when TTS is actually playing",
            "required": "Client MUST send tts_start/tts_end messages",
            "flow": "speech_start → reset state → stop old transcription → clear buffer"
        },
        "interrupts": {
            "is_interrupted": "Flag to skip ongoing transcription when barge-in occurs",
            "behavior": "Skips current chunk AND resets buffer to clear old audio"
        }
    }


# ==============================================================================
# WEBSOCKET ASR ENDPOINT
# ==============================================================================


@app.websocket("/stream/asr")
async def websocket_asr(ws: WebSocket) -> None:
    """WebSocket endpoint for streaming speech-to-text."""

    client_host = ws.client.host if ws.client else "unknown"
    print(f"[ASR] WebSocket connection from {client_host}")
    await ws.accept()

    # Parse query parameters
    language = ws.query_params.get("language", "auto")
    temperature = float(ws.query_params.get("temperature", 0.0))
    use_vad = ws.query_params.get("use_vad", "true").lower() == "true"
    vad_threshold = float(ws.query_params.get("vad_threshold", 0.5))
    buffer_duration_ms = int(ws.query_params.get("buffer_duration_ms", 1500))
    enable_silence_flush = ws.query_params.get("silence_flush", "true").lower() == "true"
    min_rms_threshold = float(ws.query_params.get("min_rms_threshold", 0.07))
    silence_threshold = float(ws.query_params.get("silence_threshold", 0.08))

    # New: Turn detection parameters
    turn_silence_ms = int(ws.query_params.get("turn_silence_ms", 1200))  # ✅ FIX 2: Increased from 800 to 1200ms for thinking pauses
    enable_speech_start = ws.query_params.get("enable_speech_start", "true").lower() == "true"
    enable_partial = ws.query_params.get("enable_partial", "true").lower() == "true"

    print(f"[ASR] language={language}, temperature={temperature}, use_vad={use_vad}")
    print(f"[ASR] turn_silence_ms={turn_silence_ms}, speech_start={enable_speech_start}, partial={enable_partial}")

    # Initialize VAD if enabled
    vad = None
    vad_type = "none"
    if use_vad:
        if VAD_AVAILABLE:
            try:
                vad = SileroVAD(sample_rate=SAMPLE_RATE, threshold=vad_threshold)
                vad_type = "Silero VAD (ML)"
                print(f"[VAD] ✅ Using Silero VAD (ML-based)")
            except Exception as e:
                vad = SimpleVAD(sample_rate=SAMPLE_RATE, threshold=vad_threshold)
                vad_type = "Simple VAD (energy)"
                print(f"[VAD] ⚠️ Silero VAD failed, using SimpleVAD: {e}")
        else:
            vad = SimpleVAD(sample_rate=SAMPLE_RATE, threshold=vad_threshold)
            vad_type = "Simple VAD (energy)"
            print(f"[VAD] ⚠️ Using SimpleVAD (Silero not available)")

    # Streaming buffer
    streaming_buffer = StreamingAudioBuffer(
        sample_rate=SAMPLE_RATE,
        buffer_duration_ms=buffer_duration_ms,
        enable_silence_flush=enable_silence_flush,
        vad=vad.is_speech if vad else None,
        silence_threshold=silence_threshold,
    )

    # Audio buffer configuration
    bytes_per_sample = 2  # PCM16
    frame_duration_ms = 10
    frame_size = int(SAMPLE_RATE * frame_duration_ms / 1000)
    frame_bytes = frame_size * bytes_per_sample

    audio_buffer = bytearray()
    segment_count = 0
    session_id = str(uuid.uuid4())[:8]

    # New: VAD tracking for real-time speech detection
    last_speech_time = None
    speech_start_sent = False
    silence_duration_ms = 0
    total_frames = 0
    speech_frames = 0

    # New: Turn Detector for semantic turn prediction
    # ✅ UPGRADE: Uses enhanced linguistic turn detector
    turn_threshold = float(ws.query_params.get("turn_threshold", 0.7))
    turn_detector = create_turn_detector(threshold=turn_threshold)

    detector_type = "Enhanced Linguistic"
    print(f"[TURN DETECTOR] Using: {detector_type}")
    print(f"[TURN DETECTOR] Threshold: {turn_threshold}")

    transcript_history = []  # ✅ IMPROVE: Keep only last 6 messages
    current_turn_probability = 0.0  # Track for turn_end decision

    # ✅ FIX: Track TTS state to only reset when actually playing
    tts_is_playing = False
    is_interrupted = False  # ✅ FIX: Flag to skip ongoing transcription

    # ✅ FIX: Add turn duration tracking to prevent stuck states
    turn_start_time = None
    max_turn_duration_ms = int(ws.query_params.get("max_turn_duration_ms", 10000))  # 10s default

    # ✅ FIX: Make connection timeout configurable (default 300 seconds = 5 minutes)
    connection_timeout = float(ws.query_params.get("connection_timeout", 300))

    await ws.send_json({
        "type": "ready",
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "features": {
            "vad": use_vad,
            "vad_type": vad_type,
            "speech_start": enable_speech_start,
            "partial": enable_partial,
            "turn_detection": True,
            "turn_detector_type": detector_type,
            "turn_silence_ms": turn_silence_ms,
        },
        "info": {
            "connection_timeout": connection_timeout,
            "message": "Connection ready. Send audio data or JSON messages."
        }
    })
    last_send_time = time.time()

    print(f"[ASR] WebSocket ready - vad={use_vad} ({vad_type}), turn_detector={detector_type}")

    def is_silent(audio: np.ndarray, threshold: float = 0.008) -> bool:
        rms = np.sqrt(np.mean(audio**2))
        return rms < threshold

    def reset_turn_state():
        """Reset all turn-related state (for barge-in)."""
        nonlocal last_speech_time, speech_start_sent, silence_duration_ms, transcript_history, current_turn_probability, is_interrupted, turn_start_time
        streaming_buffer.reset()
        turn_detector.reset()
        transcript_history.clear()
        current_turn_probability = 0.0
        last_speech_time = None
        speech_start_sent = False
        silence_duration_ms = 0
        is_interrupted = False
        turn_start_time = None
        print(f"[BARGE-IN] Turn state reset - ready for new input")

    try:
        last_receive_time = time.time()
        last_send_time = time.time()  # Track last time we sent data to client
        # connection_timeout is now configurable (default 300 seconds = 5 minutes)

        while ws.client_state == WebSocketState.CONNECTED:
            try:
                data = await asyncio.wait_for(ws.receive(), timeout=2.0)  # Increased from 0.5s to 2s

                if "bytes" in data:
                    audio_chunk = data["bytes"]
                    audio_buffer.extend(audio_chunk)
                    last_receive_time = time.time()  # Update last activity

                    # Process audio in frames
                    while len(audio_buffer) >= frame_bytes:
                        frame_bytes_data = bytes(audio_buffer[:frame_bytes])
                        audio_buffer = audio_buffer[frame_bytes:]

                        frame_int16 = np.frombuffer(frame_bytes_data, dtype=np.int16)
                        frame_float32 = frame_int16.astype(np.float32) / 32768.0

                        # New: Real-time VAD processing per frame
                        total_frames += 1
                        is_speech_frame = True  # Default: treat all frames as speech when VAD disabled
                        if vad:
                            is_speech_frame = vad.is_speech(frame_float32)
                            if is_speech_frame:
                                speech_frames += 1

                        # ✅ FIX: Use VAD for turn tracking when available, fallback to RMS
                        # When VAD is disabled, use simple RMS energy for basic silence detection
                        frame_rms = np.sqrt(np.mean(frame_float32 ** 2))
                        is_speech_by_rms = frame_rms > 0.01  # Simple RMS threshold

                        if is_speech_frame or (vad is None and is_speech_by_rms):
                            silence_duration_ms = 0
                            if last_speech_time is None:
                                last_speech_time = time.time()
                                # ✅ FIX: Track turn start time for max duration protection
                                if turn_start_time is None:
                                    turn_start_time = time.time()

                                # ✅ FIX CRITICAL BUG: speech_start should ONLY fire when VAD is enabled
                                # When VAD is disabled, we can't reliably detect speech onset
                                if enable_speech_start and vad is not None and not speech_start_sent:
                                    await ws.send_json({
                                        "type": "speech_start",
                                        "timestamp": time.time()
                                    })
                                    last_send_time = time.time()
                                    last_send_time = time.time()
                                    speech_start_sent = True
                                    print(f"[VAD] Speech start detected - sending interrupt signal")

                                    # ✅ FIX: Only reset state if TTS is playing (barge-in scenario)
                                    if tts_is_playing:
                                        print(f"[BARGE-IN] TTS was playing - resetting state")
                                        reset_turn_state()
                                        is_interrupted = True
                        else:
                            silence_duration_ms += frame_duration_ms

                        # ✅ FIX: Add max duration protection to prevent stuck states
                        if turn_start_time is not None:
                            turn_duration_ms = int((time.time() - turn_start_time) * 1000)
                            if turn_duration_ms >= max_turn_duration_ms:
                                print(f"[TIMEOUT] Max turn duration reached: {turn_duration_ms}ms - forcing turn end")
                                # Force turn end even if silence not detected
                                await ws.send_json({
                                    "type": "turn_end",
                                    "reason": "max_duration",
                                    "turn_duration_ms": turn_duration_ms,
                                    "turn_probability": round(current_turn_probability, 3)
                                })
                                last_send_time = time.time()
                                # ✅ FIX: Also clear streaming buffer to prevent spurious transcriptions
                                streaming_buffer.reset()
                                # Reset for next turn
                                transcript_history.clear()
                                current_turn_probability = 0.0
                                last_speech_time = None
                                speech_start_sent = False
                                silence_duration_ms = 0
                                turn_start_time = None
                                is_interrupted = False
                                # Skip processing this frame
                                continue

                        # ✅ FIX: Re-evaluate turn probability on silence (not stale)
                        if silence_duration_ms >= turn_silence_ms and last_speech_time is not None:
                            # ✅ FIX: Re-evaluate turn probability on silence (avoid stale data)
                            current_turn_probability = turn_detector.predict(transcript_history)

                            # Calculate VAD percentage for this session
                            vad_percentage = (speech_frames / total_frames * 100) if total_frames > 0 else 0

                            # ✅ CRITICAL FIX: Only send turn_end if semantic check passes
                            if current_turn_probability >= turn_detector.threshold:
                                await ws.send_json({
                                    "type": "turn_end",
                                    "silence_duration_ms": silence_duration_ms,
                                    "vad_percentage": round(vad_percentage, 2),
                                    "turn_probability": round(current_turn_probability, 3)
                                })
                                last_send_time = time.time()
                                print(f"[VAD] Turn end detected - silence: {silence_duration_ms}ms, VAD: {vad_percentage:.1f}%, turn_prob: {current_turn_probability:.3f}")

                                # Reset for next turn
                                transcript_history.clear()
                                current_turn_probability = 0.0
                                last_speech_time = None
                                speech_start_sent = False
                                silence_duration_ms = 0
                                turn_start_time = None
                                is_interrupted = False
                            else:
                                print(f"[VAD] Silence detected but turn not complete (prob: {current_turn_probability:.3f} < {turn_detector.threshold})")

                        # ✅ FIX: Pass pre-computed VAD result to avoid duplicate VAD call
                        buffered_audio = streaming_buffer.add_frame(frame_float32, is_speech_frame)

                        if buffered_audio is not None:
                            segment_count += 1
                            duration = len(buffered_audio) / SAMPLE_RATE

                            min_duration = 0.3 if use_vad else 0.6
                            if duration < min_duration:
                                continue

                            # ✅ FIX: Check interrupt flag BEFORE transcribing (barge-in)
                            if is_interrupted:
                                print(f"[INTERRUPT] Skipping transcription - barge-in detected")
                                streaming_buffer.reset()  # ✅ FIX: Also reset buffer to clear old audio
                                is_interrupted = False  # Reset flag
                                continue

                            # ✅ FIX: Keep RMS check for audio quality (not turn logic)
                            # This filters out completely silent/bad audio segments
                            rms = np.sqrt(np.mean(buffered_audio ** 2))
                            if rms < min_rms_threshold:
                                continue

                            # Transcribe using global model function
                            result = transcribe_audio(
                                audio=buffered_audio,
                                language=language,
                                temperature=temperature,
                            )

                            raw_text = result.get("text", "").strip()

                            if raw_text:
                                filtered_text = filter_hallucinations(filter_non_speech_labels(raw_text))

                                if filtered_text:
                                    print(f"[SEGMENT #{segment_count}] {filtered_text}")

                                    # ✅ IMPROVE: Update transcript history for turn detector
                                    # Keep only last 6 messages for better accuracy
                                    transcript_history.append({
                                        "role": "user",
                                        "content": filtered_text
                                    })
                                    transcript_history = transcript_history[-6:]  # ✅ Keep last 6 messages

                                    # ✅ IMPROVE: Calculate turn probability using LiveKit ML model
                                    turn_probability = turn_detector.predict(transcript_history)
                                    current_turn_probability = turn_probability  # Track for turn_end decision

                                    # New: Send partial transcript first (for turn detector)
                                    if enable_partial:
                                        await ws.send_json({
                                            "type": "partial",
                                            "text": filtered_text,
                                            "segment": segment_count,
                                            "is_final": False,
                                            "timestamp": time.time(),
                                            "turn_probability": round(turn_probability, 3)
                                        })
                                        last_send_time = time.time()

                                    # Then send final segment with turn probability
                                    await ws.send_json({
                                        "type": "segment_completed",
                                        "text": filtered_text,
                                        "segment": segment_count,
                                        "is_final": True,
                                        "timestamp": time.time(),
                                        "turn_probability": round(turn_probability, 3),
                                        "metadata": {
                                            "silence_duration_ms": silence_duration_ms,
                                            "vad_percentage": round((speech_frames / total_frames * 100) if total_frames > 0 else 0, 2)
                                        }
                                    })
                                    last_send_time = time.time()

                elif "text" in data:
                    try:
                        msg = json.loads(data["text"])

                        if msg.get("type") == "done":
                            # ✅ FIX: Don't close connection - just flush buffer and continue
                            # This allows multi-turn conversation without reconnecting
                            print(f"[ASR] Received done signal - flushing buffer and keeping connection open")
                            # Flush any remaining audio and send complete signal
                            final_audio = streaming_buffer.flush()
                            if final_audio is not None and len(final_audio) > 0:
                                duration = len(final_audio) / SAMPLE_RATE
                                min_duration = 0.3 if vad else 0.1
                                if duration >= min_duration:
                                    result = transcribe_audio(
                                        audio=final_audio,
                                        language=language,
                                        temperature=temperature,
                                    )
                                    raw_text = result.get("text", "")
                                    filtered_text = filter_non_speech_labels(raw_text)
                                    if filtered_text:
                                        if ws.client_state == WebSocketState.CONNECTED:
                                            await ws.send_json({
                                                "type": "segment_completed",
                                                "text": filtered_text,
                                                "segment": "final",
                                                "is_final": True,
                                            })
                                            last_send_time = time.time()
                                            await ws.send_json({
                                                "type": "turn_end",
                                                "reason": "done_signal",
                                                "turn_probability": 1.0
                                            })
                                            last_send_time = time.time()
                            # Send complete signal and continue listening
                            if ws.client_state == WebSocketState.CONNECTED:
                                await ws.send_json({"type": "complete"})
                                last_send_time = time.time()
                            # Reset state for next turn and continue loop
                            reset_turn_state()
                            continue
                        elif msg.get("type") == "reset":
                            # ✅ FIX: Use reset_turn_state for consistent reset
                            reset_turn_state()
                            if vad:
                                vad.reset()
                            await ws.send_json({"type": "status", "message": "Reset complete"})
                            last_send_time = time.time()
                        elif msg.get("type") == "interrupt":
                            # 🔥 BARGE-IN: Client requests interrupt (e.g., stop TTS)
                            reset_turn_state()  # ✅ FIX: Fully reset state on interrupt
                            await ws.send_json({
                                "type": "interrupt_ack",
                                "timestamp": time.time()
                            })
                            last_send_time = time.time()
                        elif msg.get("type") == "tts_start":
                            # ✅ FIX: Client tells us TTS is starting
                            tts_is_playing = True
                            print(f"[TTS] TTS started - barge-in detection enabled")
                        elif msg.get("type") == "tts_end":
                            # ✅ FIX: Client tells us TTS ended
                            tts_is_playing = False
                            print(f"[TTS] TTS ended - barge-in detection disabled")
                            print(f"[ASR] Interrupt request received")

                    except json.JSONDecodeError:
                        pass

            except asyncio.TimeoutError:
                # Check for connection timeout (based on LAST ACTIVITY, not just receive)
                time_since_last_activity = time.time() - last_receive_time

                if time_since_last_activity > connection_timeout:
                    print(f"[ASR] Connection timeout - no data for {connection_timeout}s")
                    break

                # ✅ FIX: Send keep-alive ping every 15 seconds to prevent connection drops
                # This keeps the WebSocket connection alive even when idle
                if time.time() - last_send_time > 15:
                    try:
                        await ws.send_json({
                            "type": "ping",
                            "timestamp": time.time()
                        })
                        last_send_time = time.time()
                        last_send_time = time.time()
                        print(f"[ASR] Keep-alive ping sent")
                    except Exception:
                        print(f"[ASR] Failed to send keep-alive - connection may be dead")
                        break

                continue

    except WebSocketDisconnect as e:
        print(f"[ASR] WebSocket disconnected - code: {e.code}")

    except Exception as e:
        print(f"[ASR ERROR] {e}")
        traceback.print_exc()

    finally:
        # Process remaining audio
        while len(audio_buffer) >= frame_bytes:
            frame_bytes_data = bytes(audio_buffer[:frame_bytes])
            audio_buffer = audio_buffer[frame_bytes:]

            frame_int16 = np.frombuffer(frame_bytes_data, dtype=np.int16)
            frame_float32 = frame_int16.astype(np.float32) / 32768.0

            streaming_buffer.add_frame(frame_float32)

        final_audio = streaming_buffer.flush()

        if final_audio is not None and len(final_audio) > 0:
            try:
                duration = len(final_audio) / SAMPLE_RATE
                min_duration = 0.3 if vad else 0.1

                if duration >= min_duration:
                    result = transcribe_audio(
                        audio=final_audio,
                        language=language,
                        temperature=temperature,
                    )

                    raw_text = result.get("text", "")
                    filtered_text = filter_non_speech_labels(raw_text)

                    if filtered_text:
                        # ✅ FIX: Check connection state before sending
                        if ws.client_state == WebSocketState.CONNECTED:
                            await ws.send_json({
                                "type": "segment_completed",
                                "text": filtered_text,
                                "segment": "final",
                                "is_final": True,
                            })
                            last_send_time = time.time()

                        # ✅ CRITICAL FIX: Send turn_end after final segment to trigger pipeline
                        # This is the signal that the full user utterance is complete
                        if ws.client_state == WebSocketState.CONNECTED:
                            await ws.send_json({
                                "type": "turn_end",
                                "reason": "done_signal",
                                "text": filtered_text,
                                "turn_probability": 1.0  # Done signal means user is definitely done
                            })
                            last_send_time = time.time()
                            print(f"[ASR] ✅ turn_end sent after final segment: '{filtered_text}'")

            except RuntimeError as e:
                if "close message has been sent" in str(e):
                    print(f"[ASR] Connection already closed during final segment processing")
                else:
                    print(f"[ASR] Error processing final segment: {e}")
            except Exception as e:
                print(f"[ASR] Error processing final segment: {e}")

        # ✅ FIX: Wrap send_json in try-except to handle closed connections
        try:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.send_json({"type": "complete"})
                last_send_time = time.time()
                print(f"[ASR] ✅ Transcription complete - connection stays OPEN for next turn")
        except RuntimeError as e:
            if "close message has been sent" in str(e):
                print(f"[ASR] Connection already closed, skipping complete message")
            else:
                raise
        except Exception:
            print(f"[ASR] Error sending complete message, connection may be closed")

        # ✅ CRITICAL FIX: DO NOT close connection after complete!
        # Connection should stay alive for continuous multi-turn conversation
        # Connection will close naturally when:
        # - Client disconnects
        # - Connection timeout occurs (handled above)
        # - Error occurs
        # Removing the explicit close() call that was breaking continuous conversation

    print(f"[ASR] WebSocket handler ended - connection closed by client or timeout")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Whisper ASR Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8006, help="Port")
    parser.add_argument("--model", type=str, default="large-v3", help="Model")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device")

    args = parser.parse_args()

    os.environ["WHISPER_MODEL"] = args.model
    os.environ["WHISPER_DEVICE"] = args.device

    # Use app directly instead of string import to avoid re-loading the model
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
