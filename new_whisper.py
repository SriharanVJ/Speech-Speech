"""
Whisper Large Turbo v3 - Enhanced WebSocket Speech-to-Text Server

🔥 NEW FEATURES:
- Real-time Silero VAD v4 (ONNX) with speech_start events (barge-in support)
- Partial transcripts for live UX
- ✅ ML-Based Turn Detection (Smart Turn v3 from pipecat-ai)
- VAD percentage tracking
- ✅ COMPLETE BARGE-IN IMPLEMENTATION (from Z-agent-llm)
- Client interrupt support
- Max duration timeout protection (prevents stuck states)
- Generation ID tracking for stale audio prevention
- Cooldown period to prevent false barge-in triggers
- Audio queue management with instant cancellation

✅ BARGE-IN ARCHITECTURE (Z-agent-llm style):
    User Speech → Silero VAD SPEAKING → Barge-in Trigger → Pipeline Interrupt
    → Audio Queue Clear → Generation ID Invalidated → Resume Listening

Key Features:
    - 600ms cooldown after pipeline end prevents false triggers
    - Generation ID tracking drops stale audio from cancelled generations
    - Preserved VAD state during barge-in (no turn detector reset)
    - Audio kept in Python queue (not WebRTC buffer) for instant cancellation
    - Full state cleanup on interruption

Architecture:
    Audio → Silero VAD v4 → Streaming Buffer → Whisper (partial + final) → Smart Turn v3 Detector
    ↓
    [Barge-in Detection] → [Pipeline Interrupt] → [State Reset] → [Resume Listening]

Usage:
    python new_whisper.py

Install requirements:
    pip install torch transformers (optional - for future ML features)

WebSocket:
    ws://localhost:8006/stream/asr?language=en&use_vad=true&enable_speech_start=true

Events:
    - speech_start: Barge-in trigger (stop TTS immediately)
    - partial: Real-time transcript for turn detector
    - turn_end: Turn complete due to silence OR max duration
    - segment_completed: Final transcript with metadata
    - interruption: Pipeline interrupted due to barge-in (with generation_id)

Client Integration (REQUIRED):
    - Send tts_start when TTS begins (enables barge-in detection)
    - Send tts_end when TTS ends (disables barge-in detection)
    - Handle interruption event to stop playback immediately
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

# Import scipy for audio preprocessing (Phase 1 improvements)
try:
    from scipy import signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[INFO] ⚠️ scipy not available - audio preprocessing will be limited")
    print("[INFO] Install with: pip install scipy")

# Import ML-based VAD and Turn Detection
try:
    from vad import TurnDetector, TurnState, TurnEvent, create_turn_detector
    VAD_MODULE_AVAILABLE = True
    print("[INFO] ✅ ML-based VAD and Turn Detection module loaded")
except ImportError as e:
    VAD_MODULE_AVAILABLE = False
    print(f"[INFO] ⚠️ VAD module not available: {e}")
    print("[INFO] Install with: pip install -r vad_requirements.txt")
    print("[INFO] Then run: python download_models.py")
    print("[INFO] Will use fallback heuristic turn detector")

# Import Voice Gateway for LLM→TTS pipeline
try:
    from voice_gateway import VoiceGateway, create_voice_gateway, VoiceGatewayConfig
    VOICE_GATEWAY_AVAILABLE = True
    print("[INFO] ✅ Voice Gateway module loaded (LLM→TTS pipeline)")
except ImportError as e:
    VOICE_GATEWAY_AVAILABLE = False
    print(f"[INFO] ⚠️ Voice Gateway not available: {e}")
    print("[INFO] Install with: pip install -r voice_agent_requirements.txt")

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
# TURN DETECTOR (ML-based using Smart Turn v3 from pipecat-ai)
# ==============================================================================


# Fallback heuristic turn detector (used when ML model not available)
class HeuristicTurnDetector:
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

        print(f"[TURN DETECTOR] ⚠️ Fallback heuristic turn detector initialized")
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
def create_turn_detector_instance(threshold: float = 0.7):
    """
    Create a turn detector instance.

    Uses ML-based Smart Turn v3 if available, otherwise falls back to
    heuristic linguistic analysis.

    Args:
        threshold: Probability threshold for turn completion

    Returns:
        TurnDetector instance (ML or heuristic)
    """
    if VAD_MODULE_AVAILABLE:
        try:
            # Use ML-based Smart Turn v3
            detector = create_turn_detector(threshold=threshold)
            print(f"[TURN DETECTOR] ✅ Using ML-based Smart Turn v3 (threshold={threshold})")
            return detector
        except Exception as e:
            print(f"[TURN DETECTOR] ⚠️ ML detector failed to initialize: {e}")
            print(f"[TURN DETECTOR] Falling back to heuristic detector")

    # Fallback to heuristic
    print(f"[TURN DETECTOR] ⚠️ Using heuristic linguistic turn detector (threshold={threshold})")
    return HeuristicTurnDetector(threshold=threshold)


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
# BARGE-IN IMPLEMENTATION (from Z-agent-llm)
# ==============================================================================


class BargeInState:
    """
    Manages barge-in state for instant TTS interruption.

    Based on Z-agent-llm sandwich_gateway.py implementation.
    """

    # Cooldown period after pipeline end to prevent false triggers
    BARGE_IN_COOLDOWN_MS = 600

    def __init__(self):
        """Initialize barge-in state."""
        self._agent_speaking = False
        self._interruption_event = None  # Will be asyncio.Event
        self._generation_id = 0
        self._last_pipeline_end_time = 0.0
        self._was_interrupted = False
        self._interrupted_ai_text = ""

        # Audio queue for TTS output (with generation tracking)
        self._audio_queue = None  # Will be asyncio.Queue
        self._audio_queue_task = None

        print("[BARGE-IN] State initialized")

    def set_agent_speaking(self, speaking: bool) -> None:
        """Set agent speaking state."""
        self._agent_speaking = speaking
        if speaking:
            print(f"[BARGE-IN] Agent started speaking (gen={self._generation_id})")
        else:
            print(f"[BARGE-IN] Agent stopped speaking (gen={self._generation_id})")
            self._last_pipeline_end_time = time.time()

    def get_agent_speaking(self) -> bool:
        """Get agent speaking state."""
        return self._agent_speaking

    def get_generation_id(self) -> int:
        """Get current generation ID."""
        return self._generation_id

    def increment_generation_id(self) -> int:
        """Increment generation ID and return new value."""
        self._generation_id += 1
        print(f"[BARGE-IN] Generation ID incremented to {self._generation_id}")
        return self._generation_id

    def clear_interruption_event(self) -> None:
        """Clear the interruption event."""
        if self._interruption_event:
            self._interruption_event.clear()
            print(f"[BARGE-IN] Interruption event cleared")

    def set_interruption_event(self) -> None:
        """Set the interruption event (triggers barge-in)."""
        if self._interruption_event:
            self._interruption_event.set()
            print(f"[BARGE-IN] Interruption event set")

    def get_interruption_event(self):
        """Get the interruption event."""
        return self._interruption_event

    def set_interruption_event_object(self, event) -> None:
        """Set the interruption event object (asyncio.Event)."""
        self._interruption_event = event

    def was_interrupted(self) -> bool:
        """Check if was interrupted."""
        return self._was_interrupted

    def set_was_interrupted(self, interrupted: bool) -> None:
        """Set interrupted flag."""
        self._was_interrupted = interrupted
        if interrupted:
            print(f"[BARGE-IN] Was interrupted flag set")

    def get_interrupted_ai_text(self) -> str:
        """Get the AI text that was interrupted."""
        return self._interrupted_ai_text

    def set_interrupted_ai_text(self, text: str) -> None:
        """Set the AI text that was interrupted."""
        self._interrupted_ai_text = text

    def can_trigger_barge_in(self) -> bool:
        """
        Check if barge-in can be triggered.

        Returns False if:
        - Agent is not speaking
        - Cooldown period hasn't elapsed (prevents false triggers)
        """
        if not self._agent_speaking:
            return False

        # Check cooldown period (600ms after pipeline end)
        time_since_end = (time.time() - self._last_pipeline_end_time) * 1000
        if time_since_end < self.BARGE_IN_COOLDOWN_MS:
            print(f"[BARGE-IN] Ignoring barge-in - too soon after pipeline end ({time_since_end:.0f}ms < {self.BARGE_IN_COOLDOWN_MS}ms)")
            return False

        return True

    def reset_for_new_turn(self) -> None:
        """Reset state for new turn (called after successful transcription)."""
        self._was_interrupted = False
        self._interrupted_ai_text = ""
        print(f"[BARGE-IN] Reset for new turn (gen={self._generation_id})")


class AudioQueueManager:
    """
    Manages audio output queue with generation tracking and instant cancellation.

    Based on Z-agent-llm sandwich_gateway.py audio queue processor.
    """

    def __init__(self, max_size: int = 500):
        """
        Initialize audio queue manager.

        Args:
            max_size: Maximum queue size (frames)
        """
        self._queue = None  # Will be asyncio.Queue
        self._max_size = max_size
        self._processor_task = None
        print(f"[AUDIO_QUEUE] Manager initialized (max_size={max_size})")

    def initialize(self, loop) -> None:
        """
        Initialize the asyncio queue and processor.

        Args:
            loop: Event loop for async operations
        """
        import asyncio
        self._queue = asyncio.Queue(maxsize=self._max_size)
        print(f"[AUDIO_QUEUE] Async queue initialized")

    async def put_audio(self, gen_id: int, audio_data: bytes) -> bool:
        """
        Put audio data in the queue with generation tracking.

        Args:
            gen_id: Generation ID for this audio
            audio_data: PCM audio data

        Returns:
            True if queued successfully, False if queue is full
        """
        if self._queue is None:
            return False

        try:
            await asyncio.wait_for(
                self._queue.put((gen_id, audio_data)),
                timeout=0.1
            )
            return True
        except asyncio.TimeoutError:
            print(f"[AUDIO_QUEUE] Queue full - dropping audio frame")
            return False

    def clear_queue(self, gen_id: int = -1) -> int:
        """
        Clear the audio queue.

        Args:
            gen_id: If specified, only clear frames with this generation ID

        Returns:
            Number of frames cleared
        """
        if self._queue is None:
            return 0

        cleared = 0
        while not self._queue.empty():
            try:
                frame_gen, audio_data = self._queue.get_nowait()
                if gen_id < 0 or frame_gen == gen_id:
                    cleared += 1
                else:
                    # Put it back if it's from a different generation
                    try:
                        self._queue.put_nowait((frame_gen, audio_data))
                    except asyncio.QueueFull:
                        pass
            except asyncio.QueueEmpty:
                break

        if cleared > 0:
            print(f"[AUDIO_QUEUE] Cleared {cleared} audio frames from queue")

        return cleared

    def get_queue_size(self) -> int:
        """Get current queue size."""
        if self._queue is None:
            return 0
        return self._queue.qsize()

    async def get_audio_frame(self, timeout: float = 1.0):
        """
        Get audio frame from queue.

        Args:
            timeout: Timeout in seconds

        Returns:
            Tuple of (gen_id, audio_data) or None if timeout
        """
        if self._queue is None:
            return None

        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def reset(self) -> None:
        """Reset the audio queue (clear all pending audio)."""
        self.clear_queue()
        print(f"[AUDIO_QUEUE] Reset complete")


# ==============================================================================
# BARGE-IN TRIGGER AND PIPELINE INTERRUPT (Z-agent-llm implementation)
# ==============================================================================


async def trigger_barge_in(
    barge_in_state: BargeInState,
    audio_queue: AudioQueueManager,
    ws: WebSocket,
    turn_detector = None
) -> None:
    """
    Trigger barge-in with cooldown after pipeline end.

    Based on Z-agent-llm sandwich_gateway.py _trigger_barge_in().

    Args:
        barge_in_state: BargeInState instance
        audio_queue: AudioQueueManager instance
        ws: WebSocket connection
        turn_detector: Optional TurnDetector instance
    """
    if not barge_in_state.can_trigger_barge_in():
        return

    logger_msg = f"[BARGE-IN] 🔴 Barge-in triggered (Instant VAD) gen={barge_in_state.get_generation_id()}"
    print(logger_msg)

    await interrupt_pipeline(barge_in_state, audio_queue, ws, turn_detector)

    # Leave _interruption_event SET. The cancelled pipeline may still have
    # threads running briefly; clearing here would let any frames they queue
    # during that window play through. _spawn_pipeline clears it when the
    # next generation actually starts.

    # NOTE: Do NOT reset turn_detector here! The user is actively speaking.
    # Resetting destroys speech tracking state and causes it to remain stuck
    # in LISTENING mode instead of processing the Turn sequence properly.


async def interrupt_pipeline(
    barge_in_state: BargeInState,
    audio_queue: AudioQueueManager,
    ws: WebSocket,
    turn_detector = None
) -> None:
    """
    Nuclear barge-in: stop everything immediately.

    Based on Z-agent-llm sandwich_gateway.py _interrupt_pipeline().

    Args:
        barge_in_state: BargeInState instance
        audio_queue: AudioQueueManager instance
        ws: WebSocket connection
        turn_detector: Optional TurnDetector instance
    """
    if not barge_in_state.get_agent_speaking():
        return

    gen_id = barge_in_state.get_generation_id()
    logger_msg = f"[BARGE-IN] 🔴 FULL STOP on gen {gen_id}"
    print(logger_msg)

    # Set interruption event
    barge_in_state.set_interruption_event()
    barge_in_state.set_agent_speaking(False)

    # Record when TTS stopped (for cooldown in next barge-in)
    # This is handled in set_agent_speaking(False)

    # Clear audio queue completely
    cleared = audio_queue.clear_queue()
    if cleared > 0:
        print(f"[BARGE-IN] 🔴 Cleared {cleared} audio frames from queue")

    # Send interruption event to client
    try:
        await ws.send_json({
            "type": "interruption",
            "generation_id": gen_id,
            "timestamp": time.time()
        })
        print(f"[BARGE-IN] 🔴 Sent interruption event to client")
    except Exception as e:
        print(f"[BARGE-IN] Failed to send interruption event: {e}")

    # Mark as interrupted (for context injection in next LLM call)
    barge_in_state.set_was_interrupted(True)

    print(f"[BARGE-IN] 🔴 Pipeline interrupt complete (gen {gen_id})")


# ==============================================================================
# STREAMING AUDIO BUFFER
# ==============================================================================


class StreamingAudioBuffer:
    """Real-time audio buffer for frame-by-frame processing."""

    SILENCE_THRESHOLD = 0.08
    # 🔥 PHASE 1: Improved endpointing - increased from 800 to 2000ms for natural pauses
    SILENCE_DURATION_MS = 2000
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
    use_preprocessing: bool = True,
    beam_size: int = 5,
) -> Dict[str, Any]:
    """
    Transcribe audio using whisper with Phase 1 improvements.

    Args:
        audio: Input audio array
        language: Language code or "auto"
        temperature: Sampling temperature (0.0 for greedy, retry with 0.2 if empty)
        use_preprocessing: Whether to apply audio preprocessing
        beam_size: Beam size for beam search (higher = more accurate but slower)

    Returns:
        Dict with text, segments, language, and metadata
    """
    # Ensure audio is float32 normalized
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    if np.max(np.abs(audio)) > 1.0:
        audio = audio / 32768.0

    # 🔥 PHASE 1: Apply audio preprocessing
    if use_preprocessing:
        # Debug: Check audio level before preprocessing
        rms_before = np.sqrt(np.mean(audio ** 2))
        print(f"[Whisper] Audio RMS before preprocessing: {rms_before:.4f}")

        audio = preprocess_audio(
            audio,
            sample_rate=SAMPLE_RATE,
            normalize=True,
            highpass_cutoff=80,  # Remove low-frequency noise
            noise_gate_threshold=0.02,  # Soft noise gate
        )

        # Debug: Check audio level after preprocessing
        rms_after = np.sqrt(np.mean(audio ** 2))
        print(f"[Whisper] Audio RMS after preprocessing: {rms_after:.4f} (gain: {rms_after/rms_before if rms_before > 0 else 0:.2f}x)")

    try:
        # 🔥 PHASE 1: Improved Whisper configuration
        options = {
            "task": "transcribe",
            "language": None if language == "auto" else language,
            "fp16": True if MODEL_DEVICE == "cuda" else False,
            "temperature": temperature,
            "beam_size": beam_size,  # 🔥 Increased from 1 to 5 for better accuracy
            "condition_on_previous_text": False,  # Better for barge-in scenarios
            "no_speech_threshold": 0.3,
            "logprob_threshold": -1.0,
            "compression_ratio_threshold": 2.4,
        }

        # Pass audio directly as numpy array (whisper handles 16kHz audio)
        result = model.transcribe(audio, **options)

        text = result.get("text", "").strip()
        detected_lang = result.get("language", "en")

        # 🔥 PHASE 1: Retry with higher temperature if result is empty
        if not text or len(text) < 2:
            print(f"[Whisper] Empty result with T={temperature}, retrying with T=0.2")
            options["temperature"] = 0.2
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
# AUDIO PREPROCESSING PIPELINE (Phase 1 Improvements)
# ==============================================================================


def preprocess_audio(
    audio: np.ndarray,
    sample_rate: int = 16000,
    normalize: bool = True,
    highpass_cutoff: int = 80,
    noise_gate_threshold: float = 0.02,
) -> np.ndarray:
    """
    Preprocess audio for better Whisper transcription.

    Args:
        audio: Input audio (float32, -1 to 1 range)
        sample_rate: Sample rate in Hz
        normalize: Whether to normalize audio to -3dB
        highpass_cutoff: High-pass filter cutoff frequency (Hz)
        noise_gate_threshold: Noise gate threshold (RMS)

    Returns:
        Preprocessed audio (float32)
    """
    # Ensure float32 input
    processed = audio.copy().astype(np.float32)

    # 1. High-pass filter to remove low-frequency noise
    if highpass_cutoff > 0 and SCIPY_AVAILABLE:
        try:
            nyquist = sample_rate / 2
            highpass = highpass_cutoff / nyquist
            b, a = signal.butter(4, highpass, btype='high')
            # ✅ FIX: Ensure output is float32 (filtfilt returns float64 by default)
            processed = signal.filtfilt(b, a, processed).astype(np.float32)
        except Exception as e:
            print(f"[PREPROCESS] High-pass filter failed: {e}")

    # 2. Noise gate (soft gate, not hard gate)
    if noise_gate_threshold > 0:
        rms = np.sqrt(np.mean(processed ** 2))
        if rms < noise_gate_threshold:
            # Apply soft gain reduction
            gain = rms / noise_gate_threshold if rms > 0 else 0.01
            processed = processed * (0.1 + 0.9 * gain)

    # 3. Normalize to -3dB target (Whisper performs better with consistent levels)
    if normalize:
        current_rms = np.sqrt(np.mean(processed ** 2))
        if current_rms > 0:
            # Target -3dB = 0.707 RMS
            target_rms = 0.707
            gain = target_rms / current_rms
            # ✅ FIX: Increased max gain from 3.0 to 10.0 for quiet audio
            gain = min(gain, 10.0)  # Max 10x gain (was 3.0)
            processed = processed * gain

    # Clip to prevent overflow
    processed = np.clip(processed, -1.0, 1.0)

    # ✅ FIX: Ensure output is float32
    return processed.astype(np.float32)


# ==============================================================================
# SLIDING WINDOW WITH OVERLAP (Phase 1 Improvements)
# ==============================================================================


class SlidingWindowBuffer:
    """
    Sliding window audio buffer with overlap for better transcription context.

    Uses 30-second windows with 5-second overlap, processing every 2 seconds.
    This provides:
    - Better context at word boundaries
    - Reduced word-splitting errors
    - More stable partial transcripts
    """

    # Whisper max context is 30 seconds
    MAX_WINDOW_SECONDS = 30
    # Overlap between windows for context preservation
    OVERLAP_SECONDS = 5
    # Process interval for streaming
    PROCESS_INTERVAL_SECONDS = 2

    def __init__(
        self,
        sample_rate: int = 16000,
        vad: Optional[Callable[[np.ndarray], bool]] = None,
        min_vad_percentage: float = 0.30,
    ):
        self.sample_rate = sample_rate
        self.vad = vad
        self.min_vad_percentage = min_vad_percentage

        # Buffer configuration
        self.max_window_samples = int(self.MAX_WINDOW_SECONDS * sample_rate)
        self.overlap_samples = int(self.OVERLAP_SECONDS * sample_rate)
        self.process_interval_samples = int(self.PROCESS_INTERVAL_SECONDS * sample_rate)

        # State
        self.buffer: np.ndarray = np.array([], dtype=np.float32)
        self.last_process_time = 0
        self.processed_samples = 0
        self.speech_samples = 0
        self.total_samples = 0

        # Thread safety
        self.lock = threading.Lock()

        print(f"[SLIDING_WINDOW] Initialized: max={self.MAX_WINDOW_SECONDS}s, "
              f"overlap={self.OVERLAP_SECONDS}s, interval={self.PROCESS_INTERVAL_SECONDS}s")

    def add_frame(self, frame: np.ndarray, is_speech: bool = True) -> Optional[np.ndarray]:
        """
        Add audio frame and return audio to process if interval reached.

        Args:
            frame: Audio frame to add
            is_speech: VAD result for this frame

        Returns:
            Audio to process (with overlap context) or None
        """
        with self.lock:
            self.buffer = np.concatenate([self.buffer, frame])
            self.total_samples += len(frame)

            if is_speech:
                self.speech_samples += len(frame)

            # Check if we should process
            samples_since_process = self.total_samples - self.processed_samples

            if samples_since_process >= self.process_interval_samples:
                # Calculate VAD percentage
                vad_percentage = (self.speech_samples / samples_since_process) if samples_since_process > 0 else 0

                # Only process if VAD percentage is above threshold
                if vad_percentage >= self.min_vad_percentage and len(self.buffer) >= self.overlap_samples:
                    # Get audio to process (current buffer)
                    audio_to_process = self.buffer.copy()

                    # Keep overlap for next iteration
                    overlap_buffer = self.buffer[-self.overlap_samples:] if len(self.buffer) > self.overlap_samples else self.buffer.copy()
                    self.buffer = overlap_buffer

                    # Reset counters for next interval
                    self.processed_samples = self.total_samples - len(self.buffer)
                    self.speech_samples = 0

                    return audio_to_process, vad_percentage

            # Buffer too large - flush
            if len(self.buffer) >= self.max_window_samples:
                # Process what we have
                audio_to_process = self.buffer.copy()
                self.buffer = np.array([], dtype=np.float32)
                self.processed_samples = self.total_samples
                self.speech_samples = 0
                return audio_to_process, 0.0

            return None, 0.0

    def flush(self) -> Optional[tuple]:
        """Flush remaining buffer."""
        with self.lock:
            if len(self.buffer) > self.overlap_samples:
                vad_percentage = (self.speech_samples / len(self.buffer)) if len(self.buffer) > 0 else 0
                if vad_percentage >= self.min_vad_percentage:
                    audio_to_process = self.buffer.copy()
                    self.buffer = np.array([], dtype=np.float32)
                    self.speech_samples = 0
                    self.processed_samples = self.total_samples
                    return audio_to_process, vad_percentage

            self.buffer = np.array([], dtype=np.float32)
            self.speech_samples = 0
            return None, 0.0

    def reset(self) -> None:
        """Reset buffer state."""
        with self.lock:
            self.buffer = np.array([], dtype=np.float32)
            self.processed_samples = 0
            self.speech_samples = 0
            self.total_samples = 0


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
            "barge_in": "Complete (Z-agent-llm implementation)",
            "generation_id_tracking": True,
            "audio_queue_management": True,
            "cooldown_protection": True,
            "preserved_vad_state": True,
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
            "speech_start": "🔥 BARGE-IN: Speech detected (SPEAKING state) - triggers barge-in if agent speaking",
            "partial": "Partial transcript (real-time, for turn detector)",
            "segment_completed": "Final completed transcript segment",
            "turn_end": "✅ Turn ended due to silence AND semantic check (turn_probability >= 0.7)",
            "interruption": "🔥 BARGE-IN: Pipeline interrupted due to barge-in (with generation_id)",
            "interrupt_ack": "Acknowledgment of client interrupt request",
            "complete": "Session complete",
            "status": "Status update message",
            "ping": "Keep-alive ping (connection still alive)",
            "tool_call": "🆕 LLM tool call (voice gateway)",
            "assistant_response": "🆕 Assistant's final response text (voice gateway)",
            "pipeline_done": "🆕 LLM→TTS pipeline complete (voice gateway)",
            "error": "Error message"
        },
        "client_to_server": {
            "done": "End the session",
            "reset": "Reset all state (VAD, buffer, turn detector)",
            "interrupt": "🔥 BARGE-IN: Request interrupt (stop TTS/ASR)",
            "tts_start": "✅ REQUIRED: Tell server TTS started (enables barge-in, increments generation_id)",
            "tts_end": "✅ REQUIRED: Tell server TTS ended (disables barge-in)",
            "process_turn": "🆕 Manually trigger voice agent pipeline with text (voice gateway)"
        },
        "query_params": {
            "language": "auto|en|es|fr|...",
            "temperature": "0.0 (default)",
            "use_vad": "true|false",
            "vad_threshold": "0.5 (default)",
            "turn_threshold": "0.7 (default) - LiveKit turn detector threshold",
            "turn_silence_ms": "2000 (default, improved) - silence before turn end (increased from 1200ms)",
            "max_turn_duration_ms": "10000 (default) - force turn end after this duration",
            "connection_timeout": "300 (default, seconds) - auto-close after no data for this duration",
            "enable_speech_start": "true|false",
            "enable_partial": "true|false"
        },
        "turn_detection": {
            "model": "Smart Turn v3 (ML-based, pipecat-ai)" if VAD_MODULE_AVAILABLE else "Heuristic Linguistic Turn Detector (multi-factor analysis)",
            "features": [
                "Silero VAD v4 for speech detection" if VAD_MODULE_AVAILABLE else "Energy-based VAD",
                "Whisper Tiny encoder for semantic analysis" if VAD_MODULE_AVAILABLE else "Sentence structure analysis",
                "Learned turn completion patterns" if VAD_MODULE_AVAILABLE else "Punctuation patterns",
                "Context-aware prediction" if VAD_MODULE_AVAILABLE else "Word count analysis"
            ],
            "description": "Turn end requires BOTH silence AND semantic check OR max duration reached",
            "conditions": [
                "silence_duration_ms >= turn_silence_ms AND turn_probability >= turn_threshold",
                "OR turn_duration_ms >= max_turn_duration_ms (timeout protection)"
            ],
            "context": "Last 8 seconds of audio analyzed (ML) or Last 6 messages (heuristic)",
            "threshold": "0.70 (default) - adjust via turn_threshold parameter",
            "ml_available": VAD_MODULE_AVAILABLE
        },
        "barge_in": {
            "description": "🔥 COMPLETE BARGE-IN IMPLEMENTATION (from Z-agent-llm)",
            "features": [
                "Instant VAD-based speech_start detection triggers barge-in",
                "Generation ID tracking drops stale audio from cancelled generations",
                "600ms cooldown after pipeline end prevents false triggers",
                "Preserved VAD state during barge-in (no turn detector reset)",
                "Audio queue management with instant cancellation",
                "Full state cleanup on interruption"
            ],
            "flow": [
                "User speaks → Silero VAD SPEAKING → speech_start event",
                "If agent speaking → trigger_barge_in() → interrupt_pipeline()",
                "interrupt_pipeline() → Set event → Clear queue → Send interruption to client",
                "Client receives interruption → Stop TTS playback immediately",
                "Server resets state → Resume listening for user input"
            ],
            "required": "Client MUST send tts_start/tts_end messages for generation tracking",
            "cooldown": "600ms after pipeline end prevents false triggers on first utterance",
            "generation_id": "Each TTS output has unique ID; stale audio dropped on barge-in"
        },
        "interrupts": {
            "is_interrupted": "Flag to skip ongoing transcription when barge-in occurs",
            "behavior": "Skips current chunk AND resets buffer to clear old audio",
            "state_preservation": "VAD state preserved during barge-in (no turn detector reset)",
            "audio_queue": "Cleared immediately on interruption (frames dropped before playback)"
        }
    }


# =============================================================================
# VOICE GATEWAY PIPELINE PROCESSOR
# ==============================================================================


async def process_voice_gateway_pipeline(
    voice_gateway,
    user_text: str,
    ws: WebSocket
) -> None:
    """
    Process user text through the voice gateway (LLM→TTS pipeline).

    Args:
        voice_gateway: VoiceGateway instance
        user_text: User's transcribed text
        ws: WebSocket connection for sending events
    """
    print(f"[VOICE_GATEWAY] Processing user turn: '{user_text[:50]}...'")

    try:
        # Process through voice gateway
        async for event in voice_gateway.process_user_turn(user_text, ws):
            if event["type"] == "llm_token":
                # LLM token - could be used for real-time display
                pass  # Silently accumulate for TTS

            elif event["type"] == "tool_call":
                # Tool call event
                await ws.send_json({
                    "type": "tool_call",
                    "tool": event.get("tool", "unknown"),
                    "query": event.get("query", ""),
                })

            elif event["type"] == "tts_audio":
                # TTS audio chunk - send to client
                await ws.send_bytes(event["data"])

            elif event["type"] == "transcription":
                # Assistant's final response
                await ws.send_json({
                    "type": "assistant_response",
                    "text": event.get("text", ""),
                    "speaker": event.get("speaker", "assistant"),
                    "timestamp": event.get("timestamp", time.time()),
                })

            elif event["type"] == "pipeline_done":
                # Pipeline complete
                await ws.send_json({
                    "type": "pipeline_done",
                    "generation_id": event.get("generation_id", 0),
                })
                print(f"[VOICE_GATEWAY] Pipeline complete")

            elif event["type"] == "error":
                # Error occurred
                await ws.send_json({
                    "type": "error",
                    "error": event.get("error", "Unknown error"),
                })
                logger.error(f"[VOICE_GATEWAY] Pipeline error: {event.get('error')}")

    except Exception as e:
        logger.error(f"[VOICE_GATEWAY] Pipeline processing error: {e}")
        await ws.send_json({
            "type": "error",
            "error": f"Voice gateway error: {str(e)}",
        })


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
    # 🔥 PHASE 1: Improved turn silence threshold - increased from 1200 to 2000ms
    # This prevents premature turn ending during natural thinking pauses
    turn_silence_ms = int(ws.query_params.get("turn_silence_ms", 2000))
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

    # Turn Detector for semantic turn prediction
    # Uses ML-based Smart Turn v3 if available, otherwise falls back to heuristic
    turn_threshold = float(ws.query_params.get("turn_threshold", 0.70))
    turn_detector = create_turn_detector_instance(threshold=turn_threshold)

    # Detect detector type for logging
    detector_type = "ML (Smart Turn v3)" if VAD_MODULE_AVAILABLE and hasattr(turn_detector, '_smart_turn') else "Heuristic Linguistic"
    print(f"[TURN DETECTOR] Using: {detector_type}")
    print(f"[TURN DETECTOR] Threshold: {turn_threshold}")

    # ==============================================================================
    # ✅ COMPLETE BARGE-IN IMPLEMENTATION (from Z-agent-llm)
    # ==============================================================================
    # Barge-in state management
    barge_in_state = BargeInState()
    audio_queue_manager = AudioQueueManager()

    # Create interruption event for async operations
    interruption_event = asyncio.Event()
    barge_in_state.set_interruption_event_object(interruption_event)

    # Initialize audio queue with event loop
    loop = asyncio.get_event_loop()
    audio_queue_manager.initialize(loop)

    print(f"[BARGE-IN] Complete implementation initialized")
    print(f"[BARGE-IN] Cooldown period: {BargeInState.BARGE_IN_COOLDOWN_MS}ms")
    print(f"[BARGE-IN] Generation ID tracking: enabled")
    print(f"[BARGE-IN] Audio queue management: enabled")
    print(f"[BARGE-IN] Preserved VAD state: enabled")

    transcript_history = []  # Keep last 6 messages for context
    current_turn_probability = 0.0  # Track for turn_end decision

    # ✅ FIX: Track TTS state to only reset when actually playing
    tts_is_playing = False
    is_interrupted = False  # ✅ FIX: Flag to skip ongoing transcription

    # ✅ FIX: Add turn duration tracking to prevent stuck states
    turn_start_time = None
    max_turn_duration_ms = int(ws.query_params.get("max_turn_duration_ms", 10000))  # 10s default

    # ✅ FIX: Make connection timeout configurable (default 300 seconds = 5 minutes)
    connection_timeout = float(ws.query_params.get("connection_timeout", 300))

    # ==============================================================================
    # ✅ VOICE GATEWAY (LLM → TTS Pipeline)
    # ==============================================================================
    voice_gateway = None
    if VOICE_GATEWAY_AVAILABLE:
        try:
            # Create voice gateway config
            llm_api_key = os.environ.get("GEMINI_API_KEY", "")

            # 🔍 EXPLICIT: Log API key status
            print(f"[VOICE_GATEWAY] 🔑 GEMINI_API_KEY check:")
            print(f"[VOICE_GATEWAY]    Exists: {bool(llm_api_key)}")
            print(f"[VOICE_GATEWAY]    Length: {len(llm_api_key) if llm_api_key else 0}")
            print(f"[VOICE_GATEWAY]    Prefix: {llm_api_key[:10] + '...' if llm_api_key else 'N/A'}")

            if not llm_api_key:
                print("[VOICE_GATEWAY] ⚠️ GEMINI_API_KEY not set - voice gateway not available")
            else:
                gateway_config = VoiceGatewayConfig(
                    llm_api_key=llm_api_key,
                    chatbot_id=ws.query_params.get("chatbot_id", "default"),
                    thread_id=session_id,
                    tts_host_url=os.environ.get("KOKORO_TTS_HOST", "ws://localhost:8001"),
                )

                print(f"[VOICE_GATEWAY] 🚀 Creating voice gateway...")
                voice_gateway = create_voice_gateway(gateway_config)
                print(f"[VOICE_GATEWAY] 📞 Initializing voice gateway...")
                await voice_gateway.initialize()

                print(f"[VOICE_GATEWAY] ✅ Initialized with chatbot_id={gateway_config.chatbot_id}")
        except Exception as e:
            print(f"[VOICE_GATEWAY] ⚠️ Failed to initialize: {e}")
            import traceback
            traceback.print_exc()
            voice_gateway = None

    # Accumulated user text for voice gateway processing
    accumulated_user_text = ""
    voice_gateway_enabled = voice_gateway is not None

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
            "barge_in": {
                "enabled": True,
                "implementation": "Z-agent-llm complete implementation",
                "cooldown_ms": BargeInState.BARGE_IN_COOLDOWN_MS,
                "generation_id_tracking": True,
                "audio_queue_management": True,
                "preserved_vad_state": True
            }
        },
        "info": {
            "connection_timeout": connection_timeout,
            "message": "Connection ready. Send audio data or JSON messages."
        }
    })
    last_send_time = time.time()

    print(f"[ASR] WebSocket ready - vad={use_vad} ({vad_type}), turn_detector={detector_type}")
    print(f"[ASR] ✅ Complete barge-in implementation initialized (cooldown={BargeInState.BARGE_IN_COOLDOWN_MS}ms)")

    def is_silent(audio: np.ndarray, threshold: float = 0.008) -> bool:
        rms = np.sqrt(np.mean(audio**2))
        return rms < threshold

    def reset_turn_state():
        """Reset all turn-related state (for barge-in)."""
        nonlocal last_speech_time, speech_start_sent, silence_duration_ms, transcript_history, current_turn_probability, is_interrupted, turn_start_time
        streaming_buffer.reset()
        # NOTE: Do NOT reset turn_detector here! Preserved VAD state is critical
        # for proper speech tracking during barge-in (from Z-agent-llm)
        transcript_history.clear()
        current_turn_probability = 0.0
        last_speech_time = None
        speech_start_sent = False
        silence_duration_ms = 0
        is_interrupted = False
        turn_start_time = None
        # Reset barge-in state for new turn
        barge_in_state.reset_for_new_turn()
        print(f"[BARGE-IN] Turn state reset - ready for new input (VAD state preserved)")

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

                                # ✅ FIX: Restart VAD stats at speech onset so vad_percentage
                                # measures THIS turn (speech + trailing silence) instead of
                                # accumulating idle-silence frames since connect, which drove
                                # the ratio below the 10% gate and silently skipped real speech
                                total_frames = 0
                                speech_frames = 0

                                # ✅ FIX CRITICAL BUG: speech_start should ONLY fire when VAD is enabled
                                # When VAD is disabled, we can't reliably detect speech onset
                                if enable_speech_start and vad is not None and not speech_start_sent:
                                    await ws.send_json({
                                        "type": "speech_start",
                                        "timestamp": time.time()
                                    })
                                    last_send_time = time.time()
                                    speech_start_sent = True
                                    print(f"[VAD] 🟢 Speech start detected - SPEAKING state")

                                    # ✅ COMPLETE BARGE-IN IMPLEMENTATION (from Z-agent-llm)
                                    # Trigger barge-in if agent is speaking (using ML-based VAD)
                                    if barge_in_state.get_agent_speaking():
                                        print(f"[BARGE-IN] 🔴 Agent is speaking - triggering barge-in")
                                        await trigger_barge_in(barge_in_state, audio_queue_manager, ws, turn_detector)
                                        # Mark as interrupted (will be used for context injection)
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
                                total_frames = 0  # ✅ FIX: per-turn VAD stats (were session-cumulative)
                                speech_frames = 0
                                # Skip processing this frame
                                continue

                        # ✅ FIX: Re-evaluate turn probability on silence (not stale)
                        if silence_duration_ms >= turn_silence_ms and last_speech_time is not None:
                            # ✅ FIX: Re-evaluate turn probability on silence (avoid stale data)
                            current_turn_probability = turn_detector.predict(transcript_history)

                            # Calculate VAD percentage for this session
                            vad_percentage = (speech_frames / total_frames * 100) if total_frames > 0 else 0

                            # ✅ FIX: Send turn_end if EITHER:
                            # 1. Semantic check passes (probability >= threshold)
                            # 2. OR silence has been too long (fallback for short utterances)
                            # This ensures the pipeline continues even for short phrases like "hello"
                            SILENCE_FALLBACK_MS = 2000  # 2 seconds of silence forces turn end

                            should_end_turn = (
                                current_turn_probability >= turn_detector.threshold or
                                silence_duration_ms >= SILENCE_FALLBACK_MS
                            )

                            if should_end_turn:
                                await ws.send_json({
                                    "type": "turn_end",
                                    "silence_duration_ms": silence_duration_ms,
                                    "vad_percentage": round(vad_percentage, 2),
                                    "turn_probability": round(current_turn_probability, 3)
                                })
                                last_send_time = time.time()
                                print(f"[VAD] Turn end detected - silence: {silence_duration_ms}ms, VAD: {vad_percentage:.1f}%, turn_prob: {current_turn_probability:.3f}")

                                # ✅ VOICE GATEWAY: Trigger LLM→TTS pipeline if enabled
                                if voice_gateway_enabled and accumulated_user_text:
                                    print(f"[VOICE_GATEWAY] Triggering pipeline for: '{accumulated_user_text[:50]}...'")
                                    asyncio.create_task(process_voice_gateway_pipeline(
                                        voice_gateway, accumulated_user_text, ws
                                    ))
                                    accumulated_user_text = ""  # Clear for next turn

                                # Reset for next turn
                                transcript_history.clear()
                                current_turn_probability = 0.0
                                last_speech_time = None
                                speech_start_sent = False
                                silence_duration_ms = 0
                                turn_start_time = None
                                is_interrupted = False
                                total_frames = 0  # ✅ FIX: per-turn VAD stats (were session-cumulative)
                                speech_frames = 0
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

                            # 🔥 PHASE 1: VAD-gated transcription
                            # Calculate VAD percentage for this segment
                            vad_percentage = (speech_frames / total_frames * 100) if total_frames > 0 else 0

                            # 🔥 PHASE 1: Skip transcription if VAD percentage is too low
                            # This prevents transcribing background noise as speech
                            # ✅ FIX: Lowered threshold from 30% to 10% for better sensitivity
                            min_vad_percentage = 10.0  # Require 10% speech frames (was 30%)
                            if vad_percentage < min_vad_percentage:
                                print(f"[VAD] Skipping transcription - low VAD percentage: {vad_percentage:.1f}% < {min_vad_percentage}%")
                                continue

                            print(f"[VAD] Transcribing with VAD percentage: {vad_percentage:.1f}%")

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

                                    # ✅ VOICE GATEWAY: Accumulate user text for processing
                                    if voice_gateway_enabled:
                                        accumulated_user_text += (" " + filtered_text if accumulated_user_text else filtered_text)
                                        print(f"[VOICE_GATEWAY] Accumulated text: '{accumulated_user_text[:50]}...'")

                                        # ✅ FIX: DON'T trigger here - wait for turn_end instead
                                        # Triggering on each segment causes duplicate responses
                                        # Pipeline will trigger once on turn_end (line 1908)

                elif "text" in data:
                    try:
                        msg = json.loads(data["text"])

                        if msg.get("type") == "done":
                            # ✅ FIX: Don't close connection - just flush buffer and continue
                            # This allows multi-turn conversation without reconnecting
                            print(f"[ASR] Received done signal - flushing buffer and keeping connection open")
                            # Flush any remaining audio and send complete signal
                            final_audio = streaming_buffer.flush()

                            # ✅ CRITICAL FIX: Track if we sent any segment from this done signal
                            sent_segment_from_done = False

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
                                            sent_segment_from_done = True

                            # ✅ CRITICAL FIX: ALWAYS send turn_end when done is received
                            # This ensures the pipeline continues even if buffer was empty
                            # (buffer might be empty because silence was already processed)
                            if ws.client_state == WebSocketState.CONNECTED:
                                await ws.send_json({
                                    "type": "turn_end",
                                    "reason": "done_signal",
                                    "turn_probability": 1.0
                                })
                                last_send_time = time.time()
                                print(f"[ASR] ✅ turn_end sent due to done signal")

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
                            # Use the complete barge-in implementation from Z-agent-llm
                            await interrupt_pipeline(barge_in_state, audio_queue_manager, ws, turn_detector)
                            await ws.send_json({
                                "type": "interrupt_ack",
                                "timestamp": time.time()
                            })
                            last_send_time = time.time()
                        elif msg.get("type") == "tts_start":
                            # ✅ COMPLETE BARGE-IN: Client tells us TTS is starting
                            # Increment generation ID for this new TTS output
                            barge_in_state.increment_generation_id()
                            barge_in_state.set_agent_speaking(True)
                            barge_in_state.clear_interruption_event()
                            tts_is_playing = True
                            print(f"[TTS] ✅ TTS started - barge-in detection enabled (gen={barge_in_state.get_generation_id()})")
                        elif msg.get("type") == "tts_end":
                            # ✅ COMPLETE BARGE-IN: Client tells us TTS ended
                            barge_in_state.set_agent_speaking(False)
                            tts_is_playing = False
                            gen_id = barge_in_state.get_generation_id()
                            print(f"[TTS] ✅ TTS ended - barge-in detection disabled (gen={gen_id})")
                        elif msg.get("type") == "process_turn":
                            # ✅ VOICE GATEWAY: Process text through LLM→TTS pipeline
                            user_text = msg.get("text", "")
                            if voice_gateway_enabled and user_text:
                                print(f"[VOICE_GATEWAY] Manual process_turn request: '{user_text[:50]}...'")
                                asyncio.create_task(process_voice_gateway_pipeline(
                                    voice_gateway, user_text, ws
                                ))
                            else:
                                await ws.send_json({
                                    "type": "error",
                                    "error": "Voice gateway not available or no text provided"
                                })
                            last_send_time = time.time()

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
