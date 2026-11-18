"""
Smart-Turn Detector - Standalone Voice Activity Detection + Turn Completion

Uses Silero VAD v4 for speech detection and Smart Turn v3 for semantic
turn completion detection (replaces heuristic linguistic analysis).

Architecture:
    Audio → Silero VAD → Speech Detection → Smart Turn v3 → Turn End Decision
"""

import time
import math
import logging
from collections import deque
import numpy as np
import asyncio
from typing import Optional, List
from enum import Enum
from dataclasses import dataclass

from .config import VAD_CONFIG, TURN_DETECTION_CONFIG
from .smart_turn_model import SmartTurnModel
from .silero_vad import SileroVAD

logger = logging.getLogger(__name__)


class TurnState(Enum):
    LISTENING = "listening"
    SPEAKING = "speaking"
    SILENCED = "silenced"
    TURN_END = "turn_end"


@dataclass
class TurnEvent:
    state: TurnState
    timestamp: float
    confidence: float
    silence_duration_ms: float
    speech_duration_ms: float
    smart_turn_probability: Optional[float] = None


class TurnDetector:
    """
    Turn Detector using Silero VAD + Smart Turn v3.

    This is the ML-based replacement for the heuristic turn detector.
    It uses:
    - Silero VAD v4 for accurate speech detection
    - Smart Turn v3 (Whisper Tiny + classifier) for semantic turn completion

    Configuration:
        vad_silence_trigger_ms: Milliseconds of silence before checking turn (default: 280)
        smart_turn_threshold: Probability threshold for turn end (default: 0.70)
        max_silence_timeout_ms: Maximum silence before forcing turn end (default: 1500)
        min_speech_duration_ms: Minimum speech duration to consider valid (default: 250)
        sample_rate: Audio sample rate (default: 16000)
        chunk_size: VAD processing chunk size (default: 512)
        buffer_duration_ms: Maximum audio buffer duration (default: 8000)
    """

    def __init__(
        self,
        vad_silence_trigger_ms: int = 280,
        max_silence_timeout_ms: int = 1500,
        smart_turn_threshold: float = 0.70,
        min_speech_duration_ms: int = 250,
        sample_rate: int = 16000,
        chunk_size: int = 512,
        buffer_duration_ms: int = 8000,
        smart_turn_cooldown_ms: int = 250,
        num_threads: int = 4,
        fallback_grace_ms: int = 500,
        vad_threshold: float = 0.5,
    ):
        self.vad_silence_trigger_ms = vad_silence_trigger_ms
        self.smart_turn_threshold = smart_turn_threshold
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.max_duration_seconds = buffer_duration_ms / 1000.0
        self.vad_threshold = vad_threshold

        # Create SmartTurnModel and Silero VAD
        try:
            self._smart_turn = SmartTurnModel(num_threads=num_threads)
            logger.info("[TurnDetector] Smart Turn v3 loaded successfully")
        except Exception as e:
            logger.warning(f"[TurnDetector] Smart Turn v3 not available: {e}")
            logger.warning("[TurnDetector] Will use fallback heuristic detection")
            self._smart_turn = None

        try:
            self._vad = SileroVAD(threshold=vad_threshold)
            logger.info("[TurnDetector] Silero VAD v4 loaded successfully")
        except Exception as e:
            logger.error(f"[TurnDetector] Silero VAD failed to load: {e}")
            raise RuntimeError("Silero VAD is required but not available")

        # Pre-computed config values
        self.energy_gate_rms = VAD_CONFIG.get("energy_gate_rms", 0.015)
        self.pre_speech_ms = VAD_CONFIG.get("speech_trigger_ms", 200)

        chunk_ms = (self.chunk_size / self.sample_rate) * 1000.0
        self.pre_chunks = max(1, math.ceil(self.pre_speech_ms / chunk_ms))
        self.stop_chunks = max(1, math.ceil(self.vad_silence_trigger_ms / chunk_ms))
        self.max_chunks = max(1, math.ceil(self.max_duration_seconds / (self.chunk_size / self.sample_rate)))

        fallback_ms = max_silence_timeout_ms if max_silence_timeout_ms > self.vad_silence_trigger_ms else 1500
        self.max_silence_chunks = max(self.stop_chunks + 5, math.ceil(fallback_ms / chunk_ms))

        # Grace window after fallback
        self.fallback_grace_s = max(0.0, fallback_grace_ms / 1000.0)

        self.reset()
        self.is_speeching = False

    def reset(self) -> None:
        """Reset detector state."""
        self.state = TurnState.LISTENING
        self.pre_buffer = deque(maxlen=self.pre_chunks)
        self.segment = deque(maxlen=self.max_chunks)
        self.speech_active = False
        self.is_speeching = False
        self.trailing_silence = 0
        self.since_trigger_chunks = 0
        self.speech_start_time = None
        self._last_smart_turn_prob = None
        self._last_speech_silenced_time = None
        self.waiting_for_resume = False
        self.pre_speech_chunks = 0
        self._pending_turn_end_at: Optional[float] = None

    def on_speech_start(self) -> None:
        """Manual trigger (e.g. for forced barge-in)"""
        if not self.speech_active:
            self.state = TurnState.SPEAKING
            self.speech_active = True
            self.is_speeching = True
            self.speech_start_time = time.time()
            self.trailing_silence = 0
            self.since_trigger_chunks = 1

    async def process_chunk(
        self, audio_chunk_int16: np.ndarray, interruption_mode: bool = False
    ) -> Optional[TurnEvent]:
        """
        Process a 512-sample chunk of 16kHz audio.
        Returns TurnEvent if state changed, otherwise None.

        Args:
            audio_chunk_int16: Audio chunk as int16 numpy array
            interruption_mode: Whether interruption mode is enabled

        Returns:
            TurnEvent if state changed, None otherwise
        """
        current_time = time.time()

        # Scale to [-1, 1] float32 as expected by VAD
        f32 = (audio_chunk_int16.astype(np.float32)) / 32768.0

        # Energy gate to drop pure background noise
        energy_rms = np.sqrt(np.mean(f32**2))
        if energy_rms < self.energy_gate_rms:
            is_speech = False
        else:
            # Run Silero VAD inference
            silero_prob = await asyncio.to_thread(self._vad.prob, f32)
            is_speech = silero_prob > self.vad_threshold

        self.is_speeching = is_speech

        if not self.speech_active:
            # Listening mode
            self.pre_buffer.append(f32)
            if is_speech:
                self.pre_speech_chunks += 1
                if self.pre_speech_chunks >= self.pre_chunks:
                    self.speech_active = True
                    self.state = TurnState.SPEAKING
                    self.speech_start_time = current_time
                    self.trailing_silence = 0
                    self.since_trigger_chunks = self.pre_chunks
                    self.waiting_for_resume = False
                    self.segment.clear()
                    self.segment.extend(self.pre_buffer)

                    return TurnEvent(
                        state=TurnState.SPEAKING,
                        timestamp=current_time,
                        confidence=1.0,
                        silence_duration_ms=0.0,
                        speech_duration_ms=0.0
                    )
            else:
                self.pre_speech_chunks = 0
        else:
            # Speaking mode
            self.segment.append(f32)
            self.since_trigger_chunks += 1

            if is_speech:
                if getattr(self, "waiting_for_resume", False):
                    if self._pending_turn_end_at is not None:
                        logger.info(
                            "[TurnDetector] Speech RESUMED during fallback "
                            "grace — TURN_END suppressed, continuing same segment"
                        )
                    else:
                        logger.info("[TurnDetector] Speech RESUMED (User continued speaking)")
                self.trailing_silence = 0
                self.waiting_for_resume = False
                self._pending_turn_end_at = None
            else:
                self.trailing_silence += 1

            in_fallback_window = (
                getattr(self, "waiting_for_resume", False)
                and self.trailing_silence >= getattr(self, "max_silence_chunks", 48)
            )
            if in_fallback_window:
                if self._pending_turn_end_at is None:
                    self._pending_turn_end_at = current_time + self.fallback_grace_s
                    grace_ms = int(self.fallback_grace_s * 1000)
                    logger.warning(
                        f"[TurnDetector] Maximum silence fallback reached "
                        f"({self.trailing_silence} chunks). Opening {grace_ms}ms "
                        f"grace window for resume before forcing TURN_END..."
                    )
                elif current_time >= self._pending_turn_end_at:
                    logger.warning(
                        f"[TurnDetector] Fallback grace expired — forcing "
                        f"TURN_END despite Smart-Turn!"
                    )
                    self.state = TurnState.TURN_END
                    silence_ms = self.trailing_silence * (self.chunk_size / self.sample_rate) * 1000.0
                    speech_ms = (current_time - self.speech_start_time) * 1000.0 if self.speech_start_time else 0.0
                    prob = self._last_smart_turn_prob or 0.0

                    event = TurnEvent(
                        state=TurnState.TURN_END,
                        timestamp=getattr(self, "_last_speech_silenced_time", current_time) or current_time,
                        confidence=prob,
                        silence_duration_ms=silence_ms,
                        speech_duration_ms=speech_ms,
                        smart_turn_probability=prob
                    )
                    self.reset()
                    return event

            if self.trailing_silence >= self.stop_chunks and not self.waiting_for_resume:
                self._last_speech_silenced_time = current_time
                logger.info(f"[TurnDetector] Speech SILENCED (Pause detected >= {self.vad_silence_trigger_ms}ms). Evaluating Smart-Turn...")

                # Get speech probability using ML model or fallback
                if self._smart_turn is not None:
                    segment_audio = np.concatenate(self.segment, dtype=np.float32)
                    prob = await asyncio.to_thread(self._smart_turn.predict, segment_audio)
                else:
                    # Fallback: use simple heuristic
                    segment_audio = np.concatenate(self.segment, dtype=np.float32)
                    duration_ms = len(segment_audio) / self.sample_rate * 1000
                    # Simple heuristic: if speech was > 2 seconds, consider complete
                    prob = 0.8 if duration_ms > 2000 else 0.3
                    logger.info(f"[TurnDetector] Using fallback heuristic (duration: {duration_ms:.0f}ms)")

                self._last_smart_turn_prob = prob

                silence_ms = self.trailing_silence * (self.chunk_size / self.sample_rate) * 1000.0
                speech_ms = (current_time - self.speech_start_time) * 1000.0 if self.speech_start_time else 0.0

                logger.info(f"[TurnDetector] Probability check (len={len(segment_audio)} / {len(self.segment)} chunks): prob={prob:.3f}")

                if prob >= self.smart_turn_threshold:
                    # Turn complete!
                    self.state = TurnState.TURN_END
                    event = TurnEvent(
                        state=TurnState.TURN_END,
                        timestamp=self._last_speech_silenced_time or current_time,
                        confidence=prob,
                        silence_duration_ms=silence_ms,
                        speech_duration_ms=speech_ms,
                        smart_turn_probability=prob
                    )

                    self.reset()
                    return event
                else:
                    # Incomplete turn - user continues
                    logger.debug(f"[TurnDetector] INCOMPLETE (prob={prob:.3f}) - wait for user to speak again")
                    self.waiting_for_resume = True
                    return TurnEvent(
                        state=TurnState.SILENCED,
                        timestamp=self._last_speech_silenced_time or current_time,
                        confidence=prob,
                        silence_duration_ms=silence_ms,
                        speech_duration_ms=speech_ms,
                        smart_turn_probability=prob
                    )

        return None


def create_turn_detector(
    threshold: float = 0.70,
    use_ml: bool = True,
    vad_threshold: float = 0.5,
    num_threads: int = 4
):
    """
    Create a turn detector instance.

    Args:
        threshold: Probability threshold for turn completion (default 0.70)
        use_ml: Whether to use ML-based Smart Turn v3 (default True)
        vad_threshold: VAD speech threshold (default 0.5)
        num_threads: Number of threads for ONNX inference (default 4)

    Returns:
        TurnDetector instance
    """
    return TurnDetector(
        smart_turn_threshold=threshold,
        vad_threshold=vad_threshold,
        num_threads=num_threads
    )


if __name__ == "__main__":
    # Test the turn detector
    print("Testing TurnDetector...")
    detector = create_turn_detector()
    print(f"TurnDetector created successfully")
    print(f"  Smart Turn available: {detector._smart_turn is not None}")
    print(f"  Silero VAD available: {detector._vad is not None}")
