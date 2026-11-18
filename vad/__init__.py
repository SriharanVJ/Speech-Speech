"""
VAD and Turn Detection Module

Provides ML-based voice activity detection and semantic turn completion
using Silero VAD v4 and Smart Turn v3 from pipecat-ai.

Components:
- SileroVAD: Real-time speech activity detection
- SmartTurnModel: Semantic turn completion prediction
- TurnDetector: Combined VAD + turn completion detector

Usage:
    from vad import TurnDetector, create_turn_detector

    detector = create_turn_detector(threshold=0.70)
    event = await detector.process_chunk(audio_chunk)
    if event and event.state == TurnState.TURN_END:
        print("Turn complete!")
"""

from .silero_vad import SileroVAD, download_silero_vad_model
from .smart_turn_model import SmartTurnModel, download_smart_turn_model
from .turn_detector import TurnDetector, TurnState, TurnEvent, create_turn_detector
from .config import VAD_CONFIG, TURN_DETECTION_CONFIG

__all__ = [
    # Main classes
    "TurnDetector",
    "SileroVAD",
    "SmartTurnModel",

    # Factory functions
    "create_turn_detector",
    "download_silero_vad_model",
    "download_smart_turn_model",

    # Enums and dataclasses
    "TurnState",
    "TurnEvent",

    # Config
    "VAD_CONFIG",
    "TURN_DETECTION_CONFIG",
]
