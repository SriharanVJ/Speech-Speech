"""
Voice Gateway - Complete End-to-End Voice Agent

Based on Z-agent-llm app/services/sandwich/sandwich_gateway.py

Architecture:
    Whisper STT → LLM Service → Kokoro TTS → Audio Output
    ↓
    Barge-in Support (Silero VAD)

Features:
- Complete LLM → TTS pipeline
- Streaming response generation
- Barge-in with generation tracking
- Audio queue management
- Tool calling support
"""

import asyncio
import json
import logging
import re
import time
import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, AsyncGenerator

# Import LLM service
try:
    from llm_service import (
        build_graph_from_config,
        stream_response,
        VoiceRequestParams,
        LLMConfig,
        get_voice_system_prompt,
    )
except ImportError:
    print("[VoiceGateway] ⚠️ llm_service not available")

# Import TTS service
try:
    from tts_service import get_tts_provider, BaseTTSService, SentenceChunker
except ImportError:
    print("[VoiceGateway] ⚠️ tts_service not available")

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class VoiceGatewayConfig:
    """Configuration for voice gateway."""
    # LLM Config
    llm_provider: str = "GeminiAI"
    llm_model: str = "gemini-2.5-flash"
    llm_api_key: str = ""
    llm_temperature: float = 0.7

    # TTS Config
    tts_provider: str = "kokoro"
    tts_host_url: str = "ws://localhost:8001"
    tts_voice_id: str = "default"
    tts_model_id: str = "a"
    tts_speed: float = 1.0

    # Voice Agent Config
    chatbot_id: str = "default"
    thread_id: str = ""
    custom_prompt: str = ""
    time_zone: str = "UTC"
    supported_languages: List[str] = None

    # Audio Config
    output_sample_rate: int = 24000
    leading_silence_ms: int = 200

    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = ["english"]


# =============================================================================
# VOICE GATEWAY
# =============================================================================


class VoiceGateway:
    """
    Complete end-to-end voice agent gateway.

    Manages:
    - Whisper STT → LLM → TTS pipeline
    - Barge-in state and detection
    - Audio queue and output
    - Conversation history
    """

    LEADING_SILENCE_MS = 200
    BARGE_IN_COOLDOWN_MS = 600

    def __init__(self, config: VoiceGatewayConfig):
        """
        Initialize voice gateway.

        Args:
            config: VoiceGatewayConfig instance
        """
        self.config = config

        # LLM and Graph
        self._graph = None
        self._graph_params = None

        # TTS Service
        self._tts_service: Optional[BaseTTSService] = None

        # State
        self._conversation_history: List[Dict] = []
        self._agent_speaking = False
        self._generation_id = 0
        self._last_pipeline_end_time = 0.0
        self._was_interrupted = False
        self._interrupted_ai_text = ""

        # Audio queue
        self._audio_queue: Optional[asyncio.Queue] = None
        self._interruption_event: Optional[asyncio.Event] = None

        logger.info(f"[VoiceGateway] Initialized with provider {config.llm_provider}")

    async def initialize(self) -> None:
        """Initialize LLM graph and TTS service."""
        print(f"[VoiceGateway] 🚀 Initializing voice gateway...")

        # Build LLM graph config
        llm_config = {
            "llm": {
                "provider": self.config.llm_provider,
                "model": self.config.llm_model,
                "api_key": self.config.llm_api_key,
                "temperature": self.config.llm_temperature,
            },
            "chatbot_id": self.config.chatbot_id,
            "thread_id": self.config.thread_id,
            "custom_prompt": self.config.custom_prompt,
            "time_zone": self.config.time_zone,
            "supported_languages": self.config.supported_languages,
            "static_tools": [],
            "kb_config": {},
            "tts_config": {},
            "tools": [],
            "vector_db": {},
            "embedding": {},
        }

        print(f"[VoiceGateway] 📝 LLM Config: provider={llm_config['llm']['provider']}, model={llm_config['llm']['model']}")
        print(f"[VoiceGateway] 🔑 API Key length: {len(llm_config['llm']['api_key'])}")

        # Build graph
        try:
            print(f"[VoiceGateway] 📊 Building LangGraph...")
            self._graph, self._graph_params = build_graph_from_config(llm_config)
            print(f"[VoiceGateway] ✅ LangGraph built successfully")
            logger.info(f"[VoiceGateway] LangGraph built successfully")
        except Exception as e:
            print(f"[VoiceGateway] ❌ Failed to build graph: {e}")
            import traceback
            traceback.print_exc()
            logger.error(f"[VoiceGateway] Failed to build graph: {e}")
            raise

        # Initialize TTS
        try:
            print(f"[VoiceGateway] 🔊 Initializing TTS service: {self.config.tts_provider}")
            self._tts_service = get_tts_provider(
                provider=self.config.tts_provider,
                host_url=self.config.tts_host_url,
                voice_id=self.config.tts_voice_id,
                model_id=self.config.tts_model_id,
                speed=self.config.tts_speed,
            )
            print(f"[VoiceGateway] ✅ TTS service initialized: {self.config.tts_provider}")
            logger.info(f"[VoiceGateway] TTS service initialized: {self.config.tts_provider}")
        except Exception as e:
            print(f"[VoiceGateway] ❌ Failed to initialize TTS: {e}")
            import traceback
            traceback.print_exc()
            logger.error(f"[VoiceGateway] Failed to initialize TTS: {e}")
            raise

        # Initialize audio queue and interruption event
        self._audio_queue = asyncio.Queue(maxsize=500)
        self._interruption_event = asyncio.Event()

        print(f"[VoiceGateway] ✅ Initialization complete")
        logger.info(f"[VoiceGateway] Initialization complete")

    async def process_user_turn(
        self,
        user_text: str,
        ws=None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a user turn through the complete pipeline.

        Args:
            user_text: User's transcribed text
            ws: Optional WebSocket for sending events

        Yields:
            Dict with events:
            - {"type": "llm_token", "text": "..."}
            - {"type": "tool_call", "tool": "..."}
            - {"type": "tts_audio", "data": bytes}
            - {"type": "transcription", "speaker": "assistant", "text": "..."}
            - {"type": "pipeline_done"}
            - {"type": "error", "error": "..."}
        """
        if not self._graph or not self._graph_params:
            yield {"type": "error", "error": "Gateway not initialized"}
            return

        # Add user message to history
        self._conversation_history.append({
            "role": "user",
            "content": user_text
        })

        pipeline_start = time.time()
        gen_id = self._generation_id + 1
        self._generation_id = gen_id

        # Set agent speaking state
        self._agent_speaking = True
        self._interruption_event.clear()

        logger.info(f"[VoiceGateway] [gen={gen_id}] Starting pipeline for: '{user_text[:50]}...'")

        try:
            chunker = SentenceChunker()
            sentence_idx = 0
            ai_text_so_far = ""

            # Stream LLM response
            async for llm_msg in stream_response(
                user_text,
                self._conversation_history,
                self._graph,
                self._graph_params,
                self._interruption_event,
            ):
                # Check for interruption
                if self._interruption_event.is_set():
                    self._interrupted_ai_text = ai_text_so_far
                    logger.info(f"[VoiceGateway] [gen={gen_id}] Interrupted during LLM")
                    return

                if llm_msg["type"] == "llm_token":
                    ai_text_so_far += llm_msg["text"]

                    # Check for complete sentences → TTS
                    sentences = chunker.add_token(llm_msg["text"])
                    for idx, sentence in enumerate(sentences):
                        if self._interruption_event.is_set():
                            return

                        logger.info(f"[VoiceGateway] [gen={gen_id}] Sentence #{sentence_idx}: {sentence[:60]}...")

                        # Generate TTS for this sentence
                        async for tts_msg in self._run_tts_for_sentence(
                            sentence,
                            sentence_idx,
                            gen_id,
                        ):
                            if tts_msg["type"] == "tts_audio":
                                # Queue audio for output
                                if self._audio_queue:
                                    try:
                                        await self._audio_queue.put((gen_id, tts_msg["data"]))
                                    except asyncio.QueueFull:
                                        logger.warning(f"[VoiceGateway] Audio queue full")

                                # Also yield for immediate processing
                                yield {
                                    "type": "tts_audio",
                                    "generation_id": gen_id,
                                    "data": tts_msg["data"],
                                }
                            elif tts_msg["type"] == "error":
                                logger.error(f"[VoiceGateway] TTS error: {tts_msg.get('error')}")

                        sentence_idx += 1

                elif llm_msg["type"] == "tool_call":
                    tool_name = llm_msg.get("tool", "unknown")
                    logger.info(f"[VoiceGateway] Tool call: {tool_name}")
                    yield {
                        "type": "tool_call",
                        "tool": tool_name,
                        "query": llm_msg.get("query", ""),
                    }

                elif llm_msg["type"] == "end_call":
                    logger.info(f"[VoiceGateway] end_call triggered")
                    yield {"type": "end_call"}

                elif llm_msg["type"] == "human_handoff":
                    logger.info(f"[VoiceGateway] human_handoff triggered")
                    yield {"type": "human_handoff"}

                elif llm_msg["type"] == "llm_done":
                    # Flush remaining text
                    remaining = chunker.flush()
                    if remaining:
                        async for tts_msg in self._run_tts_for_sentence(
                            remaining,
                            sentence_idx,
                            gen_id,
                        ):
                            if tts_msg["type"] == "tts_audio":
                                if self._audio_queue:
                                    try:
                                        await self._audio_queue.put((gen_id, tts_msg["data"]))
                                    except asyncio.QueueFull:
                                        pass

                                yield {
                                    "type": "tts_audio",
                                    "generation_id": gen_id,
                                    "data": tts_msg["data"],
                                }
                        sentence_idx += 1

                    # If LLM returned nothing, speak a fallback
                    if sentence_idx == 0:
                        fallback = "I'm sorry, could you repeat that?"
                        logger.warning(f"[VoiceGateway] [gen={gen_id}] LLM returned 0 sentences — using fallback")
                        async for tts_msg in self._run_tts_for_sentence(fallback, 0, gen_id):
                            if tts_msg["type"] == "tts_audio":
                                if self._audio_queue:
                                    try:
                                        await self._audio_queue.put((gen_id, tts_msg["data"]))
                                    except asyncio.QueueFull:
                                        pass

                                yield {
                                    "type": "tts_audio",
                                    "generation_id": gen_id,
                                    "data": tts_msg["data"],
                                }

                    # Send assistant transcription
                    full_text = llm_msg.get("full_text", "")
                    if full_text:
                        yield {
                            "type": "transcription",
                            "speaker": "assistant",
                            "text": full_text,
                            "timestamp": time.time(),
                        }

                        # Add to conversation history
                        self._conversation_history.append({
                            "role": "assistant",
                            "content": full_text
                        })

                    # Calculate timing
                    pipeline_end = time.time()
                    pipeline_total_ms = round((pipeline_end - pipeline_start) * 1000)

                    timing = llm_msg.get("timing", {})
                    usage = llm_msg.get("usage", {})

                    logger.info(
                        f"[VoiceGateway] [gen={gen_id}] Pipeline complete: "
                        f"stt=0ms, graph={timing.get('graph_total_ms')}ms, "
                        f"ttft={timing.get('llm_ttft_ms')}ms, "
                        f"total={pipeline_total_ms}ms"
                    )

            # Pipeline complete
            self._agent_speaking = False
            self._last_pipeline_end_time = time.time()

            yield {
                "type": "pipeline_done",
                "generation_id": gen_id,
            }

        except asyncio.CancelledError:
            self._interrupted_ai_text = ai_text_so_far
            logger.info(f"[VoiceGateway] [gen={gen_id}] Cancelled via barge-in")
            self._agent_speaking = False
            self._was_interrupted = True

        except Exception as e:
            logger.error(f"[VoiceGateway] [gen={gen_id}] Pipeline error: {e}")
            self._agent_speaking = False
            yield {
                "type": "error",
                "error": str(e),
            }

    async def _run_tts_for_sentence(
        self,
        sentence: str,
        sentence_idx: int,
        gen_id: int,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Run TTS for a single sentence.

        Args:
            sentence: Text to synthesize
            sentence_idx: Sentence index
            gen_id: Generation ID

        Yields:
            Dict with TTS audio or error
        """
        if self._interruption_event.is_set():
            return

        loop = asyncio.get_event_loop()

        def _stream_tts():
            """Runs in a thread — streams TTS and collects chunks."""
            for item in self._tts_service.stream_audio(sentence):
                if self._interruption_event.is_set():
                    return
                yield item

        # Run TTS in thread
        try:
            for item in self._tts_service.stream_audio(sentence):
                if self._interruption_event.is_set():
                    logger.info(f"[VoiceGateway] TTS #{sentence_idx} cancelled mid-stream")
                    return

                if item["type"] == "tts_audio":
                    yield item
                elif item["type"] == "error":
                    logger.error(f"[VoiceGateway] TTS error: {item.get('error')}")

        except Exception as e:
            logger.error(f"[VoiceGateway] TTS streaming error: {e}")
            yield {
                "type": "error",
                "error": str(e),
            }

    def trigger_barge_in(self) -> bool:
        """
        Trigger barge-in if agent is speaking.

        Returns:
            True if barge-in was triggered, False otherwise
        """
        if not self._agent_speaking:
            return False

        # Check cooldown
        if time.time() - self._last_pipeline_end_time < (self.BARGE_IN_COOLDOWN_MS / 1000):
            logger.debug("[VoiceGateway] Ignoring barge-in — too soon after pipeline end")
            return False

        logger.info(f"[VoiceGateway] 🔴 Barge-in triggered (gen={self._generation_id})")

        # Set interruption event
        if self._interruption_event:
            self._interruption_event.set()

        # Clear audio queue
        if self._audio_queue:
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

        # Reset state
        self._agent_speaking = False
        self._was_interrupted = True

        return True

    def clear_interruption_event(self) -> None:
        """Clear the interruption event."""
        if self._interruption_event:
            self._interruption_event.clear()

    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history."""
        return self._conversation_history.copy()

    def get_generation_id(self) -> int:
        """Get current generation ID."""
        return self._generation_id

    def is_agent_speaking(self) -> bool:
        """Check if agent is currently speaking."""
        return self._agent_speaking

    def was_interrupted(self) -> bool:
        """Check if last turn was interrupted."""
        return self._was_interrupted

    def get_interrupted_ai_text(self) -> str:
        """Get the AI text that was interrupted."""
        return self._interrupted_ai_text

    def reset_interruption_state(self) -> None:
        """Reset interruption state for new turn."""
        self._was_interrupted = False
        self._interrupted_ai_text = ""


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_voice_gateway(config: VoiceGatewayConfig) -> VoiceGateway:
    """
    Create a voice gateway instance.

    Args:
        config: VoiceGatewayConfig instance

    Returns:
        VoiceGateway instance
    """
    gateway = VoiceGateway(config)
    logger.info(f"[VoiceGateway] Created gateway with config: {config.chatbot_id}")
    return gateway
