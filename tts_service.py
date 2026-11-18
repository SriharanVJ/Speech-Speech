"""
TTS Service for Voice Agent - Kokoro Only

Based on Z-agent-llm app/services/sandwich/tts_service.py

Provider:
- Kokoro (local streaming TTS)

Features:
- Sentence chunking for streaming
- Barge-in support with abort()
- WebSocket streaming
- Speed control
"""

import asyncio
import logging
import time
import threading
from abc import ABC, abstractmethod
from typing import Generator, Optional, Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)


class BaseTTSService(ABC):
    """Base class for all TTS streaming providers."""

    @abstractmethod
    def stream_audio(self, text: str) -> Generator[dict, None, None]:
        """
        Stream TTS audio chunks for the given text.

        Yields dicts:
          {"type": "tts_audio", "data": bytes}  — PCM/chunk
          {"type": "tts_done", "timing": {...}}  — completion with timing
          {"type": "error", "error": "..."}      — error
        """
        pass

    def abort(self) -> None:
        """
        Force-close any in-flight network resource (e.g. WebSocket)
        Called from the gateway on barge-in.

        Default is a no-op for providers that don't hold a long-lived socket.
        """
        return


class SentenceChunker:
    """
    Accumulates LLM tokens and yields complete sentences
    for sentence-level TTS streaming.
    """

    SENTENCE_ENDINGS = {'.', '?', '!', '。', '？', '！'}

    def __init__(self):
        self.buffer = ""

    def add_token(self, token: str) -> list[str]:
        """
        Add a token and return any complete sentences found.

        Returns a list of complete sentences (may be empty).
        """
        # Preprocess TTS-unfriendly characters
        token = token.replace("—", " ").replace("--", " ").replace("*", " ")

        self.buffer += token
        sentences = []

        while True:
            # Find the earliest sentence ending
            earliest_pos = -1
            for ending in self.SENTENCE_ENDINGS:
                pos = self.buffer.find(ending)
                if pos != -1 and (earliest_pos == -1 or pos < earliest_pos):
                    earliest_pos = pos

            if earliest_pos == -1:
                break

            sentence_end = earliest_pos + 1

            # Keep contiguous punctuation
            while sentence_end < len(self.buffer) and self.buffer[sentence_end] in self.SENTENCE_ENDINGS:
                sentence_end += 1

            # Skip if very short (likely abbreviation)
            sentence = self.buffer[:sentence_end].strip()
            if len(sentence) < 5 and earliest_pos + 1 < len(self.buffer):
                remaining = self.buffer[sentence_end:]
                if remaining.strip():
                    self.buffer = remaining
                    continue
                else:
                    break

            sentences.append(sentence)
            self.buffer = self.buffer[sentence_end:].lstrip()

        return sentences

    def flush(self) -> Optional[str]:
        """Return any remaining buffered text."""
        remaining = self.buffer.strip()
        self.buffer = ""
        return remaining if remaining else None

    def reset(self):
        """Clear the buffer."""
        self.buffer = ""


# =============================================================================
# KOKORO (LOCAL) TTS
# =============================================================================


class KokoroTTSService(BaseTTSService):
    """
    Kokoro local TTS via WebSocket.
    Connects to local Kokoro TTS server.
    """

    def __init__(
        self,
        host_url: str = "ws://localhost:8001",
        voice_id: str = "default",
        model_id: str = "a",  # Lang code
        speed: float = 1.0,
    ):
        self.host_url = host_url
        self.voice_id = voice_id
        self.lang_code = model_id
        self.speed = speed
        self._active_ws = None

        logger.info(
            f"[TTS] Kokoro initialized: host={host_url}, voice={voice_id}, lang={model_id}, speed={speed}"
        )

    def stream_audio(self, text: str) -> Generator[dict, None, None]:
        from websockets.sync.client import connect
        import json

        target_url = f"{self.host_url}/stream/tts?speed={self.speed}&voice={self.voice_id}&lang_code={self.lang_code}"

        tts_start = time.time()
        first_chunk_time = None
        total_bytes = 0
        chunk_count = 0

        try:
            with connect(target_url) as websocket:
                self._active_ws = websocket
                payload = {
                    "type": "text",
                    "content": text
                }
                websocket.send(json.dumps(payload))

                while True:
                    response = websocket.recv()

                    if isinstance(response, bytes):
                        if first_chunk_time is None:
                            first_chunk_time = time.time()

                        chunk_len = len(response)
                        total_bytes += chunk_len
                        chunk_count += 1

                        yield {
                            "type": "tts_audio",
                            "data": response,
                        }
                    elif isinstance(response, str):
                        try:
                            msg = json.loads(response)
                            if msg.get("type") == "complete":
                                break
                        except json.JSONDecodeError:
                            pass

            tts_end = time.time()
            total_ms = round((tts_end - tts_start) * 1000)
            ttfa_ms = (
                round((first_chunk_time - tts_start) * 1000)
                if first_chunk_time
                else None
            )

            yield {
                "type": "tts_done",
                "timing": {
                    "tts_total_ms": total_ms,
                    "tts_ttfa_ms": ttfa_ms,
                    "tts_bytes": total_bytes,
                    "tts_chunks": chunk_count,
                },
            }

            logger.info(
                f"[TTS] Kokoro Complete: {len(text)} chars → {total_bytes} bytes in {chunk_count} chunks, "
                f"total={total_ms}ms, ttfa={ttfa_ms}ms"
            )

        except Exception as e:
            logger.error(f"[TTS] Kokoro Streaming error: {e}")
            yield {
                "type": "error",
                "error": f"Kokoro TTS error: {str(e)}",
            }
        finally:
            self._active_ws = None

    def abort(self) -> None:
        ws = self._active_ws
        if ws is None:
            return
        try:
            ws.close()
            logger.info("[TTS] Kokoro WebSocket force-closed on abort")
        except Exception as e:
            logger.warning(f"[TTS] Kokoro abort close failed: {e}")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def get_tts_provider(
    provider: str = "kokoro",
    host_url: str = "ws://localhost:8001",
    voice_id: str = "default",
    model_id: str = "a",
    speed: float = 1.0,
    **kwargs
) -> BaseTTSService:
    """
    Factory to get the TTS provider (Kokoro only).

    Args:
        provider: Provider name (only "kokoro" supported)
        host_url: Kokoro server WebSocket URL
        voice_id: Voice ID
        model_id: Lang code
        speed: Speed multiplier
        **kwargs: Additional provider-specific params

    Returns:
        BaseTTSService instance (KokoroTTSService)
    """
    provider = provider.lower()

    if provider == "kokoro":
        return KokoroTTSService(
            host_url=host_url,
            voice_id=voice_id,
            model_id=model_id,
            speed=speed,
        )
    else:
        raise ValueError(f"Unsupported TTS provider: {provider}. Only 'kokoro' is supported.")
