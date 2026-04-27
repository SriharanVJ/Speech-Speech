"""
Speech-to-Speech System with Barge-In Capability

Full pipeline with interruptible TTS:
- WebSocket receives mic audio
- VAD detects speech in real-time
- STT transcribes audio (Whisper/Parakeet)
- LLM generates response (Gemini)
- TTS synthesizes speech (Kokoro)
- BARGE-IN: TTS stops immediately when user speaks

Usage:
    python main.py

WebSocket:
    ws://localhost:8000/ws
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
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ==============================================================================
# CONFIGURATION
# ==============================================================================

SAMPLE_RATE = 16000
CHANNELS = 1
BYTES_PER_SAMPLE = 2  # PCM16

# VAD Settings
VAD_THRESHOLD = 0.5
VAD_DEBOUNCE_MS = 200  # Minimum speech duration to trigger barge-in
FRAME_DURATION_MS = 10
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 160 samples
FRAME_BYTES = FRAME_SIZE * BYTES_PER_SAMPLE  # 320 bytes

# Service ports
WHISPER_PORT = int(os.environ.get("WHISPER_PORT", 8006))
KOKORO_PORT = int(os.environ.get("KOKORO_PORT", 8001))
GEMINI_PORT = int(os.environ.get("GEMINI_PORT", 8002))

# Service URLs
WHISPER_URL = f"ws://localhost:{WHISPER_PORT}/stream/asr"
KOKORO_URL = f"ws://localhost:{KOKORO_PORT}/stream/tts"
GEMINI_URL = f"ws://localhost:{GEMINI_PORT}/ws"

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# VAD - Voice Activity Detection
# ==============================================================================

class VADDetector:
    """
    Voice Activity Detection with debouncing.
    Uses Silero VAD if available, falls back to energy-based VAD.
    """

    def __init__(self, threshold: float = VAD_THRESHOLD, debounce_ms: int = VAD_DEBOUNCE_MS):
        self.threshold = threshold
        self.debounce_frames = int(debounce_ms / FRAME_DURATION_MS)
        self.speech_frames = 0
        self.vad = None

        # Try to load Silero VAD
        try:
            from silero_vad import load_silero_vad
            self.model = load_silero_vad()
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.use_silero = True
            self.speech_buffer = np.array([], dtype=np.float32)
            logger.info(f"[VAD] Using Silero VAD (threshold={threshold})")
        except ImportError:
            self.use_silero = False
            logger.info(f"[VAD] Using energy-based VAD (threshold={threshold})")

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Detect if audio chunk contains speech."""
        if self.use_silero:
            return self._silero_vad(audio_chunk)
        else:
            return self._energy_vad(audio_chunk)

    def _silero_vad(self, audio: np.ndarray) -> bool:
        """Silero VAD with debouncing."""
        # Ensure float32 normalized [-1, 1]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / 32768.0

        # Buffer to minimum 512 samples for Silero
        self.speech_buffer = np.concatenate([self.speech_buffer, audio])

        if len(self.speech_buffer) < 512:
            return self.speech_frames > 0

        # Run VAD on first 512 samples
        audio_tensor = torch.from_numpy(self.speech_buffer[:512])
        if torch.cuda.is_available():
            audio_tensor = audio_tensor.cuda()

        with torch.no_grad():
            prob = self.model(audio_tensor, SAMPLE_RATE).item()

        is_speech_frame = prob > self.threshold
        self.speech_buffer = self.speech_buffer[512:]

        # Debouncing
        if is_speech_frame:
            self.speech_frames = self.debounce_frames
        elif self.speech_frames > 0:
            self.speech_frames -= 1
            is_speech_frame = True

        return is_speech_frame

    def _energy_vad(self, audio: np.ndarray) -> bool:
        """Simple energy-based VAD with debouncing."""
        # Compute RMS energy
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / 32768.0

        rms = np.sqrt(np.mean(audio ** 2))
        normalized_energy = min(rms * 5, 1.0)

        is_speech_frame = normalized_energy > self.threshold

        # Debouncing
        if is_speech_frame:
            self.speech_frames = self.debounce_frames
        elif self.speech_frames > 0:
            self.speech_frames -= 1
            is_speech_frame = True

        return is_speech_frame

    def reset(self) -> None:
        """Reset VAD state."""
        self.speech_frames = 0
        self.speech_buffer = np.array([], dtype=np.float32)


# ==============================================================================
# SERVICE CLIENTS
# ==============================================================================

class WhisperClient:
    """WebSocket client for Whisper ASR service."""

    def __init__(self, url: str = WHISPER_URL, language: str = "auto", temperature: float = 0.0):
        # Build URL with query parameters
        from urllib.parse import urlencode
        params = {
            "language": language,
            "temperature": str(temperature),
            "use_vad": "false",  # We handle VAD in the main orchestrator
            "silence_flush": "false",  # We handle buffering in the main orchestrator
        }
        self.url = f"{url}?{urlencode(params)}"
        self.ws = None

    async def connect(self) -> bool:
        """Connect to Whisper service."""
        try:
            from websockets.client import connect
            logger.info(f"🔌 [WHISPER CLIENT] Connecting to {self.url}...")
            self.ws = await connect(self.url)
            # Wait for ready message
            response = await self.ws.recv()
            msg = json.loads(response)
            logger.info(f"📨 [WHISPER CLIENT] Received: {msg}")
            if msg.get("type") != "ready":
                logger.warning(f"⚠️  [WHISPER CLIENT] Unexpected first message: {msg}")
            logger.info(f"✅ [WHISPER CLIENT] Connected successfully")
            return True
        except Exception as e:
            logger.error(f"❌ [WHISPER CLIENT] Connect failed: {e}")
            return False

    async def transcribe(self, audio_data: bytes) -> Optional[str]:
        """Transcribe audio data."""
        if not self.ws:
            return None

        try:
            # Send audio data
            await self.ws.send(audio_data)

            # Receive transcription result
            while True:
                response = await asyncio.wait_for(self.ws.recv(), timeout=10.0)

                # Handle both text and binary responses
                if isinstance(response, str):
                    msg = json.loads(response)
                else:
                    # Skip binary responses (not expected from new_whisper.py)
                    continue

                msg_type = msg.get("type")

                if msg_type == "segment_completed":
                    text = msg.get("text", "")
                    if text:
                        return text
                elif msg_type == "complete":
                    # ✅ FIX: Don't break on "complete" - wait for next segment
                    # This keeps the connection alive for multiple transcriptions
                    logger.info(f"[WHISPER] Got complete message, waiting for next transcription...")
                    continue
                elif msg_type == "ready":
                    continue
                elif msg_type == "turn_end":
                    # Turn ended, wait for next segment
                    continue

        except asyncio.TimeoutError:
            logger.warning("[WHISPER] Transcription timeout")
            return None
        except Exception as e:
            logger.error(f"[WHISPER] Error: {e}")
            return None

    async def close(self) -> None:
        """Close connection."""
        if self.ws:
            try:
                await self.ws.send(json.dumps({"type": "done"}))
            except:
                pass
            await self.ws.close()


class GeminiClient:
    """WebSocket client for Gemini LLM service."""

    def __init__(self, url: str = GEMINI_URL):
        self.url = url
        self.ws = None

    async def connect(self) -> bool:
        """Connect to Gemini service."""
        try:
            from websockets.client import connect
            logger.info(f"🔌 [GEMINI CLIENT] Connecting to {self.url}...")
            self.ws = await connect(self.url)
            # Wait for ready message
            response = await self.ws.recv()
            msg = json.loads(response)
            logger.info(f"📨 [GEMINI CLIENT] Received: {msg}")
            if msg.get("type") != "ready":
                logger.warning(f"⚠️  [GEMINI CLIENT] Unexpected first message: {msg}")
            logger.info(f"✅ [GEMINI CLIENT] Connected successfully")
            return True
        except Exception as e:
            logger.error(f"❌ [GEMINI CLIENT] Connect failed: {e}")
            return False

    async def get_response(self, text: str) -> Optional[str]:
        """Get LLM response for text."""
        if not self.ws:
            logger.error(f"❌ [GEMINI CLIENT] Not connected!")
            return "Sorry, the AI service is not connected. Please try reconnecting."

        try:
            # Send query
            logger.info(f"📤 [GEMINI CLIENT] Sending query: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
            await self.ws.send(json.dumps({
                "type": "query",
                "text": text
            }))

            # Receive response
            logger.info(f"📥 [GEMINI CLIENT] Waiting for response (timeout: 30s)...")
            while True:
                response = await asyncio.wait_for(self.ws.recv(), timeout=30.0)
                msg = json.loads(response)
                logger.info(f"📨 [GEMINI CLIENT] Received message type: {msg.get('type')}")

                if msg.get("type") == "response":
                    response_text = msg.get("text", "")
                    logger.info(f"✅ [GEMINI CLIENT] Got response: \"{response_text[:100]}{'...' if len(response_text) > 100 else ''}\"")
                    return response_text
                elif msg.get("type") == "error":
                    error_msg = msg.get("message", "Unknown error")
                    logger.error(f"❌ [GEMINI CLIENT] Error: {error_msg}")
                    # Return error as text so it can be spoken via TTS
                    return f"Sorry, there was an error: {error_msg}"

        except asyncio.TimeoutError:
            logger.warning(f"⏰ [GEMINI CLIENT] Response timeout after 30s")
            # Return timeout message so it can be spoken via TTS
            return "Sorry, the AI service timed out. Please try again."
        except Exception as e:
            logger.error(f"❌ [GEMINI CLIENT] Error: {e}")
            # Return error message so it can be spoken via TTS
            return f"Sorry, there was an error connecting to the AI service: {str(e)}"

    async def close(self) -> None:
        """Close connection."""
        if self.ws:
            await self.ws.close()


class KokoroClient:
    """WebSocket client for Kokoro TTS service."""

    def __init__(self, url: str = KOKORO_URL, voice: str = None, speed: float = 1.0, lang_code: str = None):
        # Build URL with query parameters
        from urllib.parse import urlencode
        params = {}
        if voice:
            params["voice"] = voice
        if speed != 1.0:
            params["speed"] = str(speed)
        if lang_code:
            params["lang_code"] = lang_code

        if params:
            self.url = f"{url}?{urlencode(params)}"
        else:
            self.url = url

        self.ws = None
        self.active = False
        self.stop_requested = False

    async def connect(self) -> bool:
        """Connect to Kokoro service."""
        try:
            from websockets.client import connect
            logger.info(f"🔌 [KOKORO CLIENT] Connecting to {self.url}...")
            self.ws = await connect(self.url)
            # Wait for ready message
            response = await self.ws.recv()
            msg = json.loads(response)
            logger.info(f"📨 [KOKORO CLIENT] Received: {msg}")
            if msg.get("type") != "ready":
                logger.warning(f"⚠️  [KOKORO CLIENT] Unexpected first message: {msg}")
            logger.info(f"✅ [KOKORO CLIENT] Connected successfully")
            return True
        except Exception as e:
            logger.error(f"❌ [KOKORO CLIENT] Connect failed: {e}")
            return False

    async def synthesize_stream(self, text: str, send_audio_callback: Callable[[bytes], None]) -> None:
        """
        Stream TTS audio for text.
        send_audio_callback is called for each audio chunk.
        Can be interrupted by setting stop_requested = True.
        """
        if not self.ws:
            logger.error(f"❌ [KOKORO CLIENT] Not connected!")
            return

        self.active = True
        self.stop_requested = False
        chunk_count = 0

        try:
            # Send text for synthesis
            logger.info(f"📤 [KOKORO CLIENT] Sending text for synthesis: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
            await self.ws.send(json.dumps({
                "type": "text",
                "content": text
            }))

            # Send done signal
            logger.info(f"📤 [KOKORO CLIENT] Sending done signal...")
            await self.ws.send(json.dumps({"type": "done"}))

            # Receive audio chunks
            logger.info(f"📥 [KOKORO CLIENT] Waiting for audio chunks...")
            while self.active:
                try:
                    response = await asyncio.wait_for(self.ws.recv(), timeout=1.0)

                    # Check if binary audio data
                    if isinstance(response, bytes):
                        if self.stop_requested:
                            logger.info(f"🛑 [KOKORO CLIENT] Stop requested, breaking...")
                            break
                        chunk_count += 1
                        if chunk_count == 1:
                            logger.info(f"🔊 [KOKORO CLIENT] Receiving audio chunks...")
                        if chunk_count % 10 == 0:
                            logger.info(f"🔊 [KOKORO CLIENT] Received {chunk_count} audio chunks...")
                        await send_audio_callback(response)

                    # JSON response
                    else:
                        msg = json.loads(response)
                        logger.info(f"📨 [KOKORO CLIENT] Received message type: {msg.get('type')}")
                        if msg.get("type") == "complete":
                            logger.info(f"✅ [KOKORO CLIENT] Synthesis complete, received {chunk_count} audio chunks")
                            break
                        elif msg.get("type") == "error":
                            logger.error(f"❌ [KOKORO CLIENT] Error: {msg.get('message')}")
                            break

                except asyncio.TimeoutError:
                    # Check stop flag
                    if self.stop_requested:
                        break
                    continue

        except Exception as e:
            logger.error(f"❌ [KOKORO CLIENT] Error: {e}")
        finally:
            self.active = False
            self.stop_requested = False

    def stop(self) -> None:
        """Stop current synthesis."""
        self.stop_requested = True
        self.active = False

    async def close(self) -> None:
        """Close connection."""
        self.stop()
        if self.ws:
            await self.ws.close()


# ==============================================================================
# AUDIO BUFFER
# ==============================================================================


# ==============================================================================
# CONNECTION MANAGER
# ==============================================================================

class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        """Accept new connection."""
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """Remove connection."""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)


manager = ConnectionManager()

# ==============================================================================
# FASTAPI APP
# ==============================================================================

app = FastAPI(title="Speech-to-Speech with Barge-In")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Serve the web UI."""
    from pathlib import Path
    html_file = Path(__file__).parent / "index.html"
    if html_file.exists():
        from fastapi.responses import FileResponse
        return FileResponse(html_file)
    return {
        "service": "Speech-to-Speech with Barge-In",
        "version": "1.0.0",
        "websocket_endpoint": "ws://localhost:8000/ws",
        "services": {
            "whisper_url": WHISPER_URL,
            "kokoro_url": KOKORO_URL,
            "gemini_url": GEMINI_URL,
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


# ==============================================================================
# MAIN WEBSOCKET ENDPOINT
# ==============================================================================

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    Main WebSocket endpoint for Speech-to-Speech with barge-in.

    Protocol:
    - Client sends PCM16 audio chunks (16kHz, mono)
    - Server detects speech via VAD
    - Streams audio to Whisper in real-time
    - Pipeline: STT → LLM → TTS
    - Audio chunks sent back to client for playback
    """
    await manager.connect(ws)

    # Helper function to send logs to client
    async def send_log(level: str, message: str):
        """Send log message to client."""
        try:
            await ws.send_json({
                "type": "log",
                "level": level,
                "message": message
            })
        except Exception:
            pass  # Ignore errors if client disconnected

    async def log_info(message: str):
        """Log to both console and client."""
        logger.info(message)
        await send_log('info', message)

    async def log_success(message: str):
        """Log to both console and client."""
        logger.info(message)
        await send_log('success', message)

    async def log_warning(message: str):
        """Log to both console and client."""
        logger.warning(message)
        await send_log('warning', message)

    async def log_error(message: str):
        """Log to both console and client."""
        logger.error(message)
        await send_log('error', message)

    # Initialize services
    vad = VADDetector(threshold=VAD_THRESHOLD)

    # Service clients
    whisper_client = WhisperClient()
    gemini_client = GeminiClient()
    kokoro_client = KokoroClient()

    # Connect to services
    await log_info("🔌 Connecting to services...")
    whisper_ok = await whisper_client.connect()
    gemini_ok = await gemini_client.connect()
    kokoro_ok = await kokoro_client.connect()

    if not all([whisper_ok, gemini_ok, kokoro_ok]):
        await ws.send_json({
            "type": "error",
            "message": "Failed to connect to one or more services"
        })
        manager.disconnect(ws)
        await ws.close()
        return

    # Send ready message
    await ws.send_json({
        "type": "ready",
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
        "vad_threshold": VAD_THRESHOLD,
    })

    # Pipeline state
    pipeline_task: Optional[asyncio.Task] = None
    is_pipeline_running = False
    is_streaming_to_whisper = False
    waiting_for_whisper_response = False  # Track when we're waiting for Whisper response
    whisper_response_timeout_frames = 0
    listener_task: Optional[asyncio.Task] = None  # Track the listener task

    session_id = str(uuid.uuid4())[:8]
    request_count = 0

    # Silence detection for Whisper flush
    silence_frames = 0
    SPEECH_END_SILENCE_FRAMES = 120  # ✅ FIX 1: 1200ms of silence (120 * 10ms) to allow natural pauses and longer sentences
    WHISPER_RESPONSE_TIMEOUT = 100  # 100 frames (1 second) before giving up on Whisper response

    # ✅ FIX 2 & 3: Accumulate segments instead of processing immediately
    accumulated_text = ""  # Store all segments until turn_end

    logger.info(f"[SESSION {session_id}] Started")

    async def send_audio_to_client(audio_bytes: bytes):
        """Send audio chunk to client."""
        try:
            await ws.send_bytes(audio_bytes)
        except Exception as e:
            logger.error(f"[SEND] Error: {e}")

    async def run_pipeline(text: str):
        """
        Run the full STT → LLM → TTS pipeline.
        Can be interrupted if user speaks again.
        """
        nonlocal is_pipeline_running, waiting_for_whisper_response

        if is_pipeline_running:
            await log_warning(f"⚠️  [PIPELINE] Already running, skipping new request")
            return

        is_pipeline_running = True

        try:
            await log_info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            await log_info(f"📋 [PIPELINE] Starting pipeline with input: \"{text}\"")
            await log_info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

            # Step 1: Send text to client (show user speech)
            await log_info(f"📤 [CLIENT] Sending transcribed text to web client...")
            await ws.send_json({
                "type": "llm_text",
                "data": text
            })

            # Step 2: Get LLM response
            await log_info(f"🤖 [GEMINI] Sending text to Gemini LLM...")
            await log_info(f"📥 [GEMINI] Waiting for Gemini response...")

            llm_response = await gemini_client.get_response(text)

            if not llm_response:
                await log_warning(f"⚠️  [GEMINI] No response received!")
                # Send a default error message to TTS
                llm_response = "Sorry, the AI service didn't respond. Please try again."

            await log_success(f"✅ [GEMINI] Response received: \"{llm_response[:100]}{'...' if len(llm_response) > 100 else ''}\"")
            await log_info(f"📤 [CLIENT] Sending Gemini response to web client...")

            # Send response text to client
            await ws.send_json({
                "type": "llm_response",
                "data": llm_response
            })

            # Step 3: Stream TTS audio
            await log_info(f"🔊 [KOKORO] Sending text to TTS for audio synthesis...")
            await log_info(f"📥 [KOKORO] Streaming audio chunks to client...")

            await kokoro_client.synthesize_stream(llm_response, send_audio_to_client)

            await log_success(f"✅ [KOKORO] Audio synthesis complete")
            await log_info(f"📤 [CLIENT] Sending TTS complete message...")

            await ws.send_json({
                "type": "tts_complete"
            })

            await log_info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            await log_success("✅ [PIPELINE] Pipeline completed successfully!")
            await log_info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        except Exception as e:
            await log_error(f"❌ [PIPELINE] Error: {e}")
            traceback.print_exc()
        finally:
            # ✅ FIX 1 & 5: Reset pipeline state and prepare for next turn
            is_pipeline_running = False
            waiting_for_whisper_response = False
            await log_info(f"🔄 [PIPELINE] Pipeline state reset - ready for next turn")

    async def listen_for_whisper_responses():
        """Listen for transcription responses from Whisper while streaming audio."""
        nonlocal is_streaming_to_whisper, request_count, waiting_for_whisper_response, pipeline_task, listener_task, accumulated_text

        try:
            await log_info("🎤 [WHISPER LISTENER] Started listening for Whisper responses...")
            listen_count = 0
            # ✅ FIX: Keep listening as long as WebSocket is open, not just while streaming
            while whisper_client.ws and not whisper_client.ws.closed:
                try:
                    listen_count += 1
                    if listen_count % 10 == 0:
                        await log_info(f"🎧 [WHISPER LISTENER] Still listening... (iteration {listen_count})")

                    response = await asyncio.wait_for(whisper_client.ws.recv(), timeout=1.0)

                    # Handle both text and binary responses
                    if isinstance(response, str):
                        msg = json.loads(response)
                    else:
                        await log_info(f"📦 [WHISPER] Received binary data ({len(response)} bytes, ignored)")
                        continue

                    msg_type = msg.get("type")
                    await log_info(f"📨 [WHISPER] Received message type: {msg_type}")

                    if msg_type == "segment_completed":
                        text = msg.get("text", "")
                        if text:
                            waiting_for_whisper_response = False
                            await log_success(f"✅ [STT → TEXT] Segment received: \"{text}\"")

                            # ✅ FIX 2 & 3: Accumulate text instead of processing immediately
                            # segment_completed ≠ final user sentence
                            # (accumulated_text is already declared as nonlocal at function top)
                            accumulated_text += (" " if accumulated_text else "") + text

                            await log_info(f"📝 [ACCUMULATOR] Text so far: \"{accumulated_text}\"")
                            await log_info(f"📤 [STT] Sending segment to client for display...")

                            await ws.send_json({
                                "type": "stt_text",
                                "data": text,
                                "accumulated": accumulated_text  # Send accumulated text for UI display
                            })

                            # ✅ FIX 2: DO NOT call run_pipeline here
                            # Wait for turn_end before processing
                            await log_info(f"⏳ [ACCUMULATOR] Waiting for more segments or turn_end...")

                            # ✅ CRITICAL FIX: Don't exit listener, just continue listening
                            # Don't break, don't return - just keep the listener alive
                            await log_info(f"🎧 [WHISPER] Listener continuing to listen for more transcriptions...")
                            continue

                    elif msg_type == "complete":
                        waiting_for_whisper_response = False
                        await log_success(f"✅ [WHISPER] Stream complete - keeping connection alive for next transcription")
                        # ✅ FIX: Reset streaming state but keep connection alive
                        is_streaming_to_whisper = False
                        # Don't break - keep listening for next transcription
                        continue

                    elif msg_type == "ready":
                        await log_info(f"✅ [WHISPER] Ready message received")
                        continue

                    elif msg_type == "turn_end":
                        # ✅ FIX 4: turn_end is the ONLY trigger to call LLM
                        # Turn ended by server (max duration or silence)
                        await log_info(f"🔄 [WHISPER] Turn ended by server - processing accumulated text")
                        waiting_for_whisper_response = False
                        is_streaming_to_whisper = False

                        # ✅ FIX 3 & 4: Process accumulated text on turn_end
                        # (accumulated_text is already declared as nonlocal at function top)
                        if accumulated_text.strip():
                            request_count += 1
                            final_text = accumulated_text.strip()
                            await log_success(f"✅ [TURN END] Full sentence: \"{final_text}\"")

                            # Send final accumulated text to client
                            await ws.send_json({
                                "type": "turn_complete",
                                "data": final_text
                            })

                            # Cancel previous pipeline if still running
                            if pipeline_task and not pipeline_task.done():
                                await log_info(f"⚠️ [PIPELINE] Cancelling previous pipeline...")
                                pipeline_task.cancel()

                            # ✅ FIX 4: NOW call run_pipeline with the full accumulated text
                            await log_info(f"🚀 [PIPELINE] Starting STT → LLM → TTS pipeline with full sentence...")
                            pipeline_task = asyncio.create_task(run_pipeline(final_text))

                            # Reset accumulator for next turn
                            accumulated_text = ""
                            await log_info(f"🔄 [ACCUMULATOR] Reset for next turn")
                        else:
                            await log_info(f"⚠️ [TURN END] No accumulated text to process")

                        continue

                    elif msg_type == "ping":
                        # Keep-alive ping from server
                        await log_info(f"💓 [WHISPER] Keep-alive ping received")
                        continue

                    else:
                        await log_info(f"📨 [WHISPER] Other message: {msg}")

                except asyncio.TimeoutError:
                    await log_info(f"⏳ [WHISPER LISTENER] Timeout waiting for message (iteration {listen_count})")
                    continue
                except Exception as e:
                    await log_error(f"❌ [WHISPER LISTEN] Error: {e}")
                    is_streaming_to_whisper = False
                    waiting_for_whisper_response = False
                    # ✅ FIX: Don't break - try to recover
                    await log_info(f"🔄 [WHISPER] Listener attempting to recover...")
                    await asyncio.sleep(1)  # Wait a bit before retrying
                    continue  # Try to continue listening

        except Exception as e:
            await log_error(f"❌ [WHISPER LISTEN] Fatal error: {e}")
            traceback.print_exc()
            is_streaming_to_whisper = False
            waiting_for_whisper_response = False
            # Listener will exit, but main loop should restart it

    try:
        await log_info(f"🎬 [MAIN] Starting to receive audio from client...")
        frame_count = 0
        last_receive_time = time.time()
        connection_timeout = 300.0  # ✅ FIX: 5 minutes timeout (was implicit 0.1s)
        while ws.client_state == WebSocketState.CONNECTED:
            try:
                # Receive audio chunk with longer timeout
                data = await asyncio.wait_for(ws.receive(), timeout=2.0)
                last_receive_time = time.time()  # ✅ FIX: Update last activity time

                if "bytes" in data:
                    frame_count += 1
                    audio_chunk = data["bytes"]
                    if frame_count % 100 == 0:  # Log every 100 frames
                        await log_info(f"📥 [AUDIO] Received {frame_count} frames ({len(audio_chunk)} bytes each)")

                    # Convert to numpy for VAD
                    audio_int16 = np.frombuffer(audio_chunk, dtype=np.int16)
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0

                    # 🎤 Always check speech (even during TTS)
                    if vad.is_speech(audio_float32):

                        # Reset silence counter
                        silence_frames = 0

                        # 🔥 BARGE-IN: Stop everything
                        if is_pipeline_running:
                            await log_info("🔥 [BARGE-IN] User spoke, stopping TTS and cancelling pipeline...")

                            # Stop TTS immediately
                            kokoro_client.stop()

                            # Cancel pipeline task
                            if pipeline_task and not pipeline_task.done():
                                pipeline_task.cancel()

                            await ws.send_json({
                                "type": "barge_in",
                                "message": "TTS interrupted by user speech"
                            })

                        # Start streaming to Whisper if not already streaming
                        if not is_streaming_to_whisper:
                            is_streaming_to_whisper = True
                            await log_info(f"🎤 [VAD] Speech detected! Starting Whisper stream...")

                            # ✅ FIX 2: Start listening for responses in background (if not already running)
                            if listener_task is None or listener_task.done():
                                listener_task = asyncio.create_task(listen_for_whisper_responses())
                                await log_info(f"🎧 [WHISPER] Listener task started")
                            else:
                                await log_info(f"🎧 [WHISPER] Listener task already running, continuing...")

                        # Stream audio chunk to Whisper
                        if whisper_client.ws and not whisper_client.ws.closed:
                            await whisper_client.ws.send(audio_chunk)
                            if frame_count % 50 == 0:  # Log every 50 speech frames
                                await log_info(f"📤 [WHISPER] Streaming audio frame to Whisper...")

                    else:
                        # Silence detected
                        # If we're streaming to Whisper and have been silent for a while, send "done"
                        if is_streaming_to_whisper:
                            silence_frames += 1
                            if silence_frames == 10:
                                await log_info(f"🤫 [VAD] Silence detected ({silence_frames}/50 frames)...")
                            if silence_frames >= SPEECH_END_SILENCE_FRAMES:
                                await log_success(f"✅ [VAD] Speech ended after {SPEECH_END_SILENCE_FRAMES} frames of silence")
                                await log_info(f"📤 [WHISPER] Sending 'done' signal to flush audio buffer...")
                                if whisper_client.ws and not whisper_client.ws.closed:
                                    await whisper_client.ws.send(json.dumps({"type": "done"}))
                                waiting_for_whisper_response = True
                                whisper_response_timeout_frames = 0
                                silence_frames = 0

                elif "text" in data:
                    try:
                        msg = json.loads(data["text"])

                        if msg.get("type") == "ping":
                            await ws.send_json({"type": "pong"})

                        elif msg.get("type") == "reset":
                            vad.reset()
                            kokoro_client.stop()
                            is_streaming_to_whisper = False
                            waiting_for_whisper_response = False
                            await ws.send_json({"type": "reset_complete"})

                    except json.JSONDecodeError:
                        pass

            except asyncio.TimeoutError:
                # ✅ FIX: Check for connection timeout (no data for 5 minutes)
                if time.time() - last_receive_time > connection_timeout:
                    await log_warning(f"⏰ [MAIN] Connection timeout - no data for {connection_timeout}s")
                    break

                # Check for Whisper response timeout even when no data received
                if waiting_for_whisper_response:
                    whisper_response_timeout_frames += 1
                    if whisper_response_timeout_frames >= WHISPER_RESPONSE_TIMEOUT:
                        await log_error(f"⏰ [WHISPER] Timeout waiting for response after {WHISPER_RESPONSE_TIMEOUT} frames")
                        is_streaming_to_whisper = False
                        waiting_for_whisper_response = False
                        whisper_response_timeout_frames = 0

                # ✅ FIX: Send keep-alive ping every 30 seconds
                if time.time() - last_receive_time > 30:
                    try:
                        await ws.send_json({"type": "ping"})
                        await log_info(f"💓 [MAIN] Keep-alive ping sent")
                    except Exception:
                        await log_error(f"❌ [MAIN] Failed to send keep-alive - connection may be dead")
                        break

                continue

    except WebSocketDisconnect:
        logger.info(f"[SESSION {session_id}] Client disconnected")

    except Exception as e:
        logger.error(f"[ERROR] {e}")
        traceback.print_exc()

    finally:
        # Cleanup
        is_streaming_to_whisper = False

        if pipeline_task and not pipeline_task.done():
            pipeline_task.cancel()

        kokoro_client.stop()

        # ✅ FIX: Log before closing services
        await log_info(f"🧹 [CLEANUP] Closing service connections...")
        await whisper_client.close()
        await gemini_client.close()
        await kokoro_client.close()
        await log_info(f"✅ [CLEANUP] All services closed")

        manager.disconnect(ws)

        try:
            await ws.close()
        except:
            pass

        logger.info(f"[SESSION {session_id}] Ended")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")

    print("=" * 60)
    print("  Speech-to-Speech with Barge-In")
    print("=" * 60)
    print(f"  Host: {host}")
    print(f"  Port: {port}")
    print(f"  WebSocket: ws://{host}:{port}/ws")
    print()
    print("  Dependencies:")
    print(f"    - Whisper ASR: {WHISPER_URL}")
    print(f"    - Kokoro TTS: {KOKORO_URL}")
    print(f"    - Gemini LLM: {GEMINI_URL}")
    print()
    print("  VAD:")
    print(f"    - Threshold: {VAD_THRESHOLD}")
    print(f"    - Debounce: {VAD_DEBOUNCE_MS}ms")
    print("=" * 60)
    print()

    uvicorn.run(app, host=host, port=port)
