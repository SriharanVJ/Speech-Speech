"""
Parakeet ASR - Simple WebSocket Speech-to-Text Server

Features:
- NVIDIA NeMo Parakeet for transcription
- Simple streaming - no VAD, no buffering

Usage:
    python new_parakeet.py

WebSocket:
    ws://localhost:8005/stream/asr
"""

import asyncio
import json
import logging
import os
import time
import traceback
import uuid
from typing import Any, Dict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState
import uvicorn

# Try importing NeMo ASR
try:
    import nemo.collections.asr as nemo_asr
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    print("[INFO] NeMo not installed. Install with: pip install nemo-toolkit[asr]")

SAMPLE_RATE = 16000
CHANNELS = 1

logger = logging.getLogger(__name__)


# ==============================================================================
# PARAKEET ASR SERVICE (NeMo)
# ==============================================================================


class ParakeetASRService:
    """Speech-to-Text service using NVIDIA NeMo Parakeet model."""

    def __init__(
        self,
        model_name: str = "nvidia/multitalker-parakeet-streaming-0.6b-v1",
        device: str = "cuda",
    ):
        if not NEMO_AVAILABLE:
            raise RuntimeError(
                "NeMo not available. Install with: pip install nemo-toolkit[asr]"
            )

        self.model_name = model_name
        self.device = device
        self.model = None

        print(f"[ParakeetASR] Model: {model_name}, Device: {device}")

    def load(self) -> None:
        """Load the NeMo Parakeet model."""
        self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name)

        # Move to CUDA if available and requested
        if self.device == "cuda" and torch.cuda.is_available():
            self.model = self.model.cuda()

        # Set model to eval mode
        self.model.eval()

        print(f"[ParakeetASR] Model loaded successfully")

    def transcribe(
        self,
        audio: np.ndarray,
        batch_size: int = 1,
    ) -> Dict[str, Any]:
        """Transcribe audio using NeMo Parakeet."""
        if not self.model:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            # Resample if needed (assuming 16kHz input)
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)

            # Normalize audio to [-1, 1] if needed
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / 32768.0

            # Add batch dimension if needed
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]

            # NeMo transcribe expects a list of file paths or numpy arrays
            with torch.no_grad():
                transcriptions = self.model.transcribe(
                    audio,
                    batch_size=batch_size,
                    return_hypotheses=False,
                )

            # Extract text from transcriptions
            if isinstance(transcriptions, list) and len(transcriptions) > 0:
                text = transcriptions[0].strip() if isinstance(transcriptions[0], str) else str(transcriptions[0])
            else:
                text = str(transcriptions).strip() if transcriptions else ""

            duration = audio.shape[1] / SAMPLE_RATE

            return {
                "text": text,
                "segments": [{"text": text, "start": 0.0, "end": duration}],
            }

        except Exception as e:
            print(f"[ParakeetASR ERROR] {e}")
            traceback.print_exc()
            return {
                "text": "",
                "segments": [],
                "error": str(e),
            }


# ==============================================================================
# FASTAPI APP
# ==============================================================================


app = FastAPI(title="Parakeet ASR")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize ASR service on startup."""
    model_path = os.environ.get("PARAKEET_MODEL", "nvidia/multitalker-parakeet-streaming-0.6b-v1")
    device = os.environ.get("PARAKEET_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 50)
    print("  Parakeet ASR - Starting")
    print(f"  Model: {model_path}")
    print(f"  Device: {device}")
    print("=" * 50)

    # Load Parakeet ASR service
    asr_service = ParakeetASRService(
        model_name=model_path,
        device=device,
    )
    asr_service.load()

    app.state.asr_service = asr_service
    app.state.device = device

    port = os.environ.get("PORT", "8005")
    print(f"  Ready! ws://localhost:{port}/stream/asr")
    print()


@app.get("/")
async def index():
    """Root endpoint."""
    return {
        "service": "Parakeet ASR",
        "status": "running",
        "backend": "nemo",
    }


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "model_loaded": app.state.asr_service is not None,
    }


@app.get("/config")
async def config():
    """Get server configuration."""
    service: ParakeetASRService = app.state.asr_service
    return {
        "model": service.model_name,
        "device": service.device,
        "backend": "nemo",
        "sample_rate": SAMPLE_RATE,
    }


# ==============================================================================
# WEBSOCKET ASR ENDPOINT
# ==============================================================================


@app.websocket("/stream/asr")
async def websocket_asr(ws: WebSocket) -> None:
    """
    WebSocket endpoint for speech-to-text.

    Simply receives audio chunks and transcribes them.
    """
    client_host = ws.client.host if ws.client else "unknown"
    print(f"[ASR] WebSocket connection from {client_host}")
    await ws.accept()

    service: ParakeetASRService = app.state.asr_service

    if not service:
        print(f"[ASR] ERROR: ASR service not available")
        await ws.send_json({"type": "error", "message": "ASR service not available"})
        await ws.close(code=1011, reason="ASR not available")
        return

    # Audio buffer for accumulating received chunks
    audio_buffer = bytearray()
    segment_count = 0
    session_id = str(uuid.uuid4())[:8]
    print(f"[ASR] Session ID: {session_id}")

    # Send ready message
    await ws.send_json({
        "type": "ready",
        "sample_rate": SAMPLE_RATE,
        "channels": CHANNELS,
    })

    try:
        while ws.client_state == WebSocketState.CONNECTED:
            try:
                data = await asyncio.wait_for(ws.receive(), timeout=0.1)

                if "bytes" in data:
                    # Accumulate audio data
                    audio_chunk = data["bytes"]
                    audio_buffer.extend(audio_chunk)

                    # Convert accumulated audio to numpy array
                    audio_int16 = np.frombuffer(bytes(audio_buffer), dtype=np.int16)
                    audio_float32 = audio_int16.astype(np.float32) / 32768.0

                    # Check if we have enough audio (minimum 0.5 seconds)
                    min_samples = int(SAMPLE_RATE * 0.5)
                    if len(audio_float32) >= min_samples:
                        segment_count += 1
                        duration = len(audio_float32) / SAMPLE_RATE
                        segment_start_time = time.time()

                        print(f"[SEGMENT #{segment_count}] transcribing {duration:.2f}s of audio...")

                        # Transcribe
                        result = service.transcribe(audio=audio_float32)
                        transcribe_time = time.time() - segment_start_time

                        text = result.get("text", "").strip()

                        if text:
                            print(f"[SEGMENT #{segment_count}] text: \"{text}\"")
                            print(f"[LATENCY] total: {transcribe_time*1000:.1f}ms | audio: {duration*1000:.1f}ms")

                            await ws.send_json({
                                "type": "segment_completed",
                                "text": text,
                                "segment": segment_count,
                                "is_final": True,
                            })

                        # Clear buffer after processing
                        audio_buffer = bytearray()

                elif "text" in data:
                    try:
                        msg = json.loads(data["text"])
                        print(f"[ASR] RECEIVED JSON: {msg}")

                        if msg.get("type") == "done":
                            break

                    except json.JSONDecodeError:
                        pass

            except asyncio.TimeoutError:
                continue

    except WebSocketDisconnect as e:
        print(f"[ASR] WebSocket disconnected - code: {e.code}, reason: {e.reason}")

    except Exception as e:
        print(f"[ASR ERROR] {e}")
        traceback.print_exc()

    finally:
        # Process any remaining audio in buffer
        if len(audio_buffer) > 0:
            audio_int16 = np.frombuffer(bytes(audio_buffer), dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0

            # Only process if we have meaningful audio
            min_samples = int(SAMPLE_RATE * 0.3)
            if len(audio_float32) >= min_samples:
                try:
                    duration = len(audio_float32) / SAMPLE_RATE
                    print(f"[FINAL SEGMENT] transcribing {duration:.2f}s of audio...")

                    result = service.transcribe(audio=audio_float32)
                    text = result.get("text", "").strip()

                    if text:
                        await ws.send_json({
                            "type": "segment_completed",
                            "text": text,
                            "segment": "final",
                            "is_final": True,
                        })
                        print(f'[FINAL SEGMENT] text: "{text}"')

                except Exception as e:
                    print(f"[ASR] Error processing final segment: {e}")

        # Send complete message
        await ws.send_json({"type": "complete"})
        print(f"[ASR] Session complete")

        try:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.close()
        except:
            pass

    print(f"[ASR] Connection closed")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parakeet ASR - Simple Transcription"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8005, help="Port")
    parser.add_argument("--model", type=str, default="nvidia/multitalker-parakeet-streaming-0.6b-v1", help="Model")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device")

    args = parser.parse_args()

    os.environ["PARAKEET_MODEL"] = args.model
    os.environ["PARAKEET_DEVICE"] = args.device

    uvicorn.run("new_parakeet:app", host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
