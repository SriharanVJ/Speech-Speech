import asyncio
import json
import os
import sys
import logging
import threading
import traceback
import io
import re
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from starlette.websockets import WebSocketDisconnect, WebSocketState
from pydantic import BaseModel

# Kokoro TTS
from kokoro import KPipeline

logger = logging.getLogger(__name__)

BASE = Path(__file__).parent
SAMPLE_RATE = 24_000
CHANNELS = 1
DTYPE = "float32"
BYTES_PER_SAMPLE = 4
CHUNK_SIZE = 1024


# ============= Kokoro TTS Service =============

def remove_emojis(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002700-\U000027BF"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub("", text)

def  split_numbers(text: str) -> str:
    return re.sub(r'\d+', lambda m: " ".join(list(m.group())), text)


class KokoroTTSService:
    """
    Kokoro TTS Service following the same pattern as StreamingTTSService.
    Supports multiple languages by maintaining separate KPipeline instances.
    """
    def __init__(
        self,
        lang_codes: list = None,
        device: str = "cuda",
    ) -> None:
        self.lang_codes = lang_codes or ["a", "b", "h", "j"]
        self.sample_rate = SAMPLE_RATE
        self.pipelines: Dict[str, KPipeline] = {}
        self.voice_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self.default_voice: str = "af_heart"
        self.available_voices = [
            # American English
            "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
            # American English Male
            "am_michael", "am_adam", "am_echo",
            # British English
            "bf_emma", "bf_george", "bf_isabella",
            # British English Male
            "bm_lewis", "bm_daniel",
            # Indian English
            "if_sara",
            # Indian English Male
            "im_richard",
            # Japanese
            "jf_alpha", "jf_kumo", "jf_teagan", "jf_gongitsune",
            # Japanese Male
            "jm_kuma",
        ]
        self.default_voices_by_lang = {
            "a": "af_heart",
            "b": "bf_emma",
            "h": "if_sara",
            "j": "jf_alpha",
        }

        if device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS not available. Falling back to CPU.")
            device = "cpu"
        if device == "mpx":
            device = "mps"
        self.device = device
        self._torch_device = torch.device(device)

        # Disable gradients for inference
        torch.set_grad_enabled(False)

        # Initialize voice cache for each language
        for lang_code in self.lang_codes:
            self.voice_cache[lang_code] = {}

    @property
    def lang_code(self) -> str:
        """Return the primary language code (for compatibility)."""
        return self.lang_codes[0] if self.lang_codes else "a"

    def load(self) -> None:
        """Load the Kokoro pipelines for all configured languages."""
        for lang_code in self.lang_codes:
            print(f"[startup] Loading Kokoro pipeline for lang_code: {lang_code}")
            try:
                self.pipelines[lang_code] = KPipeline(
                    lang_code=lang_code,
                    device=self.device,
                    trf=False
                )
                print(f"[startup] Kokoro pipeline for {lang_code} loaded successfully")
            except Exception as e:
                print(f"[startup] Warning: Failed to load pipeline for {lang_code}: {e}")
        print(f"[startup] Total pipelines loaded: {len(self.pipelines)}")

    def _get_pipeline(self, lang_code: Optional[str] = None) -> Optional[KPipeline]:
        """Get the appropriate pipeline for the language code."""
        if lang_code and lang_code in self.pipelines:
            return self.pipelines[lang_code]
        # Fallback to first available pipeline
        return self.pipelines[self.lang_codes[0]] if self.pipelines else None

    def get_voice(self, voice: str, lang_code: Optional[str] = None) -> Optional[torch.Tensor]:
        """Get voice embedding, with caching."""
        pipeline = self._get_pipeline(lang_code)
        if not pipeline:
            return None

        cache = self.voice_cache[lang_code or self.lang_codes[0]]
        if voice not in cache:
            print(f"[voice] Loading voice: {voice} for lang: {lang_code or self.lang_codes[0]}")
            v = pipeline.load_voice(voice)
            # Keep voice on CPU - Kokoro handles device transfer internally
            cache[voice] = v
        return cache[voice]

    def _clean_text(self, text: str) -> list:
        """Clean input text for TTS."""

        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)

        # Remove markdown / formatting symbols
        text = re.sub(r'[*_`#~>|]', '', text)

        # Remove only http/https protocol (keep domain)
        text = re.sub(r'https?://', '', text)

        # Replace slashes so TTS doesn't say "slash"
        text = text.replace("/", " ")

        # Remove emojis
        text = remove_emojis(text)

        # Convert numbers to single digits
        text = split_numbers(text)

        # Handle literal "\n"
        text = text.replace("\\n", " ")

        # Replace real new lines
        text = text.replace("\n", " ")

        # Normalize spaces
        text = re.sub(r'\s+', ' ', text)

        # Split lines (for Kokoro pipeline compatibility)
        lines = [line.strip() for line in text.split(".") if line.strip()]

        return lines

    def _create_wav_header(self) -> bytes:
        """Create WAV header for streaming."""
        header = io.BytesIO()

        header.write(b"RIFF")
        header.write((0).to_bytes(4, "little"))
        header.write(b"WAVE")

        header.write(b"fmt ")
        header.write((16).to_bytes(4, "little"))
        header.write((3).to_bytes(2, "little"))  # float32
        header.write((CHANNELS).to_bytes(2, "little"))
        header.write((SAMPLE_RATE).to_bytes(4, "little"))

        byte_rate = SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE
        header.write((byte_rate).to_bytes(4, "little"))
        header.write((CHANNELS * BYTES_PER_SAMPLE).to_bytes(2, "little"))
        header.write((BYTES_PER_SAMPLE * 8).to_bytes(2, "little"))

        header.write(b"data")
        header.write((0).to_bytes(4, "little"))
        return header.getvalue()

    def stream(
        self,
        text: str,
        voice: Optional[str] = None,
        speed: float = 1.0,
        lang_code: Optional[str] = None,
        log_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[np.ndarray]:
        """
        Stream audio chunks from Kokoro TTS.

        Args:
            text: Input text to synthesize
            voice: Voice preset name (e.g., 'af_heart', 'am_michael')
            speed: Speech speed multiplier
            lang_code: Language code (overrides default)
            log_callback: Optional callback for logging events
            stop_event: Optional event to stop generation

        Yields:
            np.ndarray: Audio chunks as float32 arrays
        """
        if not text.strip():
            return

        # Determine language code and pipeline
        target_lang = lang_code if lang_code in self.pipelines else self.lang_codes[0]
        pipeline = self._get_pipeline(target_lang)

        if not pipeline:
            raise RuntimeError(f"No pipeline available for lang_code: {target_lang}")

        # Select voice based on language
        default_voice = self.default_voices_by_lang.get(target_lang, self.default_voice)
        selected_voice = voice if voice and voice in self.available_voices else default_voice
        voice_embedding = self.get_voice(selected_voice, target_lang)

        if voice_embedding is None:
            raise RuntimeError(f"Failed to load voice: {selected_voice}")

        def emit(event: str, **payload: Any) -> None:
            if log_callback:
                try:
                    log_callback(event, **payload)
                except Exception as exc:
                    print(f"[log_callback] Error while emitting {event}: {exc}")

        emit("generation_start", voice=selected_voice, lang_code=target_lang, text_length=len(text))

        # Clean text
        text_segments = self._clean_text(text)
        if not text_segments:
            return

        combined_text = " ".join(text_segments)

        try:
            print(f"[Kokoro] Starting generation: lang={target_lang}, voice={selected_voice}, speed={speed}, text_len={len(combined_text)}")

            # Kokoro generator call with language-specific pipeline
            generator = pipeline(
                combined_text,
                voice=voice_embedding,
                speed=speed
            )

            generated_samples = 0
            result_count = 0

            for result in generator:
                result_count += 1
                print(f"[Kokoro] Got result #{result_count}: {type(result)}, has_audio={hasattr(result, 'audio')}")
                if stop_event and stop_event.is_set():
                    break

                if hasattr(result, "audio") and result.audio is not None:
                    audio_tensor = result.audio.cpu().numpy().astype(DTYPE)
                    print(f"[Kokoro] Audio chunk shape: {audio_tensor.shape}, dtype: {audio_tensor.dtype}")

                    # Handle multi-channel audio
                    if audio_tensor.ndim > 1:
                        audio_tensor = audio_tensor.reshape(-1)

                    # Normalize if needed
                    peak = np.max(np.abs(audio_tensor)) if audio_tensor.size else 0.0
                    if peak > 1.0:
                        audio_tensor = audio_tensor / peak

                    generated_samples += int(audio_tensor.size)
                    emit(
                        "model_progress",
                        generated_sec=generated_samples / self.sample_rate,
                        chunk_sec=audio_tensor.size / self.sample_rate,
                    )

                    yield audio_tensor.astype(np.float32, copy=False)

        except Exception as e:
            emit("generation_error", message=str(e))
            raise e

    def chunk_to_pcm16(self, chunk: np.ndarray) -> bytes:
        """Convert float32 audio chunk to PCM16 bytes."""
        chunk = np.clip(chunk, -1.0, 1.0)
        pcm = (chunk * 32767.0).astype(np.int16)
        return pcm.tobytes()


# ============= FastAPI App =============

app = FastAPI(title="Kokoro TTS Server")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _startup() -> None:
    # Get languages from env or use defaults (American English, British English, Hindi, Japanese)
    lang_codes_str = os.environ.get("KOKORO_LANG_CODES", "a,b,h,j")
    lang_codes = [lc.strip() for lc in lang_codes_str.split(",")]
    device = os.environ.get("KOKORO_DEVICE", "cuda")

    service = KokoroTTSService(lang_codes=lang_codes, device=device)
    service.load()

    app.state.tts_service = service
    app.state.lang_codes = lang_codes
    app.state.device = device
    print(f"[startup] Kokoro TTS Model ready. Loaded languages: {', '.join(service.pipelines.keys())}")


@app.get("/")
def index():
    return FileResponse(BASE / "index.html")


@app.get("/config")
def get_config():
    service: KokoroTTSService = app.state.tts_service

    config = {
        "voices": service.available_voices,
        "default_voice": service.default_voice,
        "lang_codes": service.lang_codes,
        "loaded_languages": list(service.pipelines.keys()),
        "device": service.device,
    }

    return config


@app.get("/languages")
def get_supported_languages():
    """Get supported languages and voices for Kokoro TTS."""
    service: KokoroTTSService = app.state.tts_service

    # Kokoro supported language codes
    language_info = {
        "a": {
            "code": "a",
            "name": "American English",
            "native_name": "English (US)",
            "flag": "🇺🇸",
            "voices": [
                {"id": "af_heart", "name": "Heart", "gender": "female"},
                {"id": "af_bella", "name": "Bella", "gender": "female"},
                {"id": "af_nicole", "name": "Nicole", "gender": "female"},
                {"id": "af_sarah", "name": "Sarah", "gender": "female"},
                {"id": "af_sky", "name": "Sky", "gender": "female"},
                {"id": "am_michael", "name": "Michael", "gender": "male"},
                {"id": "am_adam", "name": "Adam", "gender": "male"},
                {"id": "am_echo", "name": "Echo", "gender": "male"},
            ]
        },
        "b": {
            "code": "b",
            "name": "British English",
            "native_name": "English (UK)",
            "flag": "🇬🇧",
            "voices": [
                {"id": "bf_emma", "name": "Emma", "gender": "female"},
                {"id": "bf_george", "name": "George", "gender": "male"},
                {"id": "bf_isabella", "name": "Isabella", "gender": "female"},
                {"id": "bm_lewis", "name": "Lewis", "gender": "male"},
                {"id": "bm_daniel", "name": "Daniel", "gender": "male"},
            ]
        },
        "e": {
            "code": "e",
            "name": "Spanish",
            "native_name": "Español",
            "flag": "🇪🇸",
            "voices": []
        },
        "f": {
            "code": "f",
            "name": "French",
            "native_name": "Français",
            "flag": "🇫🇷",
            "voices": []
        },
        "h": {
            "code": "h",
            "name": "Hindi",
            "native_name": "हिन्दी",
            "flag": "🇮🇳",
            "voices": [
                {"id": "if_sara", "name": "Sara", "gender": "female"},
                {"id": "im_richard", "name": "Richard", "gender": "male"},
            ]
        },
        "i": {
            "code": "i",
            "name": "Italian",
            "native_name": "Italiano",
            "flag": "🇮🇹",
            "voices": []
        },
        "j": {
            "code": "j",
            "name": "Japanese",
            "native_name": "日本語",
            "flag": "🇯🇵",
            "voices": [
                {"id": "jf_alpha", "name": "Alpha", "gender": "female"},
                {"id": "jf_kumo", "name": "Kumo", "gender": "female"},
                {"id": "jf_teagan", "name": "Teagan", "gender": "female"},
                {"id": "jf_gongitsune", "name": "Gongitsune", "gender": "female"},
                {"id": "jm_kuma", "name": "Kuma", "gender": "male"},
            ]
        },
        "k": {
            "code": "k",
            "name": "Korean",
            "native_name": "한국어",
            "flag": "🇰🇷",
            "voices": []
        },
        "p": {
            "code": "p",
            "name": "Portuguese",
            "native_name": "Português",
            "flag": "🇵🇹",
            "voices": []
        },
        "z": {
            "code": "z",
            "name": "Chinese",
            "native_name": "中文",
            "flag": "🇨🇳",
            "voices": []
        },
    }

    # Build response
    languages_list = []
    for lang_code, info in language_info.items():
        languages_list.append({
            "code": info["code"],
            "name": info["name"],
            "native_name": info["native_name"],
            "flag": info["flag"],
            "voice_count": len(info["voices"]),
            "voices": info["voices"],
        })

    return {
        "supported_languages": languages_list,
        "total_languages": len(language_info),
        "total_voices": len(service.available_voices),
        "default_voice": service.default_voice,
        "loaded_languages": list(service.pipelines.keys()),
        "model_info": {
            "name": "Kokoro TTS",
            "sample_rate": SAMPLE_RATE,
            "note": "Kokoro supports multiple languages with the 'lang_code' parameter."
        }
    }


# ============= TTS WebSocket Endpoint =============

@app.websocket("/stream/tts")
async def websocket_tts_streaming(ws: WebSocket) -> None:
    """WebSocket endpoint for streaming Kokoro TTS, compatible with VibeVoice client protocol."""
    await ws.accept()

    service: KokoroTTSService = app.state.tts_service

    try:
        # Get query parameters
        speed = float(ws.query_params.get("speed", 1.0))
        voice_param = ws.query_params.get("voice")
        lang_code = ws.query_params.get("lang_code")

        event_loop = asyncio.get_running_loop()

        # Punctuation marks that trigger generation
        PUNCTUATION_MARKS = {'.', ',', '!', '?', ';', ':', '\n'}

        # State for streaming
        text_buffer = ""
        generation_active = False
        generation_thread = None
        audio_queue: asyncio.Queue = asyncio.Queue()
        stop_signal = threading.Event()
        chunk_count = 0
        text_chunks_received = 0

        def generate(text_to_generate: str):
            """Generate audio for the given text."""
            nonlocal chunk_count
            try:
                for audio_chunk in service.stream(
                    text_to_generate,
                    voice=voice_param,
                    speed=speed,
                    lang_code=lang_code,
                    stop_event=stop_signal,
                ):
                    if stop_signal.is_set():
                        break
                    asyncio.run_coroutine_threadsafe(audio_queue.put(audio_chunk), event_loop)

                # Small silence tail
                silence = np.zeros(int(0.03 * SAMPLE_RATE), dtype=np.float32)
                asyncio.run_coroutine_threadsafe(audio_queue.put(silence), event_loop)
            except Exception as e:
                print(f"[TTS ERROR] {e}")
            finally:
                asyncio.run_coroutine_threadsafe(audio_queue.put(None), event_loop)

        def should_trigger_generation(text: str) -> bool:
            """Check if text ends with punctuation."""
            if not text:
                return False
            stripped = text.rstrip()
            if not stripped:
                return False
            return stripped[-1] in PUNCTUATION_MARKS

        def start_generation(text: str):
            """Start generation in a separate thread."""
            nonlocal generation_active, generation_thread, chunk_count
            if text.strip() and not generation_active:
                print(f"[TTS] Starting generation for text chunk: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
                generation_active = True
                stop_signal.clear()
                chunk_count = 0
                generation_thread = threading.Thread(
                    target=generate,
                    args=(text,),
                    daemon=True
                )
                generation_thread.start()

        await ws.send_json({"type": "ready"})
        print(f"[TTS] SENT JSON: {{\"type\": \"ready\"}}")

        while ws.client_state == WebSocketState.CONNECTED:
            # Check for incoming audio to send
            try:
                audio = await asyncio.wait_for(audio_queue.get(), timeout=0.01)
                if audio is None:
                    # Generation complete
                    print(f"[TTS] Generation complete - sent {chunk_count} audio chunks for this segment")
                    response = {"type": "complete"}
                    await ws.send_json(response)
                    print(f"[TTS] SENT JSON: {response}")
                    # Reset state for next cycle
                    generation_active = False
                    if generation_thread:
                        generation_thread.join(timeout=1)
                        generation_thread = None
                    stop_signal.clear()
                    chunk_count = 0
                    continue
                payload = service.chunk_to_pcm16(audio)
                await ws.send_bytes(payload)
                chunk_count += 1
                print(f"[TTS] Sent audio chunk #{chunk_count} ({len(payload)} bytes)")
            except asyncio.TimeoutError:
                pass

            # Check for incoming messages from client
            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=0.01)
            except asyncio.TimeoutError:
                continue

            msg_type = msg.get("type")

            if msg_type == "text":
                # Accumulate text and check for punctuation trigger
                chunk = msg.get("content", "")
                text_chunks_received += 1
                print(f"[TTS] Received text chunk #{text_chunks_received}: \"{chunk[:50]}{'...' if len(chunk) > 50 else ''}\"")
                text_buffer += chunk

                # Check if buffer ends with punctuation
                if should_trigger_generation(text_buffer):
                    start_generation(text_buffer)
                    text_buffer = ""

            elif msg_type == "done":
                print(f"[TTS] Received 'done' signal (total text chunks: {text_chunks_received})")
                # Force generate any remaining text in buffer
                if text_buffer.strip():
                    start_generation(text_buffer)
                    text_buffer = ""
                elif not generation_active:
                    # No text to generate, just send complete
                    response = {"type": "complete"}
                    await ws.send_json(response)
                    print(f"[TTS] SENT JSON: {response}")

            elif msg_type == "clear":
                # Clear the text buffer without generating
                text_buffer = ""

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[TTS ERROR] {e}")
        traceback.print_exc()
    finally:
        stop_signal.set()
        if generation_thread:
            generation_thread.join(timeout=1)
        try:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.close()
        except:
            pass


# ============= HTTP Streaming Endpoint (for direct testing) =============

from fastapi.responses import StreamingResponse
from fastapi import Query
import soundfile as sf


@app.get("/stream/tts_http")
async def http_tts_stream(
    text: str = Query(..., description="Text to synthesize"),
    voice: str = Query(None, description="Voice preset"),
    lang_code: str = Query(None, description="Language code"),
    speed: float = Query(1.0, description="Speech speed")
):
    """HTTP streaming endpoint for Kokoro TTS (returns audio/wav)."""

    async def audio_stream():
        service: KokoroTTSService = app.state.tts_service

        # Send WAV header first
        header = service._create_wav_header()
        yield header

        # Stream audio chunks
        async for chunk in generate_audio_chunks(service, text, voice, lang_code, speed):
            yield chunk

    return StreamingResponse(audio_stream(), media_type="audio/wav")


# ============= HTTP Generate and Store Endpoint =============

class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    lang_code: Optional[str] = None
    speed: float = 1.0


@app.post("/generate")
async def generate_tts_audio(request: TTSRequest):
    """
    Generate TTS audio and store it locally.
    Returns audio_url that can be used to play/download the audio.
    """
    service: KokoroTTSService = app.state.tts_service

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Ensure output directory exists
    output_dir = BASE / "output"
    output_dir.mkdir(exist_ok=True)

    # Generate unique filename
    unique_id = str(uuid.uuid4())
    lang_code = request.lang_code or service.lang_codes[0]
    filename = f"{lang_code}_{unique_id}.wav"
    output_path = output_dir / filename

    # Generate audio
    audio_chunks = []
    try:
        for audio_chunk in service.stream(
            text=request.text,
            voice=request.voice,
            speed=request.speed,
            lang_code=request.lang_code
        ):
            audio_chunks.append(audio_chunk)

        # Combine all chunks
        if audio_chunks:
            audio_combined = np.concatenate(audio_chunks)
        else:
            audio_combined = np.zeros(1, dtype=np.float32)

        # Save to file using soundfile
        sf.write(str(output_path), audio_combined, SAMPLE_RATE)

        # Return download URL
        return {
            "audio_url": f"/download/{filename}"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate audio: {str(e)}")


@app.get("/download/{filename}")
async def download_audio(filename: str):
    """
    Serve the generated audio file for playback or download.
    """
    output_dir = BASE / "output"
    file_path = output_dir / filename

    # Security check: ensure filename doesn't escape output directory
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")

    return FileResponse(
        path=str(file_path),
        media_type="audio/wav",
        filename=filename
    )


async def generate_audio_chunks(service, text, voice, lang_code, speed):
    """Helper to generate audio chunks for HTTP streaming."""
    loop = asyncio.get_event_loop()

    def generate():
        for chunk in service.stream(
            text=text,
            voice=voice,
            speed=speed,
            lang_code=lang_code
        ):
            yield chunk

    # Run generation in thread pool and yield chunks
    for chunk in generate():
        # Convert to PCM16 bytes
        pcm = service.chunk_to_pcm16(chunk)
        yield pcm
        await asyncio.sleep(0.001)


# ============= Main Entry Point =============

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("KOKORO_PORT", 8001))
    host = os.environ.get("KOKORO_HOST", "0.0.0.0")
    lang_codes = os.environ.get("KOKORO_LANG_CODES", "a,b,h,j")

    print(f"Starting Kokoro TTS Server on http://{host}:{port}")
    print(f"Language codes: {lang_codes}")
    print(f"Device: {os.environ.get('KOKORO_DEVICE', 'cuda')}")

    uvicorn.run(app, host=host, port=port)
