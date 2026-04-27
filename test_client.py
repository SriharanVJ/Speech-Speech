"""
Test client for Speech-to-Speech system with barge-in capability.

This script demonstrates how to connect to the main WebSocket endpoint,
send audio from the microphone, and play back received TTS audio.

Usage:
    python test_client.py

Requirements:
    - PyAudio for microphone input
    - sounddevice/soundfile for audio playback
    - Connection to running main.py server
"""

import asyncio
import json
import queue
import threading
import time
from typing import Optional

import websockets
import numpy as np
import sounddevice as sd

# Configuration
SERVER_URL = "ws://localhost:8000/ws"
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = 1024
AUDIO_FORMAT = np.int16


class AudioRecorder:
    """Records audio from microphone."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS):
        self.sample_rate = sample_rate
        self.channels = channels
        self.queue: queue.Queue[bytes] = queue.Queue()
        self.stream: Optional[sd.InputStream] = None
        self.is_recording = False

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input."""
        if status:
            print(f"Audio callback status: {status}")

        # Convert to bytes (PCM16)
        audio_bytes = (indata * 32767).astype(np.int16).tobytes()
        self.queue.put(audio_bytes)

    def start(self):
        """Start recording."""
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            callback=self._audio_callback,
            blocksize=CHUNK_SIZE
        )
        self.stream.start()
        self.is_recording = True
        print("[MIC] Recording started...")

    def stop(self):
        """Stop recording."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.is_recording = False
        print("[MIC] Recording stopped")

    def get_chunk(self, timeout: float = 0.1) -> Optional[bytes]:
        """Get audio chunk."""
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None


class AudioPlayer:
    """Plays audio through speakers."""

    def __init__(self, sample_rate: int = 24000):
        self.sample_rate = sample_rate
        self.queue: queue.Queue[np.ndarray] = queue.Queue()
        self.stream: Optional[sd.OutputStream] = None
        self.is_playing = False
        self.play_thread: Optional[threading.Thread] = None

    def _audio_callback(self, outdata, frames, time_info, status):
        """Callback for audio output."""
        if status:
            print(f"Audio callback status: {status}")

        try:
            audio_chunk = self.queue.get_nowait()
            outdata[:] = audio_chunk.reshape(-1, 1)
        except queue.Empty:
            outdata[:] = np.zeros((frames, 1), dtype=np.float32)

    def start(self):
        """Start playback."""
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
            callback=self._audio_callback,
            blocksize=CHUNK_SIZE
        )
        self.stream.start()
        self.is_playing = True
        print("[SPEAKER] Playback started")

    def stop(self):
        """Stop playback."""
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.is_playing = False
        print("[SPEAKER] Playback stopped")

    def add_chunk(self, audio_bytes: bytes):
        """Add audio chunk to playback queue."""
        # Convert PCM16 bytes to float32
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        # Resample if needed (Kokoro is 24kHz)
        if self.sample_rate != SAMPLE_RATE:
            import samplerate
            resampler = samplerate.Resampler('sinc_bestest', channels=1)
            audio_float32 = resampler.process(
                audio_float32.reshape(-1, 1),
                ratio=self.sample_rate / SAMPLE_RATE
            ).flatten()

        self.queue.put(audio_float32)


async def main():
    """Main test client."""
    print("=" * 60)
    print("  Speech-to-Speech Test Client")
    print("=" * 60)
    print(f"  Connecting to: {SERVER_URL}")
    print()

    # Initialize audio
    recorder = AudioRecorder()
    player = AudioPlayer(sample_rate=24000)  # Kokoro uses 24kHz

    # Start recording
    recorder.start()
    player.start()

    # Statistics
    requests_sent = 0
    responses_received = 0
    barge_in_count = 0

    try:
        async with websockets.connect(SERVER_URL) as ws:
            print("[CONNECTED] Connected to server")

            # Receive ready message
            ready_msg = await ws.recv()
            print(f"[SERVER] {ready_msg}")

            print()
            print("Speak now... (Press Ctrl+C to stop)")
            print("-" * 40)

            # Start receive task
            async def receive_messages():
                nonlocal responses_received, barge_in_count

                while True:
                    try:
                        # Receive message (could be JSON or binary)
                        message = await ws.recv()

                        # Binary audio data
                        if isinstance(message, bytes):
                            player.add_chunk(message)

                        # JSON message
                        else:
                            msg = json.loads(message)
                            msg_type = msg.get("type")

                            if msg_type == "stt_text":
                                print(f"\n[STT] \"{msg.get('data')}\"")

                            elif msg_type == "llm_response":
                                print(f"[LLM] \"{msg.get('data')}\"")
                                responses_received += 1

                            elif msg_type == "llm_text":
                                print(f"[LLM Streaming] \"{msg.get('data')}\"")

                            elif msg_type == "tts_complete":
                                print("[TTS] Complete")

                            elif msg_type == "barge_in":
                                barge_in_count += 1
                                print(f"\n[BARGE-IN #{barge_in_count}] TTS interrupted!")

                            elif msg_type == "error":
                                print(f"[ERROR] {msg.get('message')}")

                    except websockets.exceptions.ConnectionClosed:
                        print("[DISCONNECTED] Connection closed")
                        break
                    except Exception as e:
                        print(f"[RECEIVE ERROR] {e}")

            # Start receive task
            receive_task = asyncio.create_task(receive_messages())

            # Send audio chunks
            while True:
                audio_chunk = recorder.get_chunk(timeout=0.1)

                if audio_chunk:
                    await ws.send(audio_chunk)
                    requests_sent += 1

                    # Periodic status
                    if requests_sent % 100 == 0:
                        print(f"[STATUS] Sent {requests_sent} chunks, {responses_received} responses, {barge_in_count} barge-ins")

                # Small sleep to prevent overwhelming
                await asyncio.sleep(0.001)

    except KeyboardInterrupt:
        print("\n[STOP] Stopping...")

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        recorder.stop()
        player.stop()

        print()
        print("=" * 40)
        print("  Statistics")
        print("=" * 40)
        print(f"  Audio chunks sent: {requests_sent}")
        print(f"  Responses received: {responses_received}")
        print(f"  Barge-ins detected: {barge_in_count}")
        print()


if __name__ == "__main__":
    print()
    print("Requirements:")
    print("  - sounddevice: pip install sounddevice")
    print("  - PortAudio: system audio library")
    print()

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[EXIT] Goodbye!")
