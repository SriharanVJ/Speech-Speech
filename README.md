# Speech-to-Speech with Barge-In

Full-duplex voice interaction system with instant interruptibility (barge-in capability).

## 🎯 Features

- **Real-time Voice Activity Detection (VAD)** - Silero VAD with fallback to energy-based
- **Barge-In Support** - TTS stops immediately when user speaks
- **Multiple STT Backends** - Whisper Large v3 Turbo, Parakeet
- **Multi-language TTS** - Kokoro with support for EN, ES, FR, HI, JA, and more
- **Streaming LLM** - Gemini 2.5 Flash for low-latency responses
- **WebSocket Protocol** - Efficient bidirectional streaming

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Browser                         │
│  (Microphone → WebSocket → Audio Playback)                  │
└───────────────────────┬─────────────────────────────────────┘
                        │ WebSocket
┌───────────────────────▼─────────────────────────────────────┐
│                   main.py (Port 8000)                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │  1. Receive mic audio (PCM16, 16kHz)               │    │
│  │  2. VAD: Detect speech in every chunk              │    │
│  │  3. Barge-in: Stop TTS if user speaks              │    │
│  │  4. Buffer audio & transcribe (Whisper/Parakeet)   │    │
│  │  5. Get LLM response (Gemini)                      │    │
│  │  6. Stream TTS audio (Kokoro)                      │    │
│  │  7. Send audio chunks to client                    │    │
│  └────────────────────────────────────────────────────┘    │
└───┬──────────┬──────────┬──────────────────────────────────┘
    │          │          │
    ▼          ▼          ▼
┌─────────┐ ┌────────┐ ┌─────────┐
│ Whisper │ │ Kokoro │ │ Gemini  │
│ASR:8004 │ │TTS:8001│ │LLM:8002 │
└─────────┘ └────────┘ └─────────┘
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install system audio libraries (Ubuntu/Debian)
sudo apt-get install portaudio19-dev python3-pyaudio
```

### 2. Start Individual Services

In separate terminals:

```bash
# Start Whisper ASR (Port 8004)
python new_whisper.py --host 0.0.0.0 --port 8004

# Start Kokoro TTS (Port 8001)
KOKORO_PORT=8001 python kokoro_app.py

# Start Gemini LLM (Port 8002)
export GEMINI_API_KEY=your_api_key
python gemini_websocket.py
```

### 3. Start Main Service

```bash
python main.py
```

The main service connects to all three services and provides the barge-in enabled WebSocket endpoint.

### 4. Test with Client

**Option 1: Web UI (Recommended)**

Open your browser and navigate to:
```
http://localhost:8000
```

The web UI provides:
- Real-time audio visualizer
- Connection status indicator
- Conversation history
- Barge-in support
- Adjustable VAD threshold

**Option 2: Python Client**

```bash
python test_client.py
```

**Option 3: Startup Script**

For convenience, use the provided startup script:

```bash
# Start all services at once
./start.sh

# Stop all services
./stop.sh
```

## 🔧 Configuration

### Environment Variables

```bash
# Service ports
WHISPER_PORT=8004
KOKORO_PORT=8001
GEMINI_PORT=8002

# Main service
PORT=8000
HOST=0.0.0.0

# Gemini API
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.5-flash

# Kokoro TTS
KOKORO_DEVICE=cuda  # or cpu
KOKORO_LANG_CODES=a,b,h,j  # American, British, Hindi, Japanese

# Whisper ASR
WHISPER_DEVICE=cuda
WHISPER_MODEL=openai/whisper-large-v3-turbo
```

### VAD Settings

In `main.py`:

```python
VAD_THRESHOLD = 0.5      # Speech detection threshold (0.0-1.0)
VAD_DEBOUNCE_MS = 200    # Minimum speech duration for barge-in
```

## 📡 WebSocket Protocol

### Client → Server

**Binary Messages:** PCM16 audio chunks (16kHz, mono)

```json
{"type": "ping"}
{"type": "reset"}
```

### Server → Client

**Binary Messages:** TTS audio chunks (24kHz, mono, float32/PCM16)

```json
{"type": "ready", "sample_rate": 16000, "channels": 1}
{"type": "stt_text", "data": "transcribed text"}
{"type": "llm_response", "data": "AI response"}
{"type": "barge_in", "message": "TTS interrupted"}
{"type": "tts_complete"}
```

## 🧪 Testing

### Test with Python Client

```bash
python test_client.py
```

### Test with WebSocket Client

Using websocat:
```bash
# Record audio and send
arecord -f S16_LE -c1 -r16000 | websocat ws://localhost:8000/ws
```

### Test with Browser

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onopen = () => console.log('Connected');
ws.onmessage = (event) => {
  if (event.data instanceof Blob) {
    // Audio chunk
    playAudioChunk(event.data);
  } else {
    // JSON message
    const msg = JSON.parse(event.data);
    console.log(msg);
  }
};

// Send audio from microphone
navigator.mediaDevices.getUserMedia({ audio: true })
  .then(stream => {
    const source = audioContext.createMediaStreamSource(stream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);

    processor.onaudioprocess = (e) => {
      const audioData = e.inputBuffer.getChannelData(0);
      const pcm16 = convertToPCM16(audioData);
      ws.send(pcm16);
    };

    source.connect(processor);
    processor.connect(audioContext.destination);
  });
```

## 🐛 Troubleshooting

### GPU Out of Memory (6GB Cards)

The system includes GPU memory optimizations for 6GB cards:

1. **Memory-efficient model loading**: Whisper loads to CPU first, converts to fp16, then moves to GPU
2. **Adjustable memory fraction**: Set `GPU_MEMORY_FRACTION` in `.env` (default: 0.85)

```env
# Use less GPU memory (more conservative)
GPU_MEMORY_FRACTION=0.75

# Or use a smaller model
WHISPER_MODEL=openai/whisper-medium
```

### Services Won't Connect

```bash
# Check if services are running
curl http://localhost:8004/health  # Whisper
curl http://localhost:8001/health  # Kokoro
curl http://localhost:8002/health  # Gemini
```

### VAD Too Sensitive

```python
# Increase threshold in main.py
VAD_THRESHOLD = 0.7  # Less sensitive

# Increase debounce time
VAD_DEBOUNCE_MS = 300  # Require longer speech
```

### TTS Not Interrupting

Check that:
1. VAD is detecting speech (check logs)
2. Audio is being sent in real-time (not buffered)
3. `is_pipeline_running` flag is being set correctly

## 📊 Performance

Typical latencies on GPU (RTX 3080):

| Component | Latency |
|-----------|---------|
| VAD | <10ms |
| Whisper STT | 200-500ms |
| Gemini LLM | 100-300ms |
| Kokoro TTS | 50-150ms |
| **Total E2E** | **500-1000ms** |

## 🔄 Barge-In Flow

1. **User speaks** while TTS is playing
2. **VAD detects speech** in incoming audio chunk
3. **Barge-in triggered:**
   - `kokoro_client.stop()` - Stop TTS generation
   - `pipeline_task.cancel()` - Cancel LLM task
   - Send `barge_in` message to client
4. **New pipeline starts:**
   - Buffer audio for transcription
   - Whisper transcribes
   - Gemini generates response
   - Kokoro streams TTS

## 📝 License

MIT License - Feel free to use in your projects!

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📚 References

- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Whisper](https://github.com/openai/whisper)
- [Kokoro TTS](https://github.com/remsky/Kokoro-FastAPI)
- [Gemini API](https://ai.google.dev/gemini-api/docs)
