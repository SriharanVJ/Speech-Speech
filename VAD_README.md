# ML-Based VAD and Turn Detection

This directory contains the ML-based Voice Activity Detection (VAD) and Turn Detection system using:

- **Silero VAD v4** (ONNX) for accurate speech activity detection
- **Smart Turn v3** (pipecat-ai) for semantic turn completion prediction

## Features

### Silero VAD v4
- **Free and Open Source** (MIT License)
- Real-time speech detection with high accuracy
- ONNX model for fast inference (~5ms per 512-sample chunk)
- No API keys required, runs entirely locally

### Smart Turn v3
- **Free and Open Source** (Apache 2.0 License)
- Semantic turn completion prediction
- Based on Whisper Tiny encoder + linear classifier
- Trained on conversational speech patterns
- Replaces heuristic linguistic analysis with learned patterns

## Installation

### 1. Install Dependencies

```bash
pip install -r vad_requirements.txt
```

### 2. Download Models

```bash
python download_models.py
```

This will download:
- `vad/models/silero_vad_v4.onnx` (~1.8 MB)
- `vad/models/smart-turn-v3.2-cpu.onnx` (~8.7 MB)

### 3. Set Environment Variables (Optional)

```bash
# Use GPU acceleration (requires onnxruntime-gpu)
export SMART_TURN_DEVICE=gpu

# Default is CPU
export SMART_TURN_DEVICE=cpu
```

## Usage

### Basic Usage

```python
from vad import TurnDetector, TurnState, create_turn_detector

# Create turn detector
detector = create_turn_detector(threshold=0.70)

# Process audio chunks
import numpy as np

# Get 16kHz audio as int16 numpy array
audio_chunk = np.array([...], dtype=np.int16)

# Process chunk (async)
event = await detector.process_chunk(audio_chunk)

if event:
    if event.state == TurnState.SPEAKING:
        print("User started speaking")
    elif event.state == TurnState.SILENCED:
        print("User paused (incomplete)")
    elif event.state == TurnState.TURN_END:
        print(f"Turn complete! Probability: {event.smart_turn_probability:.3f}")
        print(f"Speech duration: {event.speech_duration_ms:.0f}ms")
        print(f"Silence duration: {event.silence_duration_ms:.0f}ms")
```

### Integration with Whisper

```python
import asyncio
from vad import TurnDetector, TurnState, create_turn_detector

# In your WebSocket handler
turn_detector = create_turn_detector(threshold=0.70)

async def process_audio_stream(audio_stream):
    """Process audio stream with turn detection."""
    async for audio_chunk in audio_stream:
        # Process chunk through turn detector
        event = await turn_detector.process_chunk(audio_chunk)

        if event:
            if event.state == TurnState.TURN_END:
                # Send transcript to LLM
                await process_transcription()
                turn_detector.reset()
```

## Configuration

### Turn Detector Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vad_silence_trigger_ms` | 280 | Silence duration before turn check |
| `smart_turn_threshold` | 0.70 | Probability threshold for turn end |
| `max_silence_timeout_ms` | 1500 | Maximum silence before forcing turn end |
| `min_speech_duration_ms` | 250 | Minimum speech to consider valid |
| `sample_rate` | 16000 | Audio sample rate |
| `chunk_size` | 512 | VAD processing chunk size |

### WebSocket Query Parameters

When connecting to the WebSocket server:

```
ws://localhost:8006/stream/asr?use_vad=true&turn_threshold=0.70&vad_threshold=0.5
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_vad` | true | Enable VAD |
| `vad_threshold` | 0.5 | VAD speech threshold |
| `turn_threshold` | 0.70 | Turn completion threshold |
| `turn_silence_ms` | 280 | Silence before turn check |
| `max_turn_duration_ms` | 10000 | Force turn end after this duration |

## Architecture

```
Audio Input (16kHz PCM)
    ↓
[Silero VAD v4] → Speech detection (probability)
    ↓
[Speech Buffer] → Collect audio segment
    ↓
[Smart Turn v3] → Turn completion prediction
    ↓
[Decision] → turn_end if prob >= threshold
```

## Events

### Turn States

| State | Description |
|-------|-------------|
| `LISTENING` | Waiting for speech |
| `SPEAKING` | User is speaking |
| `SILENCED` | User paused (may continue) |
| `TURN_END` | Turn complete |

### WebSocket Events

| Event | Description |
|-------|-------------|
| `speech_start` | Speech detected (barge-in trigger) |
| `partial` | Partial transcript |
| `turn_end` | Turn complete with probability |
| `segment_completed` | Final transcript |

## Troubleshooting

### Models Not Found

If you see "Smart-Turn v3 model not found":

```bash
python download_models.py
```

### ONNX Runtime Error

```bash
pip install --upgrade onnxruntime
# Or for GPU
pip install --upgrade onnxruntime-gpu
```

### Slow Inference

- Use CPU model for fastest startup
- GPU model helps with concurrent sessions
- Reduce `smart_turn_num_threads` if CPU constrained

## Performance

### CPU Model
- Silero VAD: ~5ms per 512-sample chunk
- Smart Turn: ~50ms per 8-second audio segment
- Memory: ~50MB per session

### GPU Model
- Silero VAD: ~2ms per chunk
- Smart Turn: ~20ms per segment
- Memory: ~200MB (GPU + CPU)

## License

- Silero VAD: MIT License
- Smart Turn v3: Apache 2.0 License
- Both models are free for commercial use

## References

- [Silero VAD](https://github.com/snakers4/silero-vad)
- [Smart Turn v3](https://github.com/pipecat-ai/smart-turn)
- [HuggingFace - Smart Turn](https://huggingface.co/pipecat-ai/smart-turn-v3)

## Support

For issues or questions:
1. Check the main README.md
2. Review the code documentation
3. Check model availability on HuggingFace
