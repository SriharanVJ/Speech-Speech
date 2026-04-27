#!/bin/bash

# Speech-to-Speech System Stop Script
# This script stops all services

echo "Stopping Speech-to-Speech System..."
echo ""

# Kill services by name
echo "Killing services..."
pkill -f "new_whisper.py" 2>/dev/null && echo "  ✓ Whisper stopped" || echo "  - Whisper not running"
pkill -f "gemini_websocket.py" 2>/dev/null && echo "  ✓ Gemini stopped" || echo "  - Gemini not running"
pkill -f "kokoro_app.py" 2>/dev/null && echo "  ✓ Kokoro stopped" || echo "  - Kokoro not running"
pkill -f "main.py" 2>/dev/null && echo "  ✓ Main stopped" || echo "  - Main not running"

echo ""
echo "All services stopped."
