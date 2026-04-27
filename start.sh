#!/bin/bash

# Speech-to-Speech System Startup Script
# This script starts all services in the correct order with their conda environments

echo "================================"
echo "Starting Speech-to-Speech System"
echo "================================"
echo ""

# =============================================================================
# CONFIGURATION - Conda environment names
# =============================================================================

WHISPER_ENV="ai-agent-stt"    # Environment with whisper
KOKORO_ENV="kokoro"          # Environment with kokoro
MAIN_ENV="ai-agent-stt"      # Environment for main/gemini (same as whisper)

# =============================================================================
# Helper functions
# =============================================================================

# Function to run command in conda environment
run_in_env() {
    local env_name=$1
    local script=$2
    local log_file=$3
    local pid_file=$4
    local port=$5

    echo "Starting $script in conda env: $env_name (port: $port)..."

    # Check if port is already in use
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "  ⚠️  Port $port already in use, skipping $script"
        return 1
    fi

    # Activate conda environment and run service
    # Using conda run to execute in the environment
    conda run -n "$env_name" python "$script" > "$log_file" 2>&1 &
    local pid=$!

    # Save PID
    echo $pid > "$pid_file"

    # Wait a bit to see if it starts successfully
    sleep 3

    # Check if process is still running
    if ps -p $pid > /dev/null 2>&1; then
        echo "  ✓ $script started (PID: $pid)"
        return 0
    else
        echo "  ✗ $script failed to start. Check $log_file"
        return 1
    fi
}

# =============================================================================
# Setup
# =============================================================================

# Create necessary directories
mkdir -p logs pids

# Kill any existing services
echo "Stopping any existing services..."
pkill -f "new_whisper.py" 2>/dev/null
pkill -f "gemini_websocket.py" 2>/dev/null
pkill -f "kokoro_app.py" 2>/dev/null
pkill -f "main.py" 2>/dev/null
sleep 2
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please activate conda or use full paths."
    exit 1
fi

# =============================================================================
# Start Services
# =============================================================================

echo "Starting services..."
echo ""

# 1. Start Whisper ASR (STT) - uses ai-agent-stt env
run_in_env "$WHISPER_ENV" "new_whisper.py" "logs/whisper.log" "pids/whisper.pid" 8006
echo ""

# 2. Start Gemini LLM - uses ai-agent-stt env
run_in_env "$MAIN_ENV" "gemini_websocket.py" "logs/gemini.log" "pids/gemini.pid" 8002
echo ""

# 3. Start Kokoro TTS - uses kokoro env
run_in_env "$KOKORO_ENV" "kokoro_app.py" "logs/kokoro.log" "pids/kokoro.pid" 8001
echo ""

# 4. Start Main Orchestrator - uses ai-agent-stt env
run_in_env "$MAIN_ENV" "main.py" "logs/main.log" "pids/main.pid" 8000
echo ""

# =============================================================================
# Summary
# =============================================================================

echo "================================"
echo "Services started!"
echo "================================"
echo ""
echo "Web UI: http://localhost:8000"
echo ""
echo "Services:"
echo "  - Main Orchestrator: http://localhost:8000  (env: $MAIN_ENV)"
echo "  - Whisper STT:       http://localhost:8004  (env: $WHISPER_ENV)"
echo "  - Gemini LLM:        http://localhost:8002  (env: $MAIN_ENV)"
echo "  - Kokoro TTS:        http://localhost:8001  (env: $KOKORO_ENV)"
echo ""
echo "To stop all services, run: ./stop.sh"
echo ""
echo "To view logs:"
echo "  tail -f logs/whisper.log"
echo "  tail -f logs/gemini.log"
echo "  tail -f logs/kokoro.log"
echo "  tail -f logs/main.log"
echo ""
