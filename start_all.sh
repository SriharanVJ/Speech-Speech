#!/bin/bash

# Startup script for Speech-to-Speech system
# Starts all required services in separate tmux sessions

set -e

echo "=========================================="
echo "  Speech-to-Speech Startup Script"
echo "=========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo -e "${RED}Error: tmux not installed${NC}"
    echo "Install with: sudo apt-get install tmux"
    exit 1
fi

# Session name
SESSION="s2s-system"

# Kill existing session
if tmux has-session -t $SESSION 2>/dev/null; then
    echo -e "${YELLOW}Killing existing session...${NC}"
    tmux kill-session -t $SESSION
fi

# Create new session
echo -e "${GREEN}Creating tmux session: $SESSION${NC}"
tmux new-session -d -s $SESSION -n "main"

# Split into panes - create 2x2 grid
tmux split-window -t $SESSION:0 -v -p 50
tmux select-pane -t $SESSION:0.0
tmux split-window -t $SESSION:0.0 -h -p 50
tmux select-pane -t $SESSION:0.2
tmux split-window -t $SESSION:0.2 -h -p 50

# Send commands to each pane

# Pane 0: Whisper ASR (top-left)
echo -e "${GREEN}Starting Whisper ASR (Port 8004)...${NC}"
tmux send-keys -t $SESSION:0.0 "cd $(pwd)" C-m
tmux send-keys -t $SESSION:0.0 "python new_whisper.py --host 0.0.0.0 --port 8004" C-m

# Pane 1: Kokoro TTS (top-right)
echo -e "${GREEN}Starting Kokoro TTS (Port 8001)...${NC}"
tmux send-keys -t $SESSION:0.1 "cd $(pwd)" C-m
tmux send-keys -t $SESSION:0.1 "KOKORO_PORT=8001 python kokoro_app.py" C-m

# Pane 2: Gemini LLM (bottom-left)
echo -e "${GREEN}Starting Gemini LLM (Port 8002)...${NC}"
tmux send-keys -t $SESSION:0.2 "cd $(pwd)" C-m
tmux send-keys -t $SESSION:0.2 "python gemini_websocket.py" C-m

# Pane 3: Main service (bottom-right)
echo -e "${GREEN}Starting Main Service (Port 8000)...${NC}"
tmux send-keys -t $SESSION:0.3 "cd $(pwd)" C-m
sleep 3  # Wait for other services to start
tmux send-keys -t $SESSION:0.3 "python main.py" C-m

echo ""
echo -e "${GREEN}All services started!${NC}"
echo ""
echo "To attach to the session:"
echo "  tmux attach -t $SESSION"
echo ""
echo "To switch between panes:"
echo "  Ctrl+B then Arrow Keys"
echo ""
echo "To detach:"
echo "  Ctrl+B then D"
echo ""
echo "To kill the session:"
echo "  tmux kill-session -t $SESSION"
echo ""

# Show status
sleep 5
echo -e "${YELLOW}Service Status:${NC}"
curl -s http://localhost:8004/health && echo -e "  ${GREEN}✓${NC} Whisper ASR (8004)" || echo -e "  ${RED}✗${NC} Whisper ASR (8004)"
curl -s http://localhost:8001/health && echo -e "  ${GREEN}✓${NC} Kokoro TTS (8001)" || echo -e "  ${RED}✗${NC} Kokoro TTS (8001)"
curl -s http://localhost:8002/health && echo -e "  ${GREEN}✓${NC} Gemini LLM (8002)" || echo -e "  ${RED}✗${NC} Gemini LLM (8002)"
curl -s http://localhost:8000/health && echo -e "  ${GREEN}✓${NC} Main Service (8000)" || echo -e "  ${RED}✗${NC} Main Service (8000)"
echo ""

# Attach to session
tmux attach -t $SESSION
