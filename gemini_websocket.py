"""
Gemini AI WebSocket Server

Separate WebSocket service that handles Gemini API calls.
Receives text queries and returns AI responses.

Usage:
    python gemini_websocket.py

WebSocket Client Connection:
    ws://localhost:8002/ws

Environment Variables:
    GEMINI_API_KEY or GOOGLE_API_KEY - Your Gemini API key
    GEMINI_PORT - Port to run on (default: 8002)
"""

import asyncio
import json
import logging
import os
import sys
from typing import Optional
from datetime import datetime

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Google Gemini
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("[WARNING] google-genai not installed. Install with: pip install google-genai")

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env is optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============= Configuration =============

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_PORT = int(os.environ.get("GEMINI_PORT", "8002"))
GEMINI_HOST = os.environ.get("GEMINI_HOST", "0.0.0.0")

# System prompt for the AI assistant
SYSTEM_PROMPT = """You are a helpful, friendly AI assistant. Respond in a conversational manner.
Keep your responses concise and to the point, typically 1-3 sentences unless more detail is requested."""

# ============= FastAPI App =============

app = FastAPI(title="Gemini AI WebSocket Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GeminiWebSocketHandler:
    """Handler for Gemini WebSocket connections."""

    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.client = None

        if not self.api_key:
            logger.warning("[WARNING] No GEMINI_API_KEY or GOOGLE_API_KEY found in environment")
        elif GEMINI_AVAILABLE:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"[GEMINI] Client initialized with model: {GEMINI_MODEL}")
            except Exception as e:
                logger.error(f"[ERROR] Failed to initialize Gemini client: {e}")

    async def get_gemini_response(self, text: str) -> Optional[str]:
        """Send text to Gemini and get response."""
        if not GEMINI_AVAILABLE:
            logger.warning("[GEMINI] google-genai not available")
            return None

        if not self.api_key:
            logger.warning("[GEMINI] No API key configured")
            return None

        try:
            if self.client is None:
                self.client = genai.Client(api_key=self.api_key)

            # Generate content with system prompt
            response = self.client.models.generate_content(
                model=GEMINI_MODEL,
                contents=f"{SYSTEM_PROMPT}\n\nUser: {text}"
            )
            return response.text

        except Exception as e:
            error_msg = str(e)
            logger.error(f"[GEMINI ERROR] {error_msg}")

            # Return a user-friendly error message that can be spoken
            if "503" in error_msg or "UNAVAILABLE" in error_msg or "high demand" in error_msg:
                return "Sorry, the AI service is currently experiencing high demand. Please try again in a moment."
            elif "401" in error_msg or "403" in error_msg or "API key" in error_msg:
                return "Sorry, there's an issue with the API configuration. Please check your API key."
            elif "timeout" in error_msg.lower():
                return "Sorry, the service timed out. Please try again."
            else:
                return f"Sorry, an error occurred: {error_msg}"

    async def handle_connection(self, ws: WebSocket):
        """Handle WebSocket connection."""
        await ws.accept()

        # Send ready message
        await ws.send_json({
            "type": "ready",
            "model": GEMINI_MODEL,
            "timestamp": datetime.now().isoformat()
        })
        logger.info("[CONNECTED] Client connected")

        request_count = 0

        try:
            while True:
                # Receive message from client
                data = await ws.receive_json()
                msg_type = data.get("type")

                if msg_type == "query":
                    request_count += 1
                    query_text = data.get("text", "")

                    if not query_text:
                        await ws.send_json({
                            "type": "error",
                            "message": "Empty query text"
                        })
                        continue

                    logger.info(f"[QUERY #{request_count}] \"{query_text[:100]}...\"")

                    # Send status message
                    await ws.send_json({
                        "type": "status",
                        "message": "Processing your query...",
                        "request_id": request_count
                    })

                    # Get Gemini response
                    response_text = await self.get_gemini_response(query_text)

                    if response_text:
                        logger.info(f"[RESPONSE #{request_count}] \"{response_text[:100]}...\"")
                        await ws.send_json({
                            "type": "response",
                            "text": response_text,
                            "request_id": request_count
                        })
                    else:
                        await ws.send_json({
                            "type": "error",
                            "message": "Failed to get response from Gemini",
                            "request_id": request_count
                        })

                elif msg_type == "ping":
                    # Heartbeat/ping message
                    await ws.send_json({"type": "pong"})

                elif msg_type == "close":
                    logger.info("[INFO] Client requested close")
                    break

        except WebSocketDisconnect:
            logger.info("[DISCONNECTED] Client disconnected")
        except Exception as e:
            logger.error(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                await ws.close()
            except:
                pass


# Global handler instance
gemini_handler = GeminiWebSocketHandler()


@app.websocket("/ws")
async def websocket_gemini(ws: WebSocket):
    """WebSocket endpoint for Gemini AI responses."""
    await gemini_handler.handle_connection(ws)


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "Gemini AI WebSocket Server",
        "model": GEMINI_MODEL,
        "available": GEMINI_AVAILABLE,
        "api_key_configured": bool(gemini_handler.api_key),
        "websocket_endpoint": f"ws://{GEMINI_HOST}:{GEMINI_PORT}/ws"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if GEMINI_AVAILABLE else "unavailable",
        "model": GEMINI_MODEL,
        "client_initialized": gemini_handler.client is not None
    }


# ============= Main Entry Point =============

if __name__ == "__main__":
    # Check configuration
    print("=" * 60)
    print("Gemini AI WebSocket Server")
    print("=" * 60)
    print(f"Model: {GEMINI_MODEL}")
    print(f"Host: {GEMINI_HOST}")
    print(f"Port: {GEMINI_PORT}")
    print(f"Gemini Available: {GEMINI_AVAILABLE}")
    print(f"API Key Configured: {bool(gemini_handler.api_key)}")

    if not GEMINI_AVAILABLE:
        print()
        print("[WARNING] google-genai not installed!")
        print("Install with: pip install google-genai")

    if not gemini_handler.api_key:
        print()
        print("[WARNING] No API key found!")
        print("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")

    print("=" * 60)
    print()

    # Start server
    uvicorn.run(app, host=GEMINI_HOST, port=GEMINI_PORT)
