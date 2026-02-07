"""
server.py — Quart web server that bridges clients to Azure Voice Live.

Endpoints:
  GET  /             → serves the web frontend (index.html)
  GET  /frontend/*   → serves static JS/CSS assets
  WS   /ws           → audio WebSocket for any client (browser, iOS, etc.)

WebSocket protocol (for both browser and mobile clients):
  Client → Server:  binary frames containing raw PCM16-LE audio (24 kHz, mono)
  Server → Client:  binary frames containing raw PCM16-LE audio (24 kHz, mono)
  Server → Client:  text frames containing JSON events:
                       {"type": "speech_started"}
                       {"type": "transcript", "role": "assistant"|"user", "text": "..."}
                       {"type": "call_state",  "state": "ended"}
"""

import asyncio
import logging
import os

from dotenv import load_dotenv
from quart import Quart, send_from_directory, websocket

from voice_agent import VoiceLiveAgent

# ── Configuration ──────────────────────────────────────────────────────────

load_dotenv()

ENDPOINT = os.getenv("AZURE_VOICE_LIVE_ENDPOINT", "")
API_KEY  = os.getenv("AZURE_VOICE_LIVE_API_KEY", "")
MODEL    = os.getenv("VOICE_LIVE_MODEL", "gpt-realtime-mini")

if not ENDPOINT or not API_KEY:
    raise SystemExit(
        "ERROR: Set AZURE_VOICE_LIVE_ENDPOINT and AZURE_VOICE_LIVE_API_KEY in your .env file"
    )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Quart app ──────────────────────────────────────────────────────────────

app = Quart(__name__, static_folder="frontend")


@app.route("/")
async def index():
    """Serve the web UI."""
    return await send_from_directory("frontend", "index.html")


@app.route("/frontend/<path:filename>")
async def frontend_static(filename):
    """Serve JS, CSS, and other frontend assets."""
    return await send_from_directory("frontend", filename)


@app.websocket("/ws")
async def audio_ws():
    """
    WebSocket endpoint for streaming audio.
    Works with any client that sends/receives raw PCM16-LE binary frames.
    """
    logger.info("Client connected")
    ws = websocket._get_current_object()

    agent = VoiceLiveAgent(
        endpoint=ENDPOINT,
        api_key=API_KEY,
        model=MODEL,
        client_ws=ws,
    )

    # Start the Voice Live session in the background
    agent_task = asyncio.create_task(agent.run())

    try:
        while True:
            msg = await ws.receive()
            if isinstance(msg, bytes):
                await agent.send_audio(msg)
            # (text frames from client are ignored for now)
    except Exception:
        logger.info("Client disconnected")
    finally:
        await agent.shutdown()
        if not agent_task.done():
            agent_task.cancel()
        logger.info("Session cleaned up")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
