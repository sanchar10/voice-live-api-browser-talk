"""
voice_agent.py — Manages a single Voice Live session.

Responsibilities:
  • Connect to Azure Voice Live API (API-key auth via the SDK)
  • Stream audio in both directions between a client WebSocket and the service
  • Execute tool calls when the model requests them
  • Forward agent audio + transcript events back to the client

Audio contract (designed to work with both browser and future iOS client):
  ┌──────────┐   binary PCM16-LE   ┌────────────┐   SDK (base64)   ┌───────────────┐
  │  Client   │ ──────────────────▶ │  server.py  │ ──────────────▶ │  Voice Live   │
  │ (browser  │ ◀────────────────── │ voice_agent │ ◀────────────── │  API          │
  │  or iOS)  │   binary PCM16-LE   └────────────┘                  └───────────────┘
  └──────────┘

  • Client → Server: raw PCM16 little-endian, 24 kHz, mono (binary WS frames)
  • Server → Client: raw PCM16 little-endian, 24 kHz, mono (binary WS frames)
  • Server → Client: JSON text frames for events (transcript, speech_started, call_state)
"""

import asyncio
import base64
import json
import logging

from azure.core.credentials import AzureKeyCredential
from azure.ai.voicelive.aio import connect
from azure.ai.voicelive.models import (
    AzureStandardVoice,
    AudioInputTranscriptionOptions,
    FunctionCallOutputItem,
    InputAudioFormat,
    ItemType,
    Modality,
    OutputAudioFormat,
    RequestSession,
    ServerEventType,
    ServerVad,
    ToolChoiceLiteral,
    ServerEventConversationItemCreated,
    ServerEventResponseFunctionCallArgumentsDone,
)

from tools import TOOL_DEFINITIONS, TOOL_HANDLERS

logger = logging.getLogger(__name__)

# The system prompt — intentionally simple
INSTRUCTIONS = "You are a helpful assistant. Be polite and do what the user asks."


class VoiceLiveAgent:
    """Bridges one client WebSocket to one Azure Voice Live session."""

    def __init__(self, endpoint: str, api_key: str, model: str, client_ws):
        """
        Args:
            endpoint:  Azure AI Services endpoint URL
            api_key:   API key for authentication
            model:     Deployment name, e.g. "gpt-realtime-mini"
            client_ws: The Quart/WebSocket object for the connected client
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.model = model
        self.client_ws = client_ws

        # Internal queue: client audio bytes waiting to be sent to Voice Live
        self._audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self):
        """Open the Voice Live session and process events until shutdown."""
        credential = AzureKeyCredential(self.api_key)
        self._running = True

        try:
            async with connect(
                endpoint=self.endpoint,
                credential=credential,
                model=self.model,
            ) as conn:
                # Configure the session
                await self._setup_session(conn)

                # Run the event loop and audio sender concurrently
                self._tasks = [
                    asyncio.create_task(self._event_loop(conn)),
                    asyncio.create_task(self._audio_sender(conn)),
                ]
                # Wait until either task finishes (usually means disconnect)
                done, pending = await asyncio.wait(
                    self._tasks, return_when=asyncio.FIRST_COMPLETED
                )
                for t in pending:
                    t.cancel()
                # Re-raise any exception from the completed task
                for t in done:
                    if t.exception():
                        logger.error("Task error: %s", t.exception())

        except Exception:
            logger.exception("Voice Live session error")
        finally:
            self._running = False
            await self._send_json({"type": "call_state", "state": "ended"})

    async def send_audio(self, pcm_bytes: bytes):
        """Queue raw PCM16 audio bytes from the client for sending to Voice Live."""
        if self._running:
            await self._audio_queue.put(pcm_bytes)

    async def shutdown(self):
        """Cancel background tasks."""
        self._running = False
        for t in self._tasks:
            t.cancel()

    # ------------------------------------------------------------------
    # Session setup
    # ------------------------------------------------------------------

    async def _setup_session(self, conn):
        """Send the session configuration to Voice Live."""
        session = RequestSession(
            modalities=[Modality.TEXT, Modality.AUDIO],
            instructions=INSTRUCTIONS,
            voice=AzureStandardVoice(name="en-US-Emma2:DragonHDLatestNeural"),
            input_audio_format=InputAudioFormat.PCM16,
            output_audio_format=OutputAudioFormat.PCM16,
            turn_detection=ServerVad(
                threshold=0.5,
                prefix_padding_ms=300,
                silence_duration_ms=500,
            ),
            input_audio_transcription=AudioInputTranscriptionOptions(model="whisper-1"),
            tools=TOOL_DEFINITIONS,
            tool_choice=ToolChoiceLiteral.AUTO,
        )
        await conn.session.update(session=session)
        logger.info("Session config sent (model=%s)", self.model)

    # ------------------------------------------------------------------
    # Event loop — reads every event from Voice Live
    # ------------------------------------------------------------------

    async def _event_loop(self, conn):
        """Iterate over Voice Live events and dispatch them."""
        try:
            async for event in conn:
                etype = event.type

                if etype == ServerEventType.SESSION_UPDATED:
                    logger.info("Session ready (id=%s)", event.session.id)

                elif etype == ServerEventType.INPUT_AUDIO_BUFFER_SPEECH_STARTED:
                    # Tell the client to stop playing agent audio (barge-in)
                    await self._send_json({"type": "speech_started"})

                elif etype == ServerEventType.RESPONSE_AUDIO_DELTA:
                    # Forward audio bytes to the client as a binary frame
                    await self.client_ws.send(event.delta)

                elif etype == ServerEventType.RESPONSE_AUDIO_TRANSCRIPT_DONE:
                    # Send the full agent transcript as text
                    await self._send_json({
                        "type": "transcript",
                        "role": "assistant",
                        "text": event.transcript,
                    })

                elif etype == ServerEventType.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED:
                    # Send user transcript back so the client can display it
                    await self._send_json({
                        "type": "transcript",
                        "role": "user",
                        "text": event.transcript,
                    })

                elif etype == ServerEventType.CONVERSATION_ITEM_CREATED:
                    await self._handle_conversation_item(event, conn)

                elif etype == ServerEventType.ERROR:
                    logger.error("Voice Live error: %s", event.error.message)

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Event loop error")

    # ------------------------------------------------------------------
    # Tool / function-call handling
    # ------------------------------------------------------------------

    async def _handle_conversation_item(self, event, conn):
        """If the model created a function-call item, execute the tool."""
        if not isinstance(event, ServerEventConversationItemCreated):
            return
        if event.item.type != ItemType.FUNCTION_CALL:
            return

        function_name = event.item.name
        call_id = event.item.call_id
        previous_item_id = event.item.id
        logger.info("Tool call: %s (call_id=%s)", function_name, call_id)

        try:
            # Wait for the full arguments to arrive
            args_event = await self._wait_for_event(
                conn, ServerEventType.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE
            )
            if not isinstance(args_event, ServerEventResponseFunctionCallArgumentsDone):
                return

            # Parse arguments
            raw_args = args_event.arguments
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})

            # Wait for the response to complete before sending tool output
            await self._wait_for_event(conn, ServerEventType.RESPONSE_DONE)

            # Dispatch to the handler
            handler = TOOL_HANDLERS.get(function_name)
            if handler:
                result_json = handler(arguments)
                logger.info("Tool %s result: %s", function_name, result_json)
            else:
                result_json = json.dumps({"error": f"Unknown tool: {function_name}"})
                logger.warning("No handler for tool: %s", function_name)

            # Send the result back to Voice Live
            output = FunctionCallOutputItem(call_id=call_id, output=result_json)
            await conn.conversation.item.create(
                previous_item_id=previous_item_id, item=output
            )
            # Ask the model to generate a new response incorporating the tool result
            await conn.response.create()

        except asyncio.TimeoutError:
            logger.error("Timeout waiting for tool call completion: %s", function_name)
        except Exception:
            logger.exception("Error handling tool call: %s", function_name)

    # ------------------------------------------------------------------
    # Audio sender — drains the queue and pushes to Voice Live
    # ------------------------------------------------------------------

    async def _audio_sender(self, conn):
        """Read PCM bytes from the queue and send them to Voice Live."""
        try:
            while self._running:
                pcm_bytes = await asyncio.wait_for(
                    self._audio_queue.get(), timeout=1.0
                )
                audio_b64 = base64.b64encode(pcm_bytes).decode("ascii")
                await conn.input_audio_buffer.append(audio=audio_b64)
        except asyncio.TimeoutError:
            pass  # just loop again
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Audio sender error")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _send_json(self, data: dict):
        """Send a JSON text frame to the client WebSocket."""
        try:
            await self.client_ws.send(json.dumps(data))
        except Exception:
            pass  # client may have already disconnected

    @staticmethod
    async def _wait_for_event(conn, event_type, timeout_s: float = 10.0):
        """Consume events from the connection until we see the expected type."""
        async def _next():
            while True:
                evt = await conn.recv()
                if evt.type == event_type:
                    return evt
        return await asyncio.wait_for(_next(), timeout=timeout_s)
