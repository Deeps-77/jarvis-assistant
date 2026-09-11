"""Standalone voice console server (FastAPI + WebSocket, :8600).

Wire-up only: mic frames -> VAD -> STT -> ephemeral agent -> streaming
TTS -> playback frames. No chat threads, no steps, no persistence —
nothing spoken survives the socket.

Run: ``python -m voice_server.server`` (or the ``Voice`` run.ps1 option).
UI: ``http://localhost:8600/``.
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .agent import VoiceAgent
from .protocol import server_msg
from .stt import transcribe_utterance, wav_bytes
from .vad import VoiceEndpoint

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"


def _port() -> int:
    try:
        return int(os.environ.get("VOICE_PORT", "") or 8600)
    except ValueError:
        return 8600


def _host() -> str:
    return os.environ.get("VOICE_HOST", "0.0.0.0")


@asynccontextmanager
async def lifespan(app):
    import core
    from speech import PiperSpeaker, piper_available

    from .stt import warmup as stt_warmup

    # NOTE: chat histories / vector memory are intentionally untouched —
    # voice turns use ephemeral keys and skip learning (see agent.py).
    try:
        core.init_speech()
    except Exception:
        logger.exception("speech init failed; voice turns will degrade to text")
    await stt_warmup()
    ok, reason = piper_available()
    if not ok:
        logger.warning("Piper unavailable (%s); replies will be text-only", reason)
    app.state.speaker = PiperSpeaker()
    logger.info("voice server ready (offline=%s)", ok)
    yield


def create_app():
    app = FastAPI(title="Jarvis Voice", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/")
    async def index():
        return FileResponse(str(STATIC_DIR / "voice.html"))

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.websocket("/ws/voice")
    async def voice_socket(ws: WebSocket):
        await ws.accept()
        endpoint = VoiceEndpoint()
        agent_holder: dict = {}

        async def send(obj: dict) -> None:
            await ws.send_json(obj)

        async def on_state(state: str) -> None:
            await send(server_msg("state", state=state))

        async def on_reply_text(text: str) -> None:
            await send(server_msg("reply", text=text[:2000]))

        async def on_audio(_index: int, wav: bytes) -> None:
            await ws.send_bytes(wav)

        def agent() -> VoiceAgent:
            ag = agent_holder.get("agent")
            if ag is None:
                ag = VoiceAgent(
                    app.state.speaker, on_state, on_reply_text, on_audio
                )
                agent_holder["agent"] = ag
            return ag

        await send(server_msg("state", state="idle"))
        last_partial = 0.0
        try:
            while True:
                packet = await ws.receive()
                if "bytes" in packet and packet["bytes"] is not None:
                    utterance = endpoint.feed(bytes(packet["bytes"]))
                    if utterance:
                        await send(server_msg("state", state="endpointing"))
                        text = await transcribe_utterance(
                            wav_bytes(utterance), "utterance.wav"
                        )
                        if text and len(text.strip()) >= 2:
                            await send(server_msg("transcript", text=text[:1000]))
                            agent().start_turn(text)
                        else:
                            await send(server_msg("state", state="listening"))
                    elif endpoint.has_pending_speech:
                        # Throttled "hearing you" heartbeat so a working-but-
                        # not-yet-endpointed mic is distinguishable from dead.
                        now = time.monotonic()
                        if now - last_partial >= 0.5:
                            last_partial = now
                            await send(
                                server_msg("partial", speechMs=endpoint.pending_ms)
                            )
                elif "text" in packet and packet["text"] is not None:
                    try:
                        msg = json.loads(packet["text"])
                    except (json.JSONDecodeError, TypeError):
                        continue
                    kind = msg.get("type")
                    if kind == "start":
                        endpoint.reset()
                        await send(server_msg("state", state="listening"))
                    elif kind == "threshold":
                        tuned = endpoint.set_threshold(msg.get("value"))
                        logger.debug("VAD threshold tuned to %.0f", tuned)
                    elif kind == "stop":
                        # Pause listening, keep the session alive: flush
                        # trailing speech (no 1s silence yet) as a normal
                        # fire-and-forget turn so its reply/audio still
                        # arrive, then report idle. The socket stays open —
                        # only a real disconnect tears down (see finally).
                        utterance = endpoint.flush()
                        endpoint.reset()
                        if utterance:
                            text = await transcribe_utterance(
                                wav_bytes(utterance), "utterance.wav"
                            )
                            if text and len(text.strip()) >= 2:
                                await send(server_msg("transcript", text=text[:1000]))
                                agent().start_turn(text)
                        await send(server_msg("state", state="idle"))
                    elif kind == "barge":
                        ag = agent_holder.get("agent")
                        if ag is not None:
                            ag.barge_in()
                        await send(server_msg("state", state="listening"))
                    elif kind == "text" and (msg.get("text") or "").strip():
                        text = msg["text"].strip()[:2000]
                        await send(server_msg("transcript", text=text))
                        agent().start_turn(text)
                    elif kind == "mute":
                        pass  # client-side only; informational
        except WebSocketDisconnect:
            pass
        except RuntimeError as e:
            # Starlette raises this when the browser goes away without a
            # clean close handshake (tab closed, refresh). Normal churn.
            if "disconnect" not in str(e).lower():
                raise
            logger.debug("voice client disconnected uncleanly; cleaned up")
        except Exception:
            logger.exception("voice socket failed")
        finally:
            ag = agent_holder.get("agent")
            if ag is not None:
                ag.cancel("disconnect")
                ag.drop_history()

    return app


app = create_app()


def main() -> None:
    import uvicorn

    print(f"[voice] serving on http://localhost:{_port()}/ (offline, ephemeral)")
    uvicorn.run(app, host=_host(), port=_port(), log_level="info")


if __name__ == "__main__":
    main()
