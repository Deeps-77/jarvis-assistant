"""Ephemeral voice brain: one turn = transcribe -> respond -> speak.

Single active turn at a time. Barge-in cancels the in-flight turn (LLM
stream stops, synth queue drops) so the new utterance takes over
immediately. History lives in ``core.chat_histories`` under a
``voice:{uuid}`` key for the session only — never resumed, never learned
(``ephemeral=True``), dropped on disconnect.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

logger = logging.getLogger(__name__)


class VoiceAgent:
    """One browser session's conversation partner."""

    def __init__(self, speaker, on_state, on_reply_text, on_audio, owner: str = "-") -> None:
        from .tts import SentenceSynth

        self._speaker = speaker
        self._on_state = on_state
        self._on_reply_text = on_reply_text
        self._on_audio = on_audio
        self._owner = owner
        self._chat_key = f"voice:{uuid.uuid4().hex[:12]}"
        self._current: asyncio.Task | None = None
        self._synth: SentenceSynth | None = None
        self._SentenceSynth = SentenceSynth

    @property
    def busy(self) -> bool:
        return self._current is not None and not self._current.done()

    def start_turn(self, transcript: str) -> None:
        """Begin answering; cancels any in-flight turn first (barge-in safe)."""
        self.cancel("new turn")
        self._current = asyncio.create_task(self._turn(transcript))

    def barge_in(self) -> None:
        """User cut in: stop speaking + thinking immediately."""
        self.cancel("barge-in")

    def cancel(self, reason: str) -> None:
        if self._synth is not None:
            self._synth.cancel()
            self._synth = None
        if self._current is not None and not self._current.done():
            logger.debug("cancelling voice turn (%s)", reason)
            self._current.cancel()
        self._current = None

    async def _turn(self, transcript: str) -> None:
        import core

        await self._on_state("thinking")
        synth = self._SentenceSynth(self._speaker, self._on_audio)
        self._synth = synth
        streamed: list[str] = []

        async def on_token(token: str):
            streamed.append(token)
            await synth.feed_token(token)

        try:
            body, _sources, _failed = await core.respond(
                self._chat_key, transcript, owner=self._owner,
                on_token=on_token, ephemeral=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("voice turn failed")
            body = "Sorry, something went wrong on my end."
            await synth.feed_token(body)
        finally:
            await synth.drain()
        full = "".join(streamed).strip() or body
        await self._on_reply_text(full)
        await self._on_state("listening")
        self._synth = None

    def drop_history(self) -> None:
        """Forget the session (disconnect). RAM only, nothing persisted."""
        try:
            import core

            core.chat_histories.pop(self._chat_key, None)
        except Exception:
            pass


__all__ = ["VoiceAgent"]
