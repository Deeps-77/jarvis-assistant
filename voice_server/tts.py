"""Streaming TTS: sentence splitter + Piper prefetch pipeline.

Perceived latency trick from the LiveKit/Pipecat playbook: never wait for
the full reply. Split the token stream into sentences, synthesize N+1
while the client plays N. Piper renders a sentence in ~200-400ms on CPU.
"""

from __future__ import annotations

import asyncio
import logging
import re

logger = logging.getLogger(__name__)

# Complete-sentence spans: punctuation run + following whitespace.
_SENTENCE_END = re.compile(r"[.!?]+\s+")


def split_sentences(text: str) -> list[str]:
    """Split reply text into speakable chunks (keeps punctuation, drops empties)."""
    out: list[str] = []
    last_end = 0
    for match in _SENTENCE_END.finditer(text):
        piece = text[last_end : match.end()].strip()
        if piece:
            out.append(piece)
        last_end = match.end()
    tail = text[last_end:].strip()
    if tail:
        out.append(tail)
    return out


class SentenceSynth:
    """Ordered prefetching synthesizer over a shared PiperSpeaker.

    ``feed_token`` buffers the LLM stream; each completed sentence is
    synthesized in a background task immediately, in order. ``drain()``
    flushes the tail. ``cancel()`` (barge-in) drops everything pending.
    Callbacks receive ``(index, wav_bytes)`` in sentence order.
    """

    def __init__(self, speaker, on_sentence) -> None:
        self._speaker = speaker
        self._on_sentence = on_sentence
        self._buf = ""
        self._next_index = 0
        self._pending: dict[int, asyncio.Task] = {}
        self._emit_index = 0
        self._cancelled = False

    def _clean(self, text: str) -> str:
        from speech import _clean_for_speech

        return _clean_for_speech(text)

    async def feed_token(self, token: str) -> None:
        if self._cancelled:
            return
        self._buf += token
        # Launch every complete sentence, keeping the unparsed remainder
        # (with its whitespace) so the next token can't glue words together.
        last_end = 0
        for match in _SENTENCE_END.finditer(self._buf):
            sentence = self._buf[last_end : match.end()].strip()
            last_end = match.end()
            if sentence:
                await self._launch(sentence)
        self._buf = self._buf[last_end:]

    async def _launch(self, sentence: str) -> None:
        index = self._next_index
        self._next_index += 1
        clean = self._clean(sentence)
        if not clean:
            # Empty after cleaning: occupy the index so ordering never stalls.
            async def _empty():
                return b""

            self._pending[index] = asyncio.create_task(_empty())
        else:
            self._pending[index] = asyncio.create_task(self._speaker.synthesize(clean))
        await self._drain_ready()

    async def _drain_ready(self) -> None:
        while self._emit_index in self._pending:
            task = self._pending[self._emit_index]
            if not task.done():
                break
            self._pending.pop(self._emit_index)
            try:
                wav = task.result()
            except (asyncio.CancelledError, Exception):
                wav = b""
            if wav and not self._cancelled:
                await self._on_sentence(self._emit_index, bytes(wav))
            self._emit_index += 1

    async def drain(self) -> None:
        """Flush the final partial sentence, then all pending audio."""
        if self._cancelled:
            return
        tail = self._buf.strip()
        self._buf = ""
        if tail:
            await self._launch(tail)
        while self._emit_index in self._pending:
            task = self._pending[self._emit_index]
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=60)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pass
            await self._drain_ready()

    def cancel(self) -> None:
        """Barge-in: stop everything pending and future."""
        self._cancelled = True
        for task in self._pending.values():
            if task is not None and not task.done():
                task.cancel()
        self._pending.clear()
        self._buf = ""
