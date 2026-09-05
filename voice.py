"""Always-on voice loop for the web UI (app.py).

Pipeline per spoken turn: mic PCM → :class:`VoiceTurnTaker` (energy VAD
endpointing) → faster-whisper → ``core.respond`` (ephemeral session) →
Piper TTS → auto-playing ``cl.Audio``.

Nothing spoken is persisted: transcripts live only for the active turn,
threads are deleted on chat end, and vector memory is skipped (see
``core.respond(..., ephemeral=True)``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


def _rms_int16(frame: bytes) -> float:
    """Root-mean-square energy of mono int16 PCM. numpy fast path, else stdlib."""
    if not frame:
        return 0.0
    try:
        import numpy as np

        samples = np.frombuffer(frame, dtype=np.int16).astype(float)
        if samples.size == 0:
            return 0.0
        return float((samples * samples).mean() ** 0.5)
    except ImportError:
        import struct

        n = len(frame) // 2
        if n == 0:
            return 0.0
        total = 0
        for (sample,) in struct.iter_unpack("<h", frame[: n * 2]):
            total += sample * sample
        return (total / n) ** 0.5


@dataclass(slots=True)
class VoiceConfig:
    """Tuning knobs (all overridable via ``VOICE_*`` env vars)."""

    sample_rate: int = 24000
    frame_ms: int = 30
    # RMS energy above this counts as speech (int16 scale: 0..32768).
    threshold: float = 500.0
    # Trailing silence that ends a turn.
    silence_ms: int = 1000
    # Longest single turn before a forced cut.
    max_turn_ms: int = 30000
    # Shorter than this after endpointing counts as a mic bump, not speech.
    min_speech_ms: int = 400

    @classmethod
    def from_env(cls) -> "VoiceConfig":
        def _int(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, "") or default)
            except ValueError:
                return default

        def _float(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, "") or default)
            except ValueError:
                return default

        return cls(
            sample_rate=_int("VOICE_SAMPLE_RATE", 24000),
            frame_ms=_int("VOICE_FRAME_MS", 30),
            threshold=_float("VOICE_VAD_THRESHOLD", 500.0),
            silence_ms=_int("VOICE_SILENCE_MS", 1000),
            max_turn_ms=_int("VOICE_MAX_TURN_MS", 30000),
            min_speech_ms=_int("VOICE_MIN_SPEECH_MS", 400),
        )


class VoiceTurnTaker:
    """Energy-VAD endpointing over a raw int16 mono PCM stream.

    Usage: call :meth:`feed` with each mic chunk; it returns the complete
    utterance bytes the moment trailing silence (or the length cap) ends
    a turn, else ``None``. Call :meth:`flush` on mic release to collect a
    partial utterance, and :meth:`reset` between turns.
    """

    def __init__(self, config: VoiceConfig | None = None) -> None:
        self.config = config or VoiceConfig.from_env()
        self._buf = bytearray()
        self._speech_ms = 0
        self._silence_ms = 0
        self._in_speech = False

    @property
    def frame_bytes(self) -> int:
        return int(self.config.sample_rate * self.config.frame_ms / 1000) * 2

    @property
    def has_pending_speech(self) -> bool:
        return self._in_speech or self._speech_ms > 0

    def reset(self) -> None:
        del self._buf[:]
        self._speech_ms = 0
        self._silence_ms = 0
        self._in_speech = False

    def _frame_energies(self, pcm: bytes) -> list[float]:
        size = self.frame_bytes
        return [_rms_int16(pcm[i : i + size]) for i in range(0, len(pcm), size) if i + size <= len(pcm)]

    def feed(self, pcm: bytes) -> bytes | None:
        """Consume mic audio. Returns a finished utterance, else None."""
        cfg = self.config
        for energy in self._frame_energies(pcm):
            if energy >= cfg.threshold:
                self._in_speech = True
                self._speech_ms += cfg.frame_ms
                self._silence_ms = 0
            elif self._in_speech:
                self._silence_ms += cfg.frame_ms
        if self._in_speech:
            # Buffer at whole-chunk granularity; a little leading/trailing
            # padding is harmless for transcription.
            self._buf.extend(pcm)
            total_ms = self._speech_ms + self._silence_ms
            if self._silence_ms >= cfg.silence_ms or total_ms >= cfg.max_turn_ms:
                return self._finish()
        return None

    def _finish(self) -> bytes | None:
        utterance = bytes(self._buf)
        speech_ms = self._speech_ms
        self.reset()
        if speech_ms < self.config.min_speech_ms:
            logger.debug("VAD dropped mic bump (%dms < %dms)", speech_ms, self.config.min_speech_ms)
            return None
        return utterance if utterance else None

    def flush(self) -> bytes | None:
        """Mic released mid-utterance: collect what speech exists, if any."""
        if not self.has_pending_speech:
            return None
        return self._finish()


__all__ = ["VoiceConfig", "VoiceTurnTaker"]
