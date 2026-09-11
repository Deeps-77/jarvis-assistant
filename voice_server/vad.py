"""Turn endpointing: energy pre-gate + Silero confirmation.

Always-on listening must ignore fans, clicks and mic bumps without
missing real speech. The cheap energy gate (see ``voice.VoiceTurnTaker``)
runs per frame; Silero (bundled with faster-whisper, ONNX CPU, ~2MB
one-time download) validates each endpointed candidate before it reaches
STT. Fail-open: if Silero can't load, candidates pass straight through.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_silero_checked: bool | None = None


def silero_available() -> bool:
    """True when faster-whisper's bundled Silero VAD imports cleanly."""
    global _silero_checked
    if _silero_checked is None:
        try:
            import faster_whisper.vad  # noqa: F401

            _silero_checked = True
        except ImportError:
            _silero_checked = False
    return _silero_checked


def _to_float32(pcm: bytes):
    try:
        import numpy as np

        return np.frombuffer(pcm, dtype=np.int16).astype("float32") / 32768.0
    except ImportError:
        return None


def has_speech(pcm_16k: bytes, min_speech_ms: int = 250) -> bool:
    """Silero check: does this 16kHz utterance contain real voiced audio?"""
    if not pcm_16k:
        return False
    audio = _to_float32(pcm_16k)
    if audio is None:
        return True  # no numpy: fail open
    try:
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        threshold = float(os.environ.get("VOICE_SILERO_THRESHOLD", "0.5"))
        stamps = get_speech_timestamps(
            audio,
            VadOptions(
                threshold=threshold,
                min_speech_duration_ms=min_speech_ms,
                min_silence_duration_ms=200,
            ),
        )
        return len(stamps) > 0
    except Exception:
        logger.exception("silero validation failed; passing audio through")
        return True


class VoiceEndpoint:
    """Energy endpointing with Silero validation on each candidate.

    Wraps :class:`voice.VoiceTurnTaker` so the frame math stays in one
    place; this class only decides whether a candidate is worth
    transcribing. Construct with a 16kHz config.
    """

    def __init__(self, taker=None, validate: bool = True) -> None:
        if taker is None:
            from voice import VoiceConfig, VoiceTurnTaker

            taker = VoiceTurnTaker(VoiceConfig(input_rate=16000))
        self._taker = taker
        self._validate = validate

    def reset(self) -> None:
        self._taker.reset()

    @property
    def has_pending_speech(self) -> bool:
        return self._taker.has_pending_speech

    @property
    def pending_ms(self) -> int:
        return self._taker.speech_ms

    def set_threshold(self, value: float) -> float:
        """Live-tune the energy gate (e.g. from the console slider)."""
        try:
            tuned = float(value)
        except (TypeError, ValueError):
            return self._taker.config.threshold
        tuned = min(5000.0, max(50.0, tuned))
        self._taker.config.threshold = tuned
        return tuned

    def _checked(self, candidate: bytes | None) -> bytes | None:
        if not candidate:
            return None
        if self._validate and not has_speech(candidate):
            logger.debug("VAD dropped non-speech candidate (%d bytes)", len(candidate))
            return None
        return candidate

    def feed(self, pcm_16k: bytes) -> bytes | None:
        """Consume 16kHz mic audio. Returns a validated utterance or None."""
        return self._checked(self._taker.feed(pcm_16k))

    def flush(self) -> bytes | None:
        """Mic/session stopping: collect validated partial speech, if any."""
        return self._checked(self._taker.flush())


__all__ = ["VoiceEndpoint", "has_speech", "silero_available"]
