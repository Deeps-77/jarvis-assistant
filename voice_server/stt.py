"""Utterance transcription with voice-tuned faster-whisper settings.

Streaming recipe (not batch defaults): greedy decode, no cross-chunk
conditioning, pinned language, VAD pre-trim. Reuses the shared
``core.speech_transcriber`` model instance.
"""

from __future__ import annotations

import asyncio
import io
import logging
import wave

logger = logging.getLogger(__name__)


def wav_bytes(pcm_16k: bytes, rate: int = 16000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm_16k)
    return buffer.getvalue()


async def warmup() -> None:
    """Front-load whisper/CUDA init so the first real turn isn't the slow one."""
    import core

    if core.speech_transcriber is None or not core.speech_transcriber.enabled:
        return
    try:
        silence = b"\x00" * 16000 * 2 // 2  # 0.5s of 16kHz silence
        await core.transcribe_audio(wav_bytes(silence), "warmup.wav")
    except Exception:
        logger.exception("stt warmup failed (non-fatal)")
    logger.info("stt warmup done")


async def transcribe_utterance(data, filename: str = "utterance.wav") -> str:
    """Transcribe one endpointed utterance. Returns text (possibly "")."""
    import core

    if core.speech_transcriber is None or not core.speech_transcriber.enabled:
        return ""
    model = core.speech_transcriber._model
    if model is None:
        # Fall back to the standard path (loads the model if needed).
        return await core.transcribe_audio(data, filename)

    # The shared model serializes inference on its own lock; write bytes
    # to temp and transcribe with streaming (not batch) options inline.
    import os
    import tempfile

    fd, tmp = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data if isinstance(data, (bytes, bytearray)) else b"")
        loop_result: list[str] = []

        def _run_file():
            segments, _info = model.transcribe(
                tmp,
                beam_size=1,
                condition_on_previous_text=False,
                language="en",
                temperature=0.0,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 300},
            )
            loop_result.append(" ".join(s.text.strip() for s in segments).strip())

        async with core.speech_transcriber._transcribe_lock:
            await asyncio.to_thread(_run_file)
        return loop_result[0] if loop_result else ""
    except Exception:
        logger.exception("voice transcription failed")
        return ""
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
