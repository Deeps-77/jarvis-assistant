"""HUD TTS: multi-tier offline-first speech synthesis.

English (fully offline): Kokoro (neural, ~330 MB weights) → Piper
(bundled ONNX) → pyttsx3 (Windows SAPI).
Tamil: EdgeTTS Valluvar (cloud, free) → Piper English notice → pyttsx3.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile

logger = logging.getLogger(__name__)

OFFLINE_NOTICE = "Sorry, the Tamil voice needs internet. Here is the text: "
TAMIL_VOICE_EDGE = "ta-IN-ValluvarNeural"  # EdgeTTS Tamil voice (owner choice)

# Kokoro pipelines are ~330 MB each — build once, reuse forever.
_KOKORO_PIPELINES: dict[str, object] = {}


def kokoro_ready() -> bool:
    """True once any Kokoro pipeline is built (startup warmup or first use)."""
    return bool(_KOKORO_PIPELINES)


async def warmup_kokoro() -> bool:
    """Build the English pipeline now so sentence one is already warm.

    Safe to call repeatedly; non-fatal by contract (tiers cover absence).
    Returns True when a pipeline is ready afterwards.
    """
    if kokoro_ready():
        return True
    try:
        out = await _kokoro_synthesize("warmup.", "en")
    except Exception:
        logger.debug("HUD Kokoro warmup raised", exc_info=True)
        return False
    ready = kokoro_ready() and bool(out)
    logger.info("HUD Kokoro warmup %s", "done" if ready else "unavailable")
    return ready
# Kokoro 0.9 lang codes (NOT BCP-47): a=US-en, b=UK-en, e=es, f=fr-fr,
# h=hi, i=it, p=pt-br, j=ja, z=zh.
_KOKORO_LANGS = {"en": "a", "hi": "h", "es": "e", "fr": "f", "it": "i",
                 "pt": "p", "ja": "j", "zh": "z"}


# ---------------------------------------------------------------------------
# Pyttsx3 helper — Windows SAPI, always available, synchronous
# ---------------------------------------------------------------------------

def _pyttsx3_speak_to_wav(text: str) -> bytes:
    """Synthesize text to WAV bytes using pyttsx3 (Windows SAPI)."""
    import pyttsx3
    import io
    import wave
    import struct

    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    engine.setProperty("volume", 0.95)

    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        engine.save_to_file(text, tmp)
        engine.runAndWait()
        engine.stop()
        with open(tmp, "rb") as fh:
            data = fh.read()
        if len(data) > 44:  # has audio beyond WAV header
            return data
        # pyttsx3 sometimes writes an empty file — build minimal silence WAV
        logger.warning("HUD pyttsx3: empty WAV, returning silence stub")
        return _silence_wav(0.5)
    except Exception:
        logger.exception("HUD pyttsx3 synthesis failed")
        return b""
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def _silence_wav(secs: float = 0.5, rate: int = 16000) -> bytes:
    """Return a minimal silent WAV (for error cases)."""
    import struct
    import wave
    import io
    n = int(rate * secs)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Kokoro helper — optional, offline neural TTS
# ---------------------------------------------------------------------------

async def _kokoro_synthesize(text: str, lang: str = "en-us") -> bytes | None:
    """Synthesize with Kokoro if installed. Returns WAV bytes or None.

    Pipelines are cached by language — a rebuild per sentence would reload
    ~330 MB every time.
    """
    try:
        from kokoro import KPipeline  # type: ignore[import]
        import numpy as np
        import soundfile as sf  # type: ignore[import]
        import io
    except ImportError:
        return None  # Kokoro not installed, silent skip
    try:
        key = _KOKORO_LANGS.get((lang or "en")[:2].lower(), "a")
        pipeline = _KOKORO_PIPELINES.get(key)
        if pipeline is None:
            # CPU-pinned on purpose: on 4GB VRAM the GPU is already shared by
            # whisper-CUDA and Ollama, and CUDA Kokoro is only ~1s faster
            # per sentence (4.1s vs 5.0s measured). Override via HUD_KOKORO_DEVICE.
            import os as _os

            device = _os.environ.get("HUD_KOKORO_DEVICE", "").strip() or "cpu"
            logger.info("HUD Kokoro: loading pipeline (%s, device=%s)…", key, device)
            pipeline = await asyncio.to_thread(KPipeline, lang_code=key, device=device)
            _KOKORO_PIPELINES[key] = pipeline
            logger.info("HUD Kokoro: pipeline ready")

        def _run():
            import os
            voice = os.environ.get("HUD_KOKORO_VOICE", "am_michael").strip() or "am_michael"
            audio_parts = []
            for _, _, audio in pipeline(text, voice=voice, speed=1.0):
                if audio is not None:
                    try:
                        arr = audio.detach().cpu().numpy()
                    except Exception:
                        arr = np.asarray(audio, dtype=np.float32)
                    audio_parts.append(arr)
            if not audio_parts:
                return None
            full = np.concatenate(audio_parts)
            buf = io.BytesIO()
            sf.write(buf, full, 24000, format="WAV")
            return buf.getvalue()

        return await asyncio.to_thread(_run)
    except Exception:
        logger.warning("HUD Kokoro synthesis failed", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Main HudSpeaker
# ---------------------------------------------------------------------------

class HudSpeaker:
    """Speak ``text`` in ``lang``. Returns audio bytes, or b"" on failure.

    English stays fully offline: Kokoro (best quality, if installed) →
    Piper (bundled) → pyttsx3 (Windows SAPI). Tamil uses cloud EdgeTTS
    (Valluvar) with the same offline chain as fallback (Piper speaks an
    English notice there since it has no Tamil voice).
    """

    def __init__(self, piper=None) -> None:
        from speech import PiperSpeaker

        self._piper = piper or PiperSpeaker()
        self._lock = asyncio.Lock()

    async def synthesize(self, text: str, lang: str = "en") -> bytes:
        import time as _time

        clean = (text or "").strip()
        if not clean:
            return b""
        t0 = _time.perf_counter()
        lang = (lang or "en").strip().lower() or "en"
        logger.info("HUD TTS start (lang=%s, %d chars)", lang, len(clean))

        is_tamil = lang.startswith("ta")

        if is_tamil:
            # --- Tamil: EdgeTTS (cloud, Valluvar) ---
            try:
                result = await self._edge(clean, TAMIL_VOICE_EDGE)
                if result:
                    logger.info(
                        "HUD TTS done: edge %d bytes in %.1fs",
                        len(result), _time.perf_counter() - t0,
                    )
                    return result
            except Exception:
                logger.warning("HUD TTS: EdgeTTS failed, trying Piper", exc_info=True)
            clean = OFFLINE_NOTICE + clean
        elif kokoro_ready():
            # --- English, best quality: Kokoro (pipeline already warm) ---
            kokoro_bytes = await _kokoro_synthesize(clean)
            if kokoro_bytes:
                logger.info(
                    "HUD TTS done: kokoro %d bytes in %.1fs",
                    len(kokoro_bytes), _time.perf_counter() - t0,
                )
                return kokoro_bytes
        else:
            # Cold start: Piper answers in ~3s while Kokoro warms up in the
            # background. Silence gaps are worse than slightly lower fidelity.
            logger.info("HUD TTS: kokoro cold — Piper covers until warm")

        # --- Piper (offline ONNX, bundled) ---
        try:
            result = await self._piper.synthesize(clean)
            if result:
                logger.info(
                    "HUD TTS done: piper %d bytes in %.1fs",
                    len(result), _time.perf_counter() - t0,
                )
                return result
        except Exception:
            logger.warning("HUD TTS: Piper failed, trying pyttsx3", exc_info=True)

        # --- pyttsx3 (Windows SAPI — last resort) ---
        try:
            result = await asyncio.to_thread(_pyttsx3_speak_to_wav, clean)
            if result:
                logger.info(
                    "HUD TTS done: pyttsx3 %d bytes in %.1fs",
                    len(result), _time.perf_counter() - t0,
                )
                return result
        except Exception:
            logger.exception("HUD TTS: pyttsx3 failed — all tiers exhausted")

        logger.warning("HUD TTS exhausted all tiers (%.1fs)", _time.perf_counter() - t0)
        return b""

    async def _edge(self, text: str, voice: str) -> bytes:
        import edge_tts

        fd, tmp = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        try:
            await edge_tts.Communicate(text, voice).save(tmp)
            with open(tmp, "rb") as fh:
                return fh.read()
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass


__all__ = ["HudSpeaker", "OFFLINE_NOTICE", "TAMIL_VOICE_EDGE", "kokoro_ready", "warmup_kokoro"]
