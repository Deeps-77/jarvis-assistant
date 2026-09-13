"""HUD speech-to-text: Tamil and English only, auto-picked per utterance.

Unlike the :8600 console (pinned ``language="en"``), the HUD lets whisper
choose — but short clips confuse acoustic LID (Tamil heard as Malayalam,
Bengali, …). So: first pass auto-detects; anything outside {ta, en} (or
low confidence) gets a forced ta-vs-en second opinion and the winner is
kept. Returns (text, lang) with lang in {"ta", "en", ""}.
Fail-soft: never disables the shared transcriber, just reports empties.

CUDA DLL setup is performed at import time so faster-whisper finds the
right NVIDIA libraries on Windows before the model is ever loaded.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Windows CUDA DLL helper — ported from reference Jarvis repo
# ---------------------------------------------------------------------------

def _setup_cuda_dll_paths() -> None:
    """Ensure CUDA / cuDNN DLLs in nvidia/ site-packages are on PATH/DLL search.

    On Windows, faster-whisper loads ctypes DLLs by name.  The nvidia-cuda-*
    wheels install libraries under <site-packages>/nvidia/*/lib/ (or /bin/).
    Without this, the loader falls back to CPU even when a GPU is present,
    and the first transcription can be very slow (30+ s on large audio),
    which looks like a hang.
    """
    if sys.platform != "win32":
        return
    import site

    candidate_dirs: set[str] = set()
    site_dirs: list[str] = []
    try:
        site_dirs.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site:
            site_dirs.append(user_site)
    except Exception:
        pass

    for s_dir in site_dirs:
        nvidia_dir = os.path.join(s_dir, "nvidia")
        if os.path.isdir(nvidia_dir):
            for root, dirs, _files in os.walk(nvidia_dir):
                for sub in ("bin", "lib"):
                    if sub in dirs:
                        candidate_dirs.add(os.path.join(root, sub))
        torch_lib = os.path.join(s_dir, "torch", "lib")
        if os.path.isdir(torch_lib):
            candidate_dirs.add(torch_lib)

    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    added: list[str] = []
    for c_dir in sorted(candidate_dirs):
        if not os.path.isdir(c_dir):
            continue
        try:
            os.add_dll_directory(c_dir)
        except Exception:
            pass
        if c_dir not in path_entries:
            path_entries.insert(0, c_dir)
            added.append(c_dir)
    if added:
        os.environ["PATH"] = os.pathsep.join(path_entries)
        logger.debug("HUD STT: added CUDA DLL dirs: %s", added)


# Run once at import time so faster-whisper can find its CUDA libs.
_setup_cuda_dll_paths()


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

#: Languages the HUD supports. Anything else falls back to a ta-vs-en
#: second opinion (short clips confuse acoustic language ID).
SUPPORTED_LANGS = ("ta", "en")
#: Accept first-pass LID at/above this probability.
LID_CONFIDENCE = 0.5


def _decode(model, tmp: str, language: str | None) -> tuple[str, str, float]:
    """One whisper pass. Returns (text, lang, language_probability)."""
    kwargs: dict = dict(
        beam_size=1,
        condition_on_previous_text=False,
        temperature=0.0,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
    )
    if language:
        kwargs["language"] = language
    segments, info = model.transcribe(tmp, **kwargs)
    text = " ".join(s.text.strip() for s in segments).strip()
    lang = str(getattr(info, "language", "") or "")
    prob = float(getattr(info, "language_probability", 0.0) or 0.0)
    return text, lang, prob


async def transcribe_with_lang(data: bytes, filename: str = "hud.wav") -> tuple[str, str]:
    """Transcribe WAV bytes. Returns (text, lang) with lang in ta/en/""."""
    import core

    tr = core.speech_transcriber
    logger.info(
        "HUD STT entry: transcriber=%s enabled=%s model=%s",
        type(tr).__name__ if tr else None,
        getattr(tr, "enabled", None),
        type(getattr(tr, "_model", None)).__name__ if tr else None,
    )

    if tr is None or not tr.enabled or tr._model is None:
        # Fall back to the standard path (loads the model if needed);
        # language is unknown there, so report "".
        logger.info("HUD STT: using core.transcribe_audio fallback")
        text = await core.transcribe_audio(data, filename)
        return text, ""

    fd, tmp = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data if isinstance(data, (bytes, bytearray)) else b"")
        result: list[tuple[str, str]] = []

        def _run():
            logger.info("HUD STT: faster-whisper transcribing %d bytes…", len(data))
            try:
                text, lang, prob = _decode(tr._model, tmp, None)
                logger.info(
                    "HUD STT: first pass — %d chars, lang=%s (p=%.2f)",
                    len(text), lang or "?", prob,
                )
                if (lang not in SUPPORTED_LANGS or prob < LID_CONFIDENCE) and text:
                    # Short-clip LID noise (e.g. Tamil heard as Malayalam):
                    # forced ta-vs-en runoff, higher confidence wins.
                    ta_text, _, ta_prob = _decode(tr._model, tmp, "ta")
                    en_text, _, en_prob = _decode(tr._model, tmp, "en")
                    logger.info(
                        "HUD STT: runoff ta(p=%.2f,%dch) vs en(p=%.2f,%dch)",
                        ta_prob, len(ta_text), en_prob, len(en_text),
                    )
                    if (ta_prob, len(ta_text)) >= (en_prob, len(en_text)):
                        text, lang = ta_text, "ta"
                    else:
                        text, lang = en_text, "en"
                result.append((text, lang if lang in SUPPORTED_LANGS else ""))
                logger.info(
                    "HUD STT: faster-whisper done — %d chars, lang=%s",
                    len(text), lang or "?"
                )
            except Exception:
                logger.exception("HUD STT: faster-whisper transcription error")

        async with tr._transcribe_lock:
            await asyncio.to_thread(_run)

        if result:
            text, lang = result[0]
            logger.info("HUD transcribed %d chars (lang=%s)", len(text), lang or "?")
            return text, lang
        return "", ""
    except Exception:
        logger.exception("HUD transcription failed")
        return "", ""
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


__all__ = ["transcribe_with_lang", "SUPPORTED_LANGS", "LID_CONFIDENCE"]
