"""Language routing for HUD turns: TTS engine/voice + model suggestion.

- Tamil (``ta*``) → cloud EdgeTTS Valluvar (male, per owner choice).
  Understanding needs a multilingual model: if the active model looks
  English-centric, suggest the first installed Qwen (callers decide
  whether to auto-switch via ``core.set_chat_model``).
- Anything else → offline Piper (current configured voice), no model change.

Pure functions + one Ollama-list lookup; Qt-free and unit-tested.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

TAMIL_VOICE = "ta-IN-ValluvarNeural"
# Measured Sep 2026 on this machine (Tamil math/poem/tool probes):
# gemma-4-E2B-it > qwen3.5:2b on fluency and correctness, so gemma first.
TAMIL_MODEL_HINTS = ("gemma", "qwen")


def decide(lang: str, current_model: str = "") -> dict:
    """Decide engine/voice for a turn. Never raises, never touches I/O."""
    code = (lang or "").strip().lower()
    if code.startswith("ta"):
        return {"engine": "edge", "voice": TAMIL_VOICE, "lang": "ta"}
    return {"engine": "piper", "voice": "", "lang": code or "en"}


def guess_text_lang(text: str) -> str:
    """Script sniff for typed input: Tamil block → "ta", else "en"."""
    for ch in text or "":
        if "\u0b80" <= ch <= "\u0bff":
            return "ta"
    return "en"


def needs_multilingual_model(lang: str, current_model: str) -> bool:
    """True when a Tamil turn meets a model unlikely to understand it."""
    if not (lang or "").strip().lower().startswith("ta"):
        return False
    model = (current_model or "").lower()
    return not any(hint in model for hint in TAMIL_MODEL_HINTS)


async def suggest_tamil_model() -> str | None:
    """Best installed Tamil-capable model, or None when none is local.

    Hint order is preference order (gemma before qwen, measured).
    """
    try:
        import core

        models = await core.list_ollama_models()
    except Exception:
        logger.debug("model list for Tamil routing failed", exc_info=True)
        return None
    names = [m.get("name", "") for m in models]
    for hint in TAMIL_MODEL_HINTS:
        for name in names:
            if hint in name.lower():
                return name
    return None


__all__ = ["TAMIL_VOICE", "TAMIL_MODEL_HINTS", "decide", "guess_text_lang", "needs_multilingual_model", "suggest_tamil_model"]
