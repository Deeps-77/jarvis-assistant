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
TAMIL_MODEL_HINTS = ("qwen",)


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
    """First installed Qwen model name, or None when none is local."""
    try:
        import core

        models = await core.list_ollama_models()
    except Exception:
        logger.debug("model list for Tamil routing failed", exc_info=True)
        return None
    for m in models:
        name = m.get("name", "")
        if any(hint in name.lower() for hint in TAMIL_MODEL_HINTS):
            return name
    return None


__all__ = ["TAMIL_VOICE", "TAMIL_MODEL_HINTS", "decide", "guess_text_lang", "needs_multilingual_model", "suggest_tamil_model"]
