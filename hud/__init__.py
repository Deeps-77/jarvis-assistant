"""Jarvis desktop HUD (PyQt6): one local window for chat + voice.

Separate frontend alongside Telegram / Chainlit web / :8600 voice console.
Owns nothing brain-side: chat turns go through ``core.respond`` (persistent
``hud:*`` session, memory on), STT reuses the shared faster-whisper model
with auto language detection, TTS is offline Piper or cloud EdgeTTS
(Tamil Valluvar) with Piper fallback. Nothing here touches the frozen
code assistant.
"""

__all__: list[str] = []
