"""Session control: graceful goodbye (quits the HUD after farewell plays)."""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from hud.skills.loader import get_provider

logger = logging.getLogger(__name__)


@tool
def say_goodbye() -> str:
    """Ends the session and closes Jarvis. Call when the user says goodbye,
    asks to quit/exit/close, or is done for now. Say a short farewell
    (the app closes a few seconds after the farewell plays)."""
    request_quit = get_provider("on_goodbye")
    if request_quit is None:
        return "ERROR: can't quit from this frontend — close the window instead."
    try:
        request_quit()
    except Exception as e:
        logger.warning("goodbye quit request failed: %s", e)
        return f"ERROR: couldn't close the app ({e})."
    logger.info("goodbye requested; app will close after farewell")
    return "Goodbye! Closing now."


SKILL = {
    "name": "session",
    "version": "1.0",
    "description": "Say goodbye and close the HUD gracefully.",
    "risk": "readonly",
    "tools": [say_goodbye],
}
