"""Look at the screen through the vision model ("what do you see?").

Screenshot capture must run on the GUI thread, so the window registers a
``screen_grabber`` provider (see hud.skills.loader.set_provider). Without
one, the tool reports cleanly instead of touching Qt off-thread.
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from hud.skills.loader import get_context, get_provider
from security import sanitize_external

logger = logging.getLogger(__name__)


@tool
async def look_at_screen(question: str = "What do you see on my screen?") -> str:
    """Captures the screen and describes it with vision.

    MUST be called when the user asks what is on screen, what you see, or
    to look at something. There is no visual ability without this tool.
    Describe what is visible and answer the user's question about it.
    """
    grabber = get_provider("screen_grabber")
    if grabber is None:
        return "ERROR: screen capture is unavailable in this frontend."
    try:
        png = grabber()
    except Exception as e:
        logger.warning("look_at_screen grab failed: %s", e)
        return f"ERROR: couldn't capture the screen ({e})."
    if not png:
        return "ERROR: screen capture returned nothing."
    chat_key = get_context("chat_key", "hud:main")
    owner = get_context("owner", "hud")
    try:
        import core

        body, failed = await core.vision_respond(
            str(chat_key), str(owner), bytes(png), "png", question or "What do you see?"
        )
    except Exception as e:
        logger.warning("look_at_screen vision failed: %s", e)
        return f"ERROR: couldn't analyze the screen ({e})."
    if failed or not (body or "").strip():
        return "ERROR: I couldn't make out the screen contents reliably."
    return sanitize_external(f"What I see:\n{body}", "screen capture")


SKILL = {
    "name": "look",
    "version": "1.0",
    "description": "Look at the screen and describe it (uses the vision model).",
    "risk": "readonly",
    "tools": [look_at_screen],
}
