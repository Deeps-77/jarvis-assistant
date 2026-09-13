"""HUD-local skill registry. Drop a ``.py`` file in this directory declaring::

    SKILL = {
        "name": "open_app",
        "version": "1.0",
        "description": "Launch applications by name (asks first).",
        "risk": "system",  # or "readonly"
        "tools": [open_app],
    }

Tools are LangChain ``@tool`` functions (same pattern as ``chat_tools.py``).
Each file imports in isolation — one broken skill logs a warning and the
rest still load. ``readonly`` skills are always on; ``system`` skills only
join turns when armed (HUD checkbox / ``HUD_SKILLS_ALLOW``). Telegram, web
and voice-console never see these tools.
"""

from .loader import (
    RISKS,
    describe,
    direct_routes,
    get_provider,
    load_skills,
    set_provider,
    set_system_armed,
    set_turn_context,
    turn_tools,
)

__all__ = ["RISKS", "describe", "direct_routes", "get_provider", "load_skills", "set_provider", "set_system_armed", "set_turn_context", "turn_tools"]
