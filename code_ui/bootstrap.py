"""Code-UI bootstrap: paths, env, secrets, logging, shared constants.

Imported first by every code_ui.* module. Depends only on stdlib,
botlog and paths — never on sibling modules.
"""
from __future__ import annotations

import logging
import os
import secrets
import sys
from pathlib import Path
from typing import Any

from paths import PROJECT_ROOT

import botlog as _botlog

# Ensure the project root is on sys.path so ``code_assistant`` imports work.
sys.path.insert(0, str(PROJECT_ROOT))

# Chainlit reads several env vars at import time — set them before import.
os.environ.setdefault("LITERAL_TELEMETRY_OFF", "true")

logger = logging.getLogger(__name__)


_secret_file = PROJECT_ROOT / ".code_ui.secret"
if not _secret_file.exists():
    _secret_file.write_text(secrets.token_hex(32), encoding="utf-8")
os.environ.setdefault(
    "CHAINLIT_AUTH_SECRET", _secret_file.read_text(encoding="utf-8").strip()
)
# ----------------------------------------------------------------- logging

# Wire the code UI into the shared botlog sinks (jarvis.log diagnostics +
# activity.log) so every session/turn/tool is captured alongside the chat UI.
_sa = getattr(_botlog, "setup_logging", None)
if callable(_sa) and os.environ.get("CODE_UI_DISABLE_LOGGING", "") != "1":
    try:
        _sa()
    except Exception:
        logger.exception("botlog.setup_logging failed")


# ----------------------------------------------------------------- helpers


UI_HOST = os.environ.get("CODE_UI_HOST", "0.0.0.0")
UI_PORT = int(os.environ.get("CODE_UI_PORT", "8500"))


# --------------------------------------------------------------------- env


def _ensure_chainlit_dir(base: Path) -> None:
    """Make sure ``.chainlit/config.toml`` exists with the features we need.

    The chat UI (``app.py``) also writes to this file. We only add the audio
    feature flag if missing and otherwise leave existing config alone so the
    two apps don't fight each other.
    """
    cfg = base / ".chainlit" / "config.toml"
    cfg.parent.mkdir(parents=True, exist_ok=True)
    if not cfg.exists():
        cfg.write_text(
            "[project]\n\n[features.audio]\nenabled = false\n"
            "\n[features.spontaneous_file_upload]\nenabled = false\n",
            encoding="utf-8",
        )
        return
    text = cfg.read_text(encoding="utf-8")
    if "[features.audio]" not in text:
        text += "\n[features.audio]\nenabled = false\n"
        cfg.write_text(text, encoding="utf-8")


_ensure_chainlit_dir(PROJECT_ROOT)


def log_code_event(event: str, **fields: Any) -> None:
    """Emit a code_UI event to activity log + events.jsonl.

    Mirrors botlog's pattern: a human-readable activity line plus a
    machine-readable JSON event. Kept local to avoid bloating botlog with
    code-assistant-specific fields.
    """
    try:
        _botlog._activity.info("💻 %s", _format_code_activity(event, fields))
    except Exception:
        pass
    try:
        _botlog._jsonl(f"code_ui.{event}", **{k: str(v) for k, v in fields.items()})
    except Exception:
        pass
    logger.info("[%s] %s", event, fields)


def _format_code_activity(event: str, fields: dict[str, Any]) -> str:
    if event == "workspace_open":
        return f"Workspace opened: {fields.get('root')}"
    if event == "workspace_switch":
        return f"Workspace switched: {fields.get('root')}"
    if event == "mode":
        return f"Mode → {fields.get('mode')}"
    if event == "tool_start":
        return f"⚙ tool: {fields.get('name')} {fields.get('args')}"
    if event == "tool_end":
        return f"  └ {fields.get('name')} {'rejected' if fields.get('rejected') else 'ok'}"
    if event == "approval":
        return f"Approval {fields.get('decision')} for {fields.get('name')}"
    if event == "turn":
        return f"Turn done in {fields.get('duration_ms')}ms ({fields.get('model')})"
    if event == "error":
        return f"⚠ {fields.get('message')}"
    return f"{event}: {fields}"


def _password() -> str:
    return os.environ.get("CHAINLIT_CODE_PASSWORD", "")


def _username() -> str:
    return os.environ.get("CHAINLIT_CODE_USERNAME", "codeuser")
