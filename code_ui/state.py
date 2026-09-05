"""Per-browser-session state: workspace, brain, tracker, chat-session keys."""
from __future__ import annotations

import uuid
from pathlib import Path

import chainlit as cl

from code_assistant.brain import CodeBrain
from code_assistant.modes import Mode
from code_assistant.sessions import SessionStore
from code_assistant.workspace import Workspace, WorkspaceRegistry
from llm_provider import LLMConfig
from token_usage import TokenTracker


def _registry() -> WorkspaceRegistry:
    return WorkspaceRegistry()


def _sessions() -> SessionStore:
    return SessionStore()


# -------------------------------------------------------------- session keys


def _session_id() -> str:
    sid = cl.user_session.get("session_id")
    if not sid:
        sid = f"code:{uuid.uuid4().hex[:12]}"
        cl.user_session.set("session_id", sid)
    return sid


def _workspace_root() -> str | None:
    return cl.user_session.get("workspace_root")


def _current_mode() -> Mode:
    raw = cl.user_session.get("mode")
    return Mode.parse(raw) if raw else Mode.PLAN


def _brain() -> CodeBrain | None:
    return cl.user_session.get("brain")


def _tracker() -> TokenTracker | None:
    return cl.user_session.get("tracker")


def _chat_session_id() -> str | None:
    return cl.user_session.get("chat_session_id")


def _chat_session_title() -> str | None:
    return cl.user_session.get("chat_session_title")


# ----------------------------------------------------------- workspace load


def _open_workspace(root_str: str) -> Workspace:
    ws = Workspace(Path(root_str).expanduser())
    ws.ensure_sandbox()
    _registry().touch(ws)
    return ws


def _build_brain(
    workspace: Workspace, tracker_sid: str | None = None
) -> tuple[CodeBrain, TokenTracker]:
    cfg = LLMConfig.from_env()
    tracker = TokenTracker(
        session_id=tracker_sid or _session_id(),
        provider=cfg.provider,
        model=cfg.model,
    )
    brain = CodeBrain(
        workspace=workspace,
        llm_config=cfg,
        tracker=tracker,
        mode=_current_mode(),
    )
    return brain, tracker


def _tracker_sid_for(chat_id: str) -> str:
    """Stable token-usage id per chat session (survives restarts)."""
    return f"code:{chat_id}"
