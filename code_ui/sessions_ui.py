"""Per-workspace chat sessions UI: picker, activate, persist, replay."""
from __future__ import annotations

from typing import Any

import chainlit as cl

from code_assistant.modes import Mode
from code_assistant.sessions import ChatSession, auto_title
from llm_provider import LLMConfig
from token_usage import TokenTracker

from .bootstrap import log_code_event
from .chrome import _push_settings, _refresh_sidebar
from .state import (
    _brain,
    _build_brain,
    _current_mode,
    _open_workspace,
    _tracker,
    _tracker_sid_for,
    _workspace_root,
)


# --------------------------------------------------------------- chat sessions


def _session_label(sess: ChatSession) -> str:
    title = sess.title if len(sess.title) <= 26 else sess.title[:25] + "…"
    return f"💬 {title}"


def _session_desc(sess: ChatSession) -> str:
    import datetime as _dt
    try:
        when = _dt.datetime.fromtimestamp(sess.updated).astimezone().strftime("%m-%d %H:%M")
    except (OSError, OverflowError, ValueError):
        when = "?"
    n = len(sess.messages)
    return f"{when} · {sess.mode} · {n} msgs · `{sess.id[:8]}`"


def _msg_text(content: Any) -> str:
    """Stringify a LangChain message content (str or block list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content or "")


#: Max messages replayed into the chat pane on session resume.
REPLAY_LIMIT = 50


async def _replay_session_messages(sess: ChatSession) -> None:
    """Render the saved history into the chat pane (restores visible context)."""
    messages = sess.messages[-REPLAY_LIMIT:]
    if len(sess.messages) > REPLAY_LIMIT:
        await cl.Message(
            content=f"_…showing last {REPLAY_LIMIT} of {len(sess.messages)} messages (full history kept in context)._"
        ).send()
    pending_tools = 0
    for m in messages:
        kind = m.__class__.__name__
        if kind == "HumanMessage":
            text = _msg_text(m.content).strip()
            if text:
                await cl.Message(content=text, author="You", type="user_message").send()
        elif kind == "AIMessage":
            text = _msg_text(m.content).strip()
            if text:
                note = f"\n\n_🔧 {pending_tools} tool step(s) ran._" if pending_tools else ""
                await cl.Message(content=text + note).send()
                pending_tools = 0
            elif getattr(m, "tool_calls", None):
                pending_tools += len(m.tool_calls)
        elif kind == "ToolMessage":
            pending_tools += 1
    if pending_tools:
        await cl.Message(content=f"_🔧 {pending_tools} tool step(s) ran._").send()


async def _activate_session(ws: Workspace, sess: ChatSession) -> None:
    """Build brain+tracker for ``sess`` and make it the active chat."""
    cl.user_session.set("mode", Mode.parse(sess.mode).value)
    brain, tracker = _build_brain(ws, tracker_sid=_tracker_sid_for(sess.id))
    tracker.rehydrate(_tracker_sid_for(sess.id))
    brain.import_history(sess.messages)
    cl.user_session.set("workspace_root", str(ws.root))
    cl.user_session.set("brain", brain)
    cl.user_session.set("tracker", tracker)
    cl.user_session.set("chat_session_id", sess.id)
    cl.user_session.set("chat_session_title", sess.title)
    cl.user_session.set("visible_text", [])
    log_code_event("session_open", root=str(ws.root), session=sess.id, title=sess.title)
    await _replay_session_messages(sess)
    n = len(sess.messages)
    resumed = f" ({n} messages restored)" if n else " (fresh chat)"
    await cl.Message(
        content=f"💬 Session **{sess.title}** active — `{sess.id[:8]}`{resumed}"
    ).send()
    await _push_settings()
    await _refresh_sidebar()


async def _show_session_picker(workspace_root: str) -> None:
    """List saved chats for the workspace, or start the first one."""
    try:
        ws = _open_workspace(workspace_root)
    except Exception as e:
        await cl.Message(content=f"Cannot open workspace: {e}").send()
        return
    store = _sessions()
    sessions = store.list(str(ws.root))
    if not sessions:
        sess = store.create(str(ws.root), title="Untitled session", mode=_current_mode().value)
        await _activate_session(ws, sess)
        return
    actions: list[cl.Action] = [
        cl.Action(
            name="sess_new",
            label="＋ New session",
            payload={"root": str(ws.root)},
            description="Start a fresh chat in this workspace",
        )
    ]
    for sess in sessions[:8]:
        actions.append(
            cl.Action(
                name="sess_open",
                label=_session_label(sess),
                payload={"root": str(ws.root), "id": sess.id},
                description=_session_desc(sess),
            )
        )
    extra = f"\n\n_Showing {min(len(sessions), 8)} of {len(sessions)} — `/sessions` lists all._" if len(sessions) > 8 else ""
    await cl.Message(
        content=f"💬 **Chats in `{ws.root}`** — pick one to resume, or start new:{extra}",
        actions=actions,
    ).send()


async def _ensure_session(first_text: str = "") -> tuple[Workspace, ChatSession] | None:
    """Return the active (workspace, session), auto-creating when needed."""
    root = _workspace_root()
    if not root:
        return None
    try:
        ws = _open_workspace(root)
    except Exception as e:
        await cl.Message(content=f"Cannot open workspace: {e}").send()
        return None
    store = _sessions()
    sid = _chat_session_id()
    if sid:
        sess = store.get(str(ws.root), sid)
        if sess is not None:
            return ws, sess
    # No active session (or it was deleted elsewhere): start one silently so
    # typing first still works without forcing the picker.
    sess = store.create(str(ws.root), title=auto_title(first_text), mode=_current_mode().value)
    await _activate_session(ws, sess)
    return ws, sess


async def _persist_session() -> None:
    """Save the active brain history + mode (+auto-title) to the store."""
    root = _workspace_root()
    sid = _chat_session_id()
    brain = _brain()
    if not root or not sid or not brain:
        return
    store = _sessions()
    sess = store.get(root, sid)
    if sess is None:
        return
    title = None
    if sess.title == "Untitled session":
        # Derive from the first human message once content exists.
        for m in brain.export_history():
            if m.__class__.__name__ == "HumanMessage" and (m.content or "").strip():
                title = auto_title(m.content if isinstance(m.content, str) else "")
                cl.user_session.set("chat_session_title", title)
                break
    store.save(root, sid, brain.export_history(), brain.mode.value, title=title)


# ------------------------------------------------------------ session actions


@cl.action_callback("sess_open")
async def _on_sess_open(action: cl.Action) -> None:
    """Resume a saved chat session."""
    payload = action.payload or {}
    root = payload.get("root", "") or _workspace_root() or ""
    sid = payload.get("id", "")
    if not root or not sid:
        return
    try:
        ws = _open_workspace(root)
    except Exception as e:
        await cl.Message(content=f"Cannot open workspace: {e}").send()
        return
    sess = _sessions().get(str(ws.root), sid)
    if sess is None:
        await cl.Message(content="That session no longer exists — pick another.").send()
        await _show_session_picker(str(ws.root))
        return
    await _activate_session(ws, sess)


@cl.action_callback("sess_new")
async def _on_sess_new(action: cl.Action) -> None:
    """Start a fresh chat session in the workspace."""
    payload = action.payload or {}
    root = payload.get("root", "") or _workspace_root() or ""
    if not root:
        await cl.Message(content="Open a workspace first.").send()
        return
    try:
        ws = _open_workspace(root)
    except Exception as e:
        await cl.Message(content=f"Cannot open workspace: {e}").send()
        return
    sess = _sessions().create(str(ws.root), title="Untitled session", mode=_current_mode().value)
    log_code_event("session_new", root=str(ws.root), session=sess.id)
    await _activate_session(ws, sess)


@cl.action_callback("sess_delete")
async def _on_sess_delete(action: cl.Action) -> None:
    """Delete a saved chat session (history + token usage purged)."""
    payload = action.payload or {}
    root = payload.get("root", "") or _workspace_root() or ""
    sid = payload.get("id", "")
    if not root or not sid:
        return
    await _delete_session(root, sid)


async def _delete_session(root: str, sid: str) -> None:
    store = _sessions()
    sess = store.get(root, sid)
    if sess is None:
        await cl.Message(content="That session no longer exists.").send()
        return
    store.delete(root, sess.id)
    tracker = _tracker()
    if tracker:
        purged = tracker.purge_session(_tracker_sid_for(sess.id))
    else:
        # No live tracker (e.g. session was never opened here): purge via a
        # throwaway tracker bound to the same log file.
        cfg = LLMConfig.from_env()
        purged = TokenTracker(
            session_id=_tracker_sid_for(sess.id), provider=cfg.provider, model=cfg.model
        ).purge_session(_tracker_sid_for(sess.id))
    log_code_event("session_delete", root=root, session=sess.id, purged_turns=purged)
    await cl.Message(
        content=f"🗑 Deleted **{sess.title}** (`{sess.id[:8]}`) — chat history and {purged} usage turn(s) purged."
    ).send()
    if _chat_session_id() == sess.id:
        cl.user_session.set("brain", None)
        cl.user_session.set("tracker", None)
        cl.user_session.set("chat_session_id", None)
        cl.user_session.set("chat_session_title", None)
        await _show_session_picker(root)
    else:
        await _refresh_sidebar()
