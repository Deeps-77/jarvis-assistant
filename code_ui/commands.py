"""Slash commands for the code UI (/workspace, /mode, /sessions, ...)."""
from __future__ import annotations

from pathlib import Path

import chainlit as cl

from code_assistant.modes import Mode

from .bootstrap import log_code_event
from .chrome import _push_settings, _refresh_sidebar
from .sessions_ui import (
    _activate_session,
    _delete_session,
    _ensure_session,
    _persist_session,
    _session_desc,
    _session_label,
    _show_session_picker,
)
from .state import (
    _brain,
    _chat_session_id,
    _chat_session_title,
    _current_mode,
    _open_workspace,
    _sessions,
    _tracker,
    _workspace_root,
)
from .workspace_ui import _show_browse_message


# ---------------------------------------------------------------- commands


async def _cmd_workspace(path_str: str) -> None:
    path_str = path_str.strip().strip('"').strip("'")
    if not path_str:
        await cl.Message(content="Usage: `/workspace <absolute path>`").send()
        return
    try:
        ws = _open_workspace(path_str)
    except Exception as e:
        await cl.Message(content=f"Cannot open workspace: {e}").send()
        return
    cl.user_session.set("workspace_root", str(ws.root))
    cl.user_session.set("brain", None)
    cl.user_session.set("tracker", None)
    cl.user_session.set("chat_session_id", None)
    cl.user_session.set("chat_session_title", None)
    cl.user_session.set("visible_text", [])
    log_code_event("workspace_switch", root=str(ws.root), source="command")
    await cl.Message(
        content=f"Workspace switched to `{ws.root}` — pick a chat to continue."
    ).send()
    await _show_session_picker(str(ws.root))


async def _cmd_mode(mode_str: str) -> None:
    new_mode = Mode.parse(mode_str)
    cl.user_session.set("mode", new_mode.value)
    brain = _brain()
    if brain:
        brain.set_mode(new_mode)
    note = ""
    if new_mode == Mode.BUILD:
        note = (
            "\n\n_Note: write tools (edit_file, run_command, …) are "
            "approval-gated in the UI._"
        )
    log_code_event("mode", mode=new_mode.value)
    await cl.Message(
        content=f"Mode set to **{new_mode.value}**.{note}"
    ).send()
    await _persist_session()
    await _push_settings()
    await _refresh_sidebar()


async def _cmd_usage() -> None:
    tracker = _tracker()
    if not tracker:
        await cl.Message(content="No active session.").send()
        return
    card = tracker.render_usage_card(history_limit=100)
    await cl.Message(content=card).send()


async def _cmd_status() -> None:
    """One-shot workspace/session/mode/usage dump (no sidebar popup)."""
    root = _workspace_root()
    mode = _current_mode()
    title = _chat_session_title()
    sid = _chat_session_id()
    tracker = _tracker()
    snap = tracker.snapshot() if tracker else None
    lines = [
        "### Status",
        f"- Workspace: `{root or '—'}`",
        f"- Chat: **{title or '—'}**" + (f" `{sid[:8]}`" if sid else ""),
        f"- Mode: **{mode.value}**",
    ]
    if snap:
        lines.append(
            f"- Tokens: **{snap['total_tokens']:,}** "
            f"(in {snap['input_tokens']:,}, out {snap['output_tokens']:,}, {snap['turns']} turns)"
        )
        lines.append(
            f"- 💰 Avoided vs cloud: **${snap.get('saved_usd', 0.0):.4f}** (est.)"
        )
    await cl.Message(content="\n".join(lines)).send()


async def _cmd_reset() -> None:
    brain = _brain()
    if brain:
        brain.reset_history()
    cl.user_session.set("visible_text", [])
    # Persist the cleared state so a resume starts empty too.
    await _persist_session()
    await cl.Message(content="Chat history cleared.").send()


async def _cmd_sessions() -> None:
    """List all saved chats for the active workspace."""
    root = _workspace_root()
    if not root:
        await cl.Message(content="Open a workspace first.").send()
        return
    sessions = _sessions().list(root)
    if not sessions:
        await cl.Message(content=f"No saved chats in `{root}` yet. Type to start one.").send()
        return
    lines = [f"💬 **Chats in `{root}`** ({len(sessions)}):", ""]
    actions: list[cl.Action] = [
        cl.Action(
            name="sess_new",
            label="＋ New session",
            payload={"root": root},
            description="Start a fresh chat",
        )
    ]
    for sess in sessions[:20]:
        active = " ← active" if sess.id == _chat_session_id() else ""
        lines.append(f"- **{sess.title}** `{sess.id[:8]}` _{sess.mode}, {len(sess.messages)} msgs_{active}")
        if len(actions) < 9:
            actions.append(
                cl.Action(
                    name="sess_open",
                    label=_session_label(sess),
                    payload={"root": root, "id": sess.id},
                    description=_session_desc(sess),
                )
            )
    lines.append("")
    lines.append("`/open <id>` resume · `/new [title]` create · `/rename <title>` · `/delete [id]`")
    await cl.Message(content="\n".join(lines), actions=actions).send()


async def _cmd_new(title: str) -> None:
    ensured = await _ensure_session()
    if ensured is None:
        return
    ws, _ = ensured
    sess = _sessions().create(str(ws.root), title=title.strip() or "Untitled session", mode=_current_mode().value)
    log_code_event("session_new", root=str(ws.root), session=sess.id)
    await _activate_session(ws, sess)


async def _cmd_open(session_ref: str) -> None:
    session_ref = session_ref.strip()
    if not session_ref:
        await cl.Message(content="Usage: `/open <session-id>` (see `/sessions`).").send()
        return
    root = _workspace_root()
    if not root:
        await cl.Message(content="Open a workspace first.").send()
        return
    try:
        ws = _open_workspace(root)
    except Exception as e:
        await cl.Message(content=f"Cannot open workspace: {e}").send()
        return
    sess = _sessions().get(str(ws.root), session_ref)
    if sess is None:
        await cl.Message(content=f"No session matching `{session_ref}` in this workspace.").send()
        return
    await _activate_session(ws, sess)


async def _cmd_rename(title: str) -> None:
    title = title.strip()
    if not title:
        await cl.Message(content="Usage: `/rename <new title>`.").send()
        return
    root = _workspace_root()
    sid = _chat_session_id()
    if not root or not sid:
        await cl.Message(content="No active session.").send()
        return
    sess = _sessions().rename(root, sid, title)
    if sess is None:
        await cl.Message(content="That session no longer exists.").send()
        return
    cl.user_session.set("chat_session_title", sess.title)
    await cl.Message(content=f"Session renamed to **{sess.title}**.").send()
    await _refresh_sidebar()


async def _cmd_delete(session_ref: str) -> None:
    root = _workspace_root()
    if not root:
        await cl.Message(content="Open a workspace first.").send()
        return
    sid = session_ref.strip() or _chat_session_id() or ""
    if not sid:
        await cl.Message(content="Usage: `/delete [session-id]` (see `/sessions`).").send()
        return
    await _delete_session(root, sid)


async def _cmd_browse(path_str: str) -> None:
    """Browse the file tree at the given path (or current workspace)."""
    if path_str:
        path = Path(path_str.strip().strip('"').strip("'")).expanduser()
    else:
        root = _workspace_root()
        if not root:
            # Fall back to home so /browse works with no workspace open.
            path = Path.home()
        else:
            path = Path(root)

    if not path.exists() or not path.is_dir():
        await cl.Message(content=f"Path does not exist or is not a directory: `{path}`").send()
        return

    await _show_browse_message(path)
