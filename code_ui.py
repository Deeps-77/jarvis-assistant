"""Chainlit frontend for the Jarvis Code Assistant.

Runs as a SECOND web UI alongside the chat UI on ``app.py``. Listens on
``CODE_UI_PORT`` (default 8500) so the two never collide.

This is Phase 1 — read-only Plan mode is fully functional; the mode toggle
in the sidebar is wired up but Build mode exposes the same read-only
toolset until Phase 2 ships write_file / edit_file / run_command.

State model
-----------
``cl.user_session`` holds one entry per browser session:

    workspace_root: str  — absolute path the user picked
    mode:           str  — "plan" or "build"
    brain:          CodeBrain
    tracker:        TokenTracker
    visible_text:   list[str]  — accumulates token deltas until the
                                  next ``retract`` event

A ``WorkspaceRegistry`` JSON file (``code_workspaces.json``) remembers the
last few workspaces so the picker can offer them as recent options.

Auth
----
The code UI has its own password (``CHAINLIT_CODE_PASSWORD``). The auth
secret lives in ``.code_ui.secret`` so the code UI and chat UI can run on
the same machine without sharing cookies.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import sys
import time
import uuid
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path so ``code_assistant`` imports work.
_PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_PROJECT_ROOT))

# Chainlit reads several env vars at import time — set them before import.
os.environ.setdefault("LITERAL_TELEMETRY_OFF", "true")

logger = logging.getLogger(__name__)


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


_ensure_chainlit_dir(_PROJECT_ROOT)

_secret_file = _PROJECT_ROOT / ".code_ui.secret"
if not _secret_file.exists():
    _secret_file.write_text(secrets.token_hex(32), encoding="utf-8")
os.environ.setdefault(
    "CHAINLIT_AUTH_SECRET", _secret_file.read_text(encoding="utf-8").strip()
)

import chainlit as cl  # noqa: E402  (import after env setup)
from chainlit.input_widget import Select  # noqa: E402

import botlog as _botlog  # noqa: E402

from code_assistant.brain import ApprovalDecision, CodeBrain  # noqa: E402
from code_assistant.modes import Mode  # noqa: E402
from code_assistant.sessions import ChatSession, SessionStore, auto_title  # noqa: E402
from code_assistant.tools import (  # noqa: E402
    get_file_info as _get_file_info,
    list_files as _list_files,
)
from code_assistant.workspace import (  # noqa: E402
    SavedWorkspace,
    Workspace,
    WorkspaceRegistry,
)
from code_assistant.tools import set_workspace as _set_tools_workspace  # noqa: E402
from llm_provider import LLMConfig  # noqa: E402
from token_usage import TokenTracker  # noqa: E402


# ----------------------------------------------------------------- logging

# Wire the code UI into the shared botlog sinks (jarvis.log diagnostics +
# activity.log) so every session/turn/tool is captured alongside the chat UI.
_sa = getattr(_botlog, "setup_logging", None)
if callable(_sa) and os.environ.get("CODE_UI_DISABLE_LOGGING", "") != "1":
    try:
        _sa()
    except Exception:
        logger.exception("botlog.setup_logging failed")


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


# ----------------------------------------------------------------- helpers


UI_HOST = os.environ.get("CODE_UI_HOST", "0.0.0.0")
UI_PORT = int(os.environ.get("CODE_UI_PORT", "8500"))


def _password() -> str:
    return os.environ.get("CHAINLIT_CODE_PASSWORD", "")


def _username() -> str:
    return os.environ.get("CHAINLIT_CODE_USERNAME", "codeuser")


def _registry() -> WorkspaceRegistry:
    return WorkspaceRegistry(_PROJECT_ROOT / "code_workspaces.json")


def _sessions() -> SessionStore:
    return SessionStore(_PROJECT_ROOT / "code_sessions.json")


# --------------------------------------------------------------------- auth


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    expected = _password()
    if not expected:
        logger.warning(
            "CHAINLIT_CODE_PASSWORD not set - code UI access denied (fail-closed)."
        )
        return None
    if username == _username() and password == expected:
        return cl.User(identifier=username)
    return None


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


# --------------------------------------------------------------- file tree element


async def _build_file_tree(root: Path, max_depth: int = 3, current_depth: int = 0) -> list[dict]:
    """Build a file tree structure for the UI."""
    if current_depth > max_depth:
        return []
    
    items = []
    try:
        entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except (PermissionError, OSError):
        return []
    
    for entry in entries:
        # Skip hidden files
        if entry.name.startswith("."):
            continue
        
        item = {
            "name": entry.name,
            "path": str(entry),
            "is_dir": entry.is_dir(),
            "children": [],
        }
        
        if entry.is_dir() and current_depth < max_depth:
            item["children"] = await _build_file_tree(entry, max_depth, current_depth + 1)
        
        items.append(item)
    
    return items


async def _render_file_tree_markdown(root: Path, max_depth: int = 2, current_depth: int = 0, prefix: str = "") -> str:
    """Render a file tree as plain markdown (no custom elements)."""
    if current_depth > max_depth:
        return ""

    try:
        entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except FileNotFoundError:
        return "_Folder no longer exists._"
    except (PermissionError, OSError):
        return "_Cannot list this folder (permission denied)._"

    # Filter hidden, cap output so the message stays readable.
    visible = [e for e in entries if not e.name.startswith(".")]
    if not visible:
        return "_Empty folder._"

    lines: list[str] = []
    shown = visible[:40]
    for i, entry in enumerate(shown):
        is_last = i == len(shown) - 1
        connector = "└── " if is_last else "├── "
        if entry.is_dir():
            lines.append(f"{prefix}{connector}📁 **{entry.name}**")
            if current_depth < max_depth:
                sub_prefix = prefix + ("    " if is_last else "│   ")
                sub_tree = await _render_file_tree_markdown(entry, max_depth, current_depth + 1, sub_prefix)
                if sub_tree:
                    lines.append(sub_tree)
        else:
            lines.append(f"{prefix}{connector}📄 {entry.name}")
    if len(visible) > len(shown):
        lines.append(f"{prefix}… _+{len(visible) - len(shown)} more entries_")

    return "\n".join(lines)


def _render_file_tree_actions(root: Path, max_dirs: int = 8) -> list[cl.Action]:
    """Build a small, predictable action row for folder browsing.

    - ``filetree_use`` — open the current folder as workspace
    - ``filetree_up`` — go to the parent folder
    - ``filetree_nav`` — descend into one of up to ``max_dirs`` subfolders

    Keeping the callback names fixed avoids the "missing callback" bug
    where buttons were created with names that had no handler.
    """
    actions: list[cl.Action] = [
        cl.Action(
            name="filetree_use",
            label="✅ Use this folder",
            payload={"path": str(root)},
            description=f"Open {root} as workspace",
        ),
    ]
    parent = root.parent
    if str(parent) != str(root):
        actions.append(
            cl.Action(
                name="filetree_up",
                label="⬆️ Parent",
                payload={"path": str(root)},
                description="Go to parent folder",
            )
        )
    try:
        entries = sorted(root.iterdir(), key=lambda p: p.name.lower())
    except (PermissionError, OSError):
        return actions
    dirs = [e for e in entries if e.is_dir() and not e.name.startswith(".")][:max_dirs]
    for entry in dirs:
        # Truncate long names so button labels stay clickable.
        label = entry.name if len(entry.name) <= 24 else entry.name[:23] + "…"
        actions.append(
            cl.Action(
                name="filetree_nav",
                label=f"📁 {label}",
                payload={"path": str(entry)},
                description=str(entry),
            )
        )
    return actions


async def _show_browse_message(path: Path) -> None:
    """Send one browse message for ``path`` with tree + nav actions."""
    tree_md = await _render_file_tree_markdown(path, max_depth=1)
    actions = _render_file_tree_actions(path)
    content = (
        f"📁 **Browsing:** `{path}`\n\n"
        f"```text\n{tree_md}\n```\n\n"
        f"Click a 📁 button to descend, ⬆️ to go up, or ✅ to open this folder as workspace."
    )
    await cl.Message(content=content, actions=actions).send()


async def _open_workspace_from_path(path: Path, source: str = "browse") -> None:
    """Open ``path`` as workspace, then offer its chat sessions."""
    if not path.is_dir():
        await cl.Message(content=f"⚠️ `{path}` is not a directory.").send()
        return
    try:
        ws = _open_workspace(str(path))
    except Exception as e:
        await cl.Message(content=f"Cannot open workspace: {e}").send()
        return
    cl.user_session.set("workspace_root", str(ws.root))
    cl.user_session.set("brain", None)
    cl.user_session.set("tracker", None)
    cl.user_session.set("chat_session_id", None)
    cl.user_session.set("chat_session_title", None)
    cl.user_session.set("visible_text", [])
    log_code_event("workspace_open", root=str(ws.root), source=source)
    await cl.Message(
        content=f"✅ Workspace set to `{ws.root}` (mode: **{_current_mode().value}**)"
    ).send()
    await _show_session_picker(str(ws.root))


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


# --------------------------------------------------------------- UI helpers


async def _send_welcome() -> None:
    browse_action = cl.Action(
        name="ws_browse",
        label="📂 Browse folders…",
        payload={"action": "browse"},
        description="Open the OS folder dialog to pick a workspace",
    )
    await cl.Message(
        content=(
            "**Jarvis Code Assistant** (Phase 2 — read/write with approval).\n\n"
            "Pick a workspace, then ask me to explore, plan, or make changes "
            "in your project.\n\n"
            "Commands:\n"
            "- `/workspace <path>` — switch to a different folder\n"
            "- `/browse` — browse the file tree\n"
            "- `/mode plan` or `/mode build` — toggle mode\n"
            "- `/sessions` — list chats in this workspace\n"
            "- `/new [title]` — start a new chat\n"
            "- `/open <id>` — resume a chat\n"
            "- `/rename <title>` — rename the active chat\n"
            "- `/delete [id]` — delete a chat (history + usage purged)\n"
            "- `/usage` — token usage summary\n"
            "- `/status` — workspace/chat/mode at a glance (no popup)\n"
            "- `/reset` — clear this chat's history"
        ),
        actions=[browse_action],
    ).send()


async def _refresh_sidebar() -> None:
    """Rebuild the sidebar with current workspace + usage info.

    Uses only built-in ``cl.Text`` elements — ``CustomElement`` names
    require a matching frontend JS component, otherwise the UI shows
    "Not Found: File not found".
    """
    root = _workspace_root()
    mode = _current_mode()
    tracker = _tracker()
    snap = tracker.snapshot() if tracker else {
        "turns": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
    }

    workspace_block = (
        f"**Workspace**\n```\n{root}\n```\n"
        if root
        else "**Workspace**: _not set — use `/workspace <path>` or the picker_"
    )
    session_title = _chat_session_title()
    session_block = (
        f"**Chat**: `{session_title}`\n\n"
        if session_title
        else "**Chat**: _none — pick or start one with `/sessions`_\n\n"
    )
    mode_block = (
        f"**Mode**: `{mode.value}`  "
        + (
            "(read-only)" if mode == Mode.PLAN else "(Phase 2 unlocks writes)"
        )
    )
    usage_block = (
        f"**Tokens (this session)**\n"
        f"- Turns: **{snap['turns']}**\n"
        f"- Input: **{snap['input_tokens']:,}**\n"
        f"- Output: **{snap['output_tokens']:,}**\n"
        f"- Total: **{snap['total_tokens']:,}**\n"
        f"- Est. cost: **${snap['cost_usd']:.4f}**\n"
        f"- 💰 Avoided vs cloud: **${snap.get('saved_usd', 0.0):.4f}** (est.)"
    )

    content = f"{workspace_block}\n\n{session_block}{mode_block}\n\n{usage_block}"
    # Sidebar updates are best-effort; ignore failures (e.g. no active context).
    # NOTE: every set_elements call pops the sidebar open, so callers must
    # only refresh on workspace/session/mode/open events — never per turn.
    try:
        await cl.ElementSidebar.set_title("Code Assistant")
        await cl.ElementSidebar.set_elements(
            [cl.Text(name="Status", content=content, display="side")],
            key="code-status",
        )
    except Exception:
        pass


async def _push_settings() -> None:
    """Sync the typing-area settings panel (gear icon) with current mode.

    Silent panel update — unlike the sidebar, this never pops open.
    Best-effort; ignored outside a live Chainlit context (e.g. unit tests).
    """
    try:
        mode = _current_mode()
        await cl.ChatSettings(
            inputs=[
                Select(
                    id="mode",
                    label="Mode",
                    values=["plan", "build"],
                    initial_index=0 if mode == Mode.PLAN else 1,
                    description="plan = read-only · build = writes need approval",
                ),
            ]
        ).send()
    except Exception:
        pass


@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    """Gear-icon panel changed (e.g. Mode select) — apply like /mode."""
    raw = (settings or {}).get("mode")
    if raw is None:
        return
    new_mode = Mode.parse(str(raw))
    if new_mode == _current_mode():
        return
    await _cmd_mode(new_mode.value)


# ------------------------------------------------------------ chat lifecycle


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("mode", Mode.PLAN.value)
    cl.user_session.set("visible_text", [])

    # If CODE_DEFAULT_WORKSPACE is set, use it. Otherwise ask.
    default = os.environ.get("CODE_DEFAULT_WORKSPACE", "").strip()
    if default and Path(default).expanduser().is_dir():
        ws = _open_workspace(default)
        cl.user_session.set("workspace_root", str(ws.root))
        await cl.Message(
            content=f"Workspace opened: `{ws.root}` (mode: **plan**)"
        ).send()
        await _show_session_picker(str(ws.root))
    else:
        # Show file tree picker for workspace selection
        await _show_workspace_picker()

    await _send_welcome()
    await _push_settings()
    await _refresh_sidebar()


async def _show_workspace_picker(current_path: Path | None = None) -> None:
    """Show an interactive file tree picker for workspace selection."""
    # Start from home directory or current path
    root = current_path if current_path else Path.home()

    # Primary action: native OS folder dialog.
    browse_action = cl.Action(
        name="ws_browse",
        label="📂 Browse folders…",
        payload={"action": "browse"},
        description="Open the OS folder dialog",
    )
    # Secondary: quick recent workspaces.
    recent_actions = []
    for e in _registry().list()[:3]:
        recent_actions.append(cl.Action(
            name="ws_recent",
            label=f"🕘 {e.name or e.root}",
            payload={"action": "open", "path": e.root},
            description=e.root,
        ))

    content = (
        f"📁 **Select a workspace**\n\n"
        "Click **📂 Browse folders…** to open the OS folder dialog, "
        "choose a folder, and start working.\n\n"
        "Or use `/workspace <path>` to type a path manually."
    )
    await cl.Message(content=content, actions=[browse_action]).send()

    if recent_actions:
        await cl.Message(
            content="**Recent workspaces** (click to reopen):",
            actions=recent_actions,
        ).send()
    else:
        await cl.Message(
            content=(
                "Tip: You can also type `/workspace <path>` manually "
                "(e.g. `D:\\Projects\\myrepo` or `/home/me/code/myrepo`)."
            )
        ).send()


# ------------------------------------------------------------ native folder picker


def _pick_folder_native_sync(title: str = "Select workspace folder", initial_dir: str | None = None) -> str | None:
    """Open the OS-native "Browse for Folder" dialog (Windows tkinter).

    Runs synchronously (blocks the calling thread); callers should offload
    to a thread via ``asyncio.to_thread``. Returns the chosen path or
    ``None`` if cancelled. Fails gracefully if no display/tkinter.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        logger.warning("tkinter unavailable for folder picker: %s", e)
        return None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)  # above browser window
        chosen = filedialog.askdirectory(
            title=title,
            initialdir=initial_dir or str(Path.home()),
            parent=root,
        )
        root.destroy()
        return chosen or None
    except Exception as e:
        logger.exception("folder picker failed")
        return None


async def _pick_folder_native(title: str = "Select workspace folder") -> str | None:
    """Async wrapper: opens the native dialog off the event loop."""
    root = _workspace_root()
    return await asyncio.to_thread(_pick_folder_native_sync, title, root)


def _workspace_select_actions() -> list:
    """Action buttons shown on the welcome/picker message."""
    return [
        cl.Action(
            name="ws_browse",
            label="📂 Browse folders…",
            payload={"action": "browse"},
            description="Open the OS folder dialog",
        ),
    ]


# ------------------------------------------------------------ file tree actions


@cl.action_callback("ws_browse")
async def _on_ws_browse(action: cl.Action) -> None:
    """Open the native OS folder dialog and open the chosen workspace."""
    await cl.Message(content="📂 Opening the folder picker…").send()
    chosen = await _pick_folder_native()
    if not chosen:
        await cl.Message(content="Picker cancelled — no folder selected.").send()
        return
    try:
        ws = _open_workspace(chosen)
    except Exception as e:
        await cl.Message(content=f"Cannot open workspace: {e}").send()
        return
    cl.user_session.set("workspace_root", str(ws.root))
    cl.user_session.set("brain", None)
    cl.user_session.set("tracker", None)
    cl.user_session.set("chat_session_id", None)
    cl.user_session.set("chat_session_title", None)
    cl.user_session.set("visible_text", [])
    log_code_event("workspace_open", root=ws.root, source="native_picker")
    await cl.Message(content=f"✅ Workspace set to `{ws.root}` (mode: **{_current_mode().value}**)").send()
    await _show_session_picker(str(ws.root))


@cl.action_callback("ws_recent")
async def _on_ws_recent(action: cl.Action) -> None:
    """Reopen a workspace from the recent list."""
    path_str = (action.payload or {}).get("path", "")
    if not path_str:
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
    log_code_event("workspace_open", root=ws.root, source="recent")
    await cl.Message(content=f"✅ Workspace set to `{ws.root}` (mode: **{_current_mode().value}**)").send()
    await _show_session_picker(str(ws.root))


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


# ---------------------------------------------------------------- file tree actions


@cl.action_callback("filetree_nav")
async def _on_filetree_nav(action: cl.Action) -> None:
    """Descend into a subfolder while browsing."""
    path_str = (action.payload or {}).get("path", "")
    if not path_str:
        return
    path = Path(path_str)
    if not path.exists() or not path.is_dir():
        await cl.Message(content=f"Path does not exist or is not a directory: `{path}`").send()
        return
    await _show_browse_message(path)


@cl.action_callback("filetree_up")
async def _on_filetree_up(action: cl.Action) -> None:
    """Go to the parent folder while browsing."""
    path_str = (action.payload or {}).get("path", "")
    if not path_str:
        return
    path = Path(path_str).parent
    if not path.exists() or not path.is_dir():
        await cl.Message(content=f"Path does not exist or is not a directory: `{path}`").send()
        return
    await _show_browse_message(path)


@cl.action_callback("filetree_use")
async def _on_filetree_use(action: cl.Action) -> None:
    """Open the browsed folder as the active workspace."""
    path_str = (action.payload or {}).get("path", "")
    if not path_str:
        return
    await _open_workspace_from_path(Path(path_str), source="browse")


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


# --------------------------------------------------------------- message


@cl.on_message
async def on_message(message: cl.Message):
    text = (message.content or "").strip()

    # Slash commands short-circuit the brain.
    if text.startswith("/"):
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""
        if cmd == "/workspace":
            await _cmd_workspace(arg)
            return
        if cmd == "/mode":
            await _cmd_mode(arg)
            return
        if cmd == "/browse":
            await _cmd_browse(arg)
            return
        if cmd == "/usage":
            await _cmd_usage()
            return
        if cmd == "/status":
            await _cmd_status()
            return
        if cmd == "/reset":
            await _cmd_reset()
            return
        if cmd == "/sessions":
            await _cmd_sessions()
            return
        if cmd == "/new":
            await _cmd_new(arg)
            return
        if cmd == "/open":
            await _cmd_open(arg)
            return
        if cmd == "/rename":
            await _cmd_rename(arg)
            return
        if cmd == "/delete":
            await _cmd_delete(arg)
            return
        if cmd == "/help":
            await _send_welcome()
            return

    ensured = await _ensure_session(first_text=text)
    if ensured is None:
        await cl.Message(
            content=(
                "No workspace is open. Use `/workspace <absolute path>` to "
                "pick a folder first."
            )
        ).send()
        return
    _, _active = ensured
    brain = _brain()
    if not brain:
        await cl.Message(content="No active session — pick one from the picker above.").send()
        return

    if not text:
        return

    # Stream the model reply into a single cl.Message.
    visible: list[str] = []
    cl.user_session.set("visible_text", visible)
    answer = cl.Message(content="")
    await answer.send()
    tool_chips: list[str] = []
    thinking_parts: list[str] = []
    thinking_msg: cl.Message | None = None
    thinking_shown = 0

    try:
        async for ev in brain.run(text):
            t = ev.type
            if t == "token":
                visible.append(ev.data["data"])
                await answer.stream_token(ev.data["data"])
            elif t == "thinking":
                # Live reasoning trace (thinking models only). One message,
                # throttled updates so the chat feels alive during long gens.
                thinking_parts.append(ev.data.get("data", ""))
                if thinking_msg is None:
                    thinking_msg = cl.Message(content="🤔 Thinking…")
                    await thinking_msg.send()
                total = sum(len(p) for p in thinking_parts)
                if total - thinking_shown >= 400:
                    thinking_shown = total
                    thinking_msg.content = (
                        "🤔 Thinking…\n\n<details><summary>reasoning</summary>\n\n"
                        + "".join(thinking_parts)[-3000:]
                        + "\n\n</details>"
                    )
                    await thinking_msg.update()
            elif t == "retract":
                visible.clear()
                answer.content = ""
                await answer.update()
            elif t == "tool_start":
                name = ev.data.get("name", "tool")
                args = ev.data.get("args", {})
                log_code_event("tool_start", name=name, args=args)
                arg_preview = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:3])
                chip = f"\n\n> 🔧 **{name}**(`{arg_preview}`)\n"
                visible.append(chip)
                tool_chips.append(chip)
                await answer.stream_token(chip)
            elif t == "tool_end":
                name = ev.data.get("name", "tool")
                output_preview = ev.data.get("output", "")[:400].replace("\n", " ")
                rejected = ev.data.get("rejected", False)
                log_code_event("tool_end", name=name, rejected=rejected, chars=len(output_preview))
                if rejected:
                    chip = f"> ↳ ❌ _{name} was rejected by the user._\n\n"
                else:
                    chip = f"> ↳ _{name} result (truncated): {output_preview}_\n\n"
                visible.append(chip)
                tool_chips.append(chip)
                await answer.stream_token(chip)
            elif t == "approval_required":
                # Show the approval card and block until the user decides.
                decision = await _request_approval(answer, ev.data)
                log_code_event("approval", decision=decision.decision, name=ev.data.get("name"))
                await brain.submit_approval(decision)
                chip = _approval_outcome_chip(decision)
                if chip:
                    visible.append(chip)
                    tool_chips.append(chip)
                    await answer.stream_token(chip)
            elif t == "usage":
                # Sidebar updates after the turn; cheap to refresh now.
                pass
            elif t == "error":
                log_code_event("error", message=ev.data.get("message", "error"))
                await cl.Message(content=f"⚠️ {ev.data.get('message', 'error')}").send()
                return
            elif t == "done":
                reply = ev.data.get("reply", "")
                # If the brain ended without writing prose (small models
                # sometimes stop right after a tool call), surface the
                # tool chips so the user sees what happened.
                if not reply.strip() or _looks_like_json_tool_call(reply):
                    if tool_chips:
                        answer.content = (
                            "_(The model returned only a tool call. The "
                            "tool outputs above show what was inspected.)_"
                        )
                    else:
                        answer.content = "(no response)"
                else:
                    answer.content = reply
                await answer.update()
        await _persist_session()
        # NOTE: no _refresh_sidebar() here — set_elements pops the sidebar
        # open, which is annoying every turn. Sidebar refreshes on
        # workspace/session/mode/open events; use /status or /usage anytime.
    except Exception as e:  # noqa: BLE001
        logger.exception("Code UI brain.run failed")
        log_code_event("error", message=str(e)[:300])
        await cl.Message(content=f"⚠️ Brain failed: {e}").send()


def _looks_like_json_tool_call(text: str) -> bool:
    """True if the final reply is just a serialised tool call (small-model artifact)."""
    s = (text or "").strip()
    if not s.startswith("{"):
        return False
    return '"name"' in s and ('"arguments"' in s or '"args"' in s)


# ------------------------------------------------------------ approval UX


def _format_args(args: dict[str, Any]) -> str:
    """Render tool args as a compact, readable block."""
    import json
    try:
        return json.dumps(args, indent=2, ensure_ascii=False)[:2000]
    except (TypeError, ValueError):
        return repr(args)[:2000]


async def _request_approval(answer: cl.Message, payload: dict[str, Any]) -> ApprovalDecision:
    """Show the approval card and block until the user responds.

    Uses Chainlit's ``AskUserMessage`` so the approval flow blocks the
    current handler; the brain's ``_approval_event`` stays set until the
    response comes back.
    """
    name = payload.get("name", "?")
    args = payload.get("args", {})
    args_block = _format_args(args)
    card = (
        f"### ⚠️ Approval required\n\n"
        f"**Tool:** `{name}`\n\n"
        f"**Arguments:**\n```json\n{args_block}\n```\n\n"
        "Reply with one of:\n"
        "- `approve` — run the tool with these arguments\n"
        "- `reject` — decline (model will adapt)\n"
        "- `reject <reason>` — decline with an explanation for the model\n"
        "- `edit` — approve with edited args (paste a JSON object matching the schema)\n"
    )
    # Stream the card into the current answer so the user sees it inline.
    answer.content = (answer.content or "") + "\n\n" + card
    await answer.update()

    response = await cl.AskUserMessage(
        content="Approve this tool call?",
        timeout=180,
    ).send()

    if response is None:
        return ApprovalDecision(decision="reject", reason="timeout")

    text = (response or "").strip()
    low = text.lower()
    if low in ("approve", "yes", "y", "ok"):
        return ApprovalDecision(decision="approve")
    if low.startswith("edit"):
        rest = text[4:].strip()
        # The user might paste JSON. Try to parse.
        import json
        try:
            edited = json.loads(rest) if rest else args
        except json.JSONDecodeError:
            await cl.Message(
                content="Couldn't parse edit JSON. Rejecting with reason."
            ).send()
            return ApprovalDecision(
                decision="reject",
                reason=f"edit was unparseable: {rest[:120]}",
            )
        return ApprovalDecision(decision="edit", args=edited)
    if low.startswith("reject"):
        reason = text[len("reject"):].strip() or "user declined"
        return ApprovalDecision(decision="reject", reason=reason)
    # Default: treat unknown reply as rejection with explanation.
    return ApprovalDecision(decision="reject", reason=f"unrecognised reply: {text[:120]}")


def _approval_outcome_chip(decision: ApprovalDecision) -> str:
    """Render a small status chip so the conversation log shows the decision."""
    if decision.decision == "approve":
        return "> ✅ _approved_\n\n"
    if decision.decision == "edit":
        return "> ✏️ _approved with edited args_\n\n"
    return f"> ❌ _rejected ({decision.reason[:80]})_\n\n"


# ----------------------------------------------------------------- main


if __name__ == "__main__":
    import threading

    import uvicorn

    # Chainlit exposes its FastAPI app via ``chainlit.server.app``.
    from chainlit.config import config as _cl_config
    from chainlit.server import app as server_app

    host = UI_HOST
    port = UI_PORT

    # Chainlit's own ``config.run`` default is 127.0.0.1:8000 and its
    # ``lifespan`` auto-opens the browser at that address (headless=False).
    # Without this override the code UI would launch http://localhost:8000
    # (the chat UI's port) even though uvicorn serves this app on :8500.
    _cl_config.run.host = "127.0.0.1" if host == "0.0.0.0" else host
    _cl_config.run.port = port

    if not _password():
        print(
            "[code_ui] CHAINLIT_CODE_PASSWORD is not set — the UI will be "
            "fail-closed (no one can log in). Set it in your .env to enable "
            "access."
        )

    print(f"[code_ui] starting on http://localhost:{port} (chat UI is on :{os.environ.get('CHAINLIT_PORT', '8000')})")
    uvicorn.run(server_app, host=host, port=port, log_level="info")
