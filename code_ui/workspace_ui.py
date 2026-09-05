"""Workspace picking + file-tree browsing UI (picker, native dialog, actions)."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import chainlit as cl

from .bootstrap import log_code_event
from .sessions_ui import _show_session_picker
from .state import _current_mode, _open_workspace, _registry, _workspace_root

logger = logging.getLogger(__name__)


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
