"""Chat chrome: welcome card, sidebar status, typing-area settings panel."""
from __future__ import annotations

import chainlit as cl
from chainlit.input_widget import Select

from code_assistant.modes import Mode

from .state import _chat_session_title, _current_mode, _tracker, _workspace_root


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
