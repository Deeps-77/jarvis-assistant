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

from code_assistant.brain import CodeBrain  # noqa: E402
from code_assistant.modes import Mode  # noqa: E402
from code_assistant.tools import (  # noqa: E402
    get_file_info as _get_file_info,
    list_files as _list_files,
)
from code_assistant.workspace import (  # noqa: E402
    SavedWorkspace,
    Workspace,
    WorkspaceRegistry,
)
from llm_provider import LLMConfig  # noqa: E402
from token_usage import TokenTracker  # noqa: E402


# ----------------------------------------------------------------- helpers


UI_HOST = os.environ.get("CODE_UI_HOST", "0.0.0.0")
UI_PORT = int(os.environ.get("CODE_UI_PORT", "8500"))


def _password() -> str:
    return os.environ.get("CHAINLIT_CODE_PASSWORD", "")


def _username() -> str:
    return os.environ.get("CHAINLIT_CODE_USERNAME", "codeuser")


def _registry() -> WorkspaceRegistry:
    return WorkspaceRegistry(_PROJECT_ROOT / "code_workspaces.json")


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


# ----------------------------------------------------------- workspace load


def _open_workspace(root_str: str) -> Workspace:
    ws = Workspace(Path(root_str).expanduser())
    ws.ensure_sandbox()
    _registry().touch(ws)
    return ws


def _build_brain(workspace: Workspace) -> tuple[CodeBrain, TokenTracker]:
    cfg = LLMConfig.from_env()
    tracker = TokenTracker(
        session_id=_session_id(),
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


# --------------------------------------------------------------- UI helpers


async def _send_welcome() -> None:
    await cl.Message(
        content=(
            "**Jarvis Code Assistant** (Phase 1 — read-only Plan mode).\n\n"
            "Pick a workspace from the sidebar, then ask me to explore, "
            "summarise, or plan changes in your project.\n\n"
            "Commands:\n"
            "- `/workspace <path>` — switch to a different folder\n"
            "- `/mode plan` or `/mode build` — toggle mode (Build is read-only "
            "until Phase 2)\n"
            "- `/usage` — token usage summary\n"
            "- `/reset` — clear this chat's history"
        )
    ).send()


async def _refresh_sidebar() -> None:
    """Rebuild the sidebar element with current workspace + usage info."""
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
        f"- Est. cost: **${snap['cost_usd']:.4f}**"
    )

    element = cl.CustomElement(
        name="CodeAssistantSidebar",
        props={
            "workspace": root or "",
            "mode": mode.value,
            "workspace_block": workspace_block,
            "mode_block": mode_block,
            "usage_block": usage_block,
        },
    )
    # Sidebar updates are best-effort; ignore failures.
    try:
        cl.context.emitter.set_sidebar(element)
    except Exception:
        pass


# ------------------------------------------------------------ chat lifecycle


@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("mode", Mode.PLAN.value)
    cl.user_session.set("visible_text", [])

    # If CODE_DEFAULT_WORKSPACE is set, use it. Otherwise ask.
    default = os.environ.get("CODE_DEFAULT_WORKSPACE", "").strip()
    if default and Path(default).expanduser().is_dir():
        ws = _open_workspace(default)
        brain, tracker = _build_brain(ws)
        cl.user_session.set("workspace_root", str(ws.root))
        cl.user_session.set("brain", brain)
        cl.user_session.set("tracker", tracker)
        await cl.Message(
            content=f"Workspace opened: `{ws.root}` (mode: **plan**)"
        ).send()
    else:
        recent = _registry().list()
        if recent:
            listing = "\n".join(
                f"- `{e.root}`  _(last used {e.last_used})_" for e in recent[:5]
            )
            await cl.Message(
                content=(
                    "Pick a workspace by typing its absolute path. "
                    f"Recent workspaces:\n{listing}\n\n"
                    "Type `/workspace <path>` to switch."
                )
            ).send()
        else:
            await cl.Message(
                content=(
                    "Pick a workspace by typing its absolute path "
                    "(e.g. `D:\\Projects\\myrepo` or `/home/me/code/myrepo`).\n\n"
                    "Type `/workspace <path>` to switch."
                )
            ).send()

    await _send_welcome()
    await _refresh_sidebar()


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
    brain, tracker = _build_brain(ws)
    cl.user_session.set("workspace_root", str(ws.root))
    cl.user_session.set("brain", brain)
    cl.user_session.set("tracker", tracker)
    cl.user_session.set("visible_text", [])
    await cl.Message(
        content=f"Workspace switched to `{ws.root}` — brain reloaded."
    ).send()
    await _refresh_sidebar()


async def _cmd_mode(mode_str: str) -> None:
    new_mode = Mode.parse(mode_str)
    cl.user_session.set("mode", new_mode.value)
    brain = _brain()
    if brain:
        brain.set_mode(new_mode)
    note = ""
    if new_mode == Mode.BUILD:
        note = (
            "\n\n_Note: write tools (edit_file, run_command, …) land in "
            "Phase 2. Until then Build mode exposes the same read-only "
            "toolset as Plan._"
        )
    await cl.Message(
        content=f"Mode set to **{new_mode.value}**.{note}"
    ).send()
    await _refresh_sidebar()


async def _cmd_usage() -> None:
    tracker = _tracker()
    if not tracker:
        await cl.Message(content="No active session.").send()
        return
    snap = tracker.snapshot()
    history = tracker.history(10)
    history_block = ""
    if history:
        rows = "\n".join(
            f"- `{time.strftime('%H:%M:%S', time.localtime(h['ts']))}` "
            f"in {h['input_tokens']} / out {h['output_tokens']} "
            f"({int(h.get('duration_ms', 0))}ms)"
            for h in history
        )
        history_block = f"\n\n**Last {len(history)} turns**\n{rows}"
    await cl.Message(
        content=(
            f"**Token usage**\n"
            f"- Turns: **{snap['turns']}**\n"
            f"- Input: **{snap['input_tokens']:,}**\n"
            f"- Output: **{snap['output_tokens']:,}**\n"
            f"- Total: **{snap['total_tokens']:,}**\n"
            f"- Est. cost: **${snap['cost_usd']:.4f}**"
            f"{history_block}"
        )
    ).send()


async def _cmd_reset() -> None:
    brain = _brain()
    if brain:
        brain.reset_history()
    cl.user_session.set("visible_text", [])
    await cl.Message(content="Chat history cleared.").send()


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
        if cmd == "/usage":
            await _cmd_usage()
            return
        if cmd == "/reset":
            await _cmd_reset()
            return
        if cmd == "/help":
            await _send_welcome()
            return

    brain = _brain()
    if not brain:
        await cl.Message(
            content=(
                "No workspace is open. Use `/workspace <absolute path>` to "
                "pick a folder first."
            )
        ).send()
        return

    if not text:
        return

    # Stream the model reply into a single cl.Message.
    visible: list[str] = []
    cl.user_session.set("visible_text", visible)
    answer = cl.Message(content="")
    await answer.send()
    tool_chips: list[str] = []

    try:
        async for ev in brain.run(text):
            t = ev.type
            if t == "token":
                visible.append(ev.data["data"])
                await answer.stream_token(ev.data["data"])
            elif t == "retract":
                visible.clear()
                answer.content = ""
                await answer.update()
            elif t == "tool_start":
                name = ev.data.get("name", "tool")
                args = ev.data.get("args", {})
                arg_preview = ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:3])
                chip = f"\n\n> 🔧 **{name}**(`{arg_preview}`)\n"
                visible.append(chip)
                tool_chips.append(chip)
                await answer.stream_token(chip)
            elif t == "tool_end":
                name = ev.data.get("name", "tool")
                output_preview = ev.data.get("output", "")[:400].replace("\n", " ")
                visible.append(f"> ↳ _{name} result (truncated): {output_preview}_\n\n")
                tool_chips.append(f"> ↳ _{name} result (truncated)_")
                # Don't stream the raw tool output into the user-visible message
                # unless the brain ends without writing a final answer (handled below).
            elif t == "usage":
                # Sidebar updates after the turn; cheap to refresh now.
                pass
            elif t == "error":
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
        await _refresh_sidebar()
    except Exception as e:  # noqa: BLE001
        logger.exception("Code UI brain.run failed")
        await cl.Message(content=f"⚠️ Brain failed: {e}").send()


def _looks_like_json_tool_call(text: str) -> bool:
    """True if the final reply is just a serialised tool call (small-model artifact)."""
    s = (text or "").strip()
    if not s.startswith("{"):
        return False
    return '"name"' in s and ('"arguments"' in s or '"args"' in s)


# ----------------------------------------------------------------- main


if __name__ == "__main__":
    import threading

    import uvicorn

    # Chainlit exposes its FastAPI app via ``chainlit.server.app``.
    from chainlit.server import app as server_app

    host = UI_HOST
    port = UI_PORT

    if not _password():
        print(
            "[code_ui] CHAINLIT_CODE_PASSWORD is not set — the UI will be "
            "fail-closed (no one can log in). Set it in your .env to enable "
            "access."
        )

    print(f"[code_ui] starting on http://{host}:{port} (chat UI is on :8000)")
    uvicorn.run(server_app, host=host, port=port, log_level="info")
