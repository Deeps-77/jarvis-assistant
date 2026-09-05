"""Chat lifecycle: auth, start, settings, message loop, server entry."""
from __future__ import annotations

import os
from pathlib import Path

import chainlit as cl
import uvicorn

from code_assistant.modes import Mode

from .approval import (
    _approval_outcome_chip,
    _looks_like_json_tool_call,
    _request_approval,
)
from .bootstrap import UI_HOST, UI_PORT, _password, _username, log_code_event, logger
from .chrome import _push_settings, _refresh_sidebar, _send_welcome
from .commands import (
    _cmd_browse,
    _cmd_delete,
    _cmd_mode,
    _cmd_new,
    _cmd_open,
    _cmd_rename,
    _cmd_reset,
    _cmd_sessions,
    _cmd_status,
    _cmd_usage,
    _cmd_workspace,
)
from .sessions_ui import _ensure_session, _persist_session
from .state import _brain, _open_workspace
from .workspace_ui import _show_workspace_picker


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


def main() -> None:
    """Serve the code UI with uvicorn (ports: code :8500, chat :8000)."""

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
