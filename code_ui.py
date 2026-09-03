"""Jarvis Code Assistant — Chainlit UI (Phase 1).

Run with:
    python code_ui.py

Environment variables (in addition to existing .env):
    CHAINLIT_CODE_USERNAME   — UI login username  (default: webuser)
    CHAINLIT_CODE_PASSWORD   — UI login password  (default: same as CHAINLIT_PASSWORD)
    CODE_UI_PORT             — override port if not using --port flag
    CODE_LLM_PROVIDER        — ollama | openai  (default: ollama)
    CODE_LLM_MODEL           — model name       (default: qwen2.5-coder:3b-instruct-q4_K_M)
    CODE_DEFAULT_WORKSPACE   — optional: pre-open this folder on every login

Features (Phase 1 — Plan mode only):
    ✓ Password authentication (separate from the chat UI)
    ✓ Workspace picker: text input → JSON registry → remembered across sessions
    ✓ Sidebar: workspace info, mode badge, live token card
    ✓ Streaming markdown responses
    ✓ Tool-call chips: 🔧 tool_name(args…) shown inline
    ✓ Token usage updated after every turn
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
import os
import sys
import uuid
from pathlib import Path

os.environ["LITERAL_TELEMETRY_OFF"] = "true"
sys.path.insert(0, str(Path(__file__).parent))

import chainlit as cl
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

_secret_file = Path(__file__).parent / ".chainlit.secret"
if not _secret_file.exists():
    _secret_file.write_text(secrets.token_hex(32), encoding="utf-8")
os.environ.setdefault(
    "CHAINLIT_AUTH_SECRET", _secret_file.read_text(encoding="utf-8").strip()
)

load_dotenv()

# ─────────────────────────── local imports ───────────────────────────────────

from llm_provider import LLMConfig
from token_usage import TokenTracker
from code_assistant.workspace import Workspace, WorkspaceRegistry
from code_assistant.brain import CodeBrain
from code_assistant.modes import Mode

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ────────────────────────────── config ───────────────────────────────────────

_USERNAME = os.environ.get("CHAINLIT_CODE_USERNAME", os.environ.get("CHAINLIT_USERNAME", "webuser"))
_PASSWORD = os.environ.get("CHAINLIT_CODE_PASSWORD", os.environ.get("CHAINLIT_PASSWORD", ""))
_DEFAULT_WS = os.environ.get("CODE_DEFAULT_WORKSPACE", "").strip()

_REGISTRY = WorkspaceRegistry()

# ─────────────────────────── auth ────────────────────────────────────────────


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> cl.User | None:
    """Simple single-user password gate."""
    if not _PASSWORD:
        # Fail-closed: if no password configured, deny everyone.
        logger.warning("CHAINLIT_CODE_PASSWORD not set — all logins rejected.")
        return None
    if username == _USERNAME and password == _PASSWORD:
        return cl.User(identifier=username, metadata={"role": "user"})
    return None


# ─────────────────────────── session helpers ──────────────────────────────────

def _get_ws() -> Workspace | None:
    return cl.user_session.get("workspace")


def _get_brain() -> CodeBrain | None:
    return cl.user_session.get("brain")


def _get_tracker() -> TokenTracker | None:
    return cl.user_session.get("tracker")


def _get_mode() -> Mode:
    return cl.user_session.get("mode", Mode.PLAN)


def _get_history() -> list:
    return cl.user_session.get("history", [])


# ─────────────────────────── UI helpers ──────────────────────────────────────


def _token_card_text(tracker: TokenTracker | None) -> str:
    if not tracker:
        return "No usage yet."
    snap = tracker.snapshot()
    cost = snap["cost_usd"]
    cost_str = f"${cost:.6f}" if cost > 0 else "free (local)"
    return (
        f"**Model:** `{snap['model']}`\n"
        f"**Turns:** {snap['turns']}\n"
        f"**Tokens in:** {snap['input_tokens']:,}\n"
        f"**Tokens out:** {snap['output_tokens']:,}\n"
        f"**Total:** {snap['total_tokens']:,}\n"
        f"**Cost:** {cost_str}"
    )


async def _update_sidebar(ws: Workspace | None, mode: Mode, tracker: TokenTracker | None) -> None:
    """Refresh all sidebar elements."""
    elements: list[cl.Element] = []

    # ── Workspace card ──
    if ws:
        ws_text = (
            f"**📁 {ws.name}**\n\n"
            f"`{ws.root}`\n\n"
            f"_Type `/switch` to change workspace_"
        )
    else:
        ws_text = "_No workspace selected._\n\nType the full path of a folder to get started."

    elements.append(
        cl.Text(
            name="workspace_info",
            content=ws_text,
            display="side",
        )
    )

    # ── Mode badge ──
    mode_text = (
        f"**Mode:** {mode.label()}\n\n"
        f"_{mode.description()}_\n\n"
        f"_(Build mode coming in Phase 2)_"
    )
    elements.append(
        cl.Text(
            name="mode_info",
            content=mode_text,
            display="side",
        )
    )

    # ── Token card ──
    elements.append(
        cl.Text(
            name="token_usage",
            content=f"### 📊 Token Usage\n\n{_token_card_text(tracker)}",
            display="side",
        )
    )

    await cl.ElementBag(elements).send()


async def _open_workspace(path_str: str) -> Workspace | None:
    """Validate + open a workspace, send a file tree preview, return the Workspace."""
    path_str = path_str.strip()
    if not path_str:
        await cl.Message(content="⚠️ Please enter a valid directory path.").send()
        return None
    try:
        ws = Workspace(root=Path(path_str), name=Path(path_str).name)
        _REGISTRY.add(ws)
        return ws
    except (ValueError, OSError) as e:
        await cl.Message(content=f"❌ Could not open workspace: **{e}**").send()
        return None


async def _send_file_tree(ws: Workspace) -> None:
    """Send a quick top-level file tree as a collapsible message."""
    from code_assistant.tools import make_read_tools
    from code_assistant.modes import filter_tools, Mode

    tools = make_read_tools(ws)
    list_tool = next((t for t in tools if t.name == "list_files"), None)
    if not list_tool:
        return

    tree = list_tool.invoke({"path": "", "pattern": "", "max_depth": 2})
    await cl.Message(
        content=f"**Workspace opened: `{ws.name}`**\n\n```\n{tree}\n```\n\n_Ready! Ask me anything about this codebase._",
        author="Jarvis Code",
    ).send()


def _make_brain(ws: Workspace, tracker: TokenTracker) -> CodeBrain:
    cfg = LLMConfig.from_env()
    return CodeBrain(workspace=ws, llm_config=cfg, tracker=tracker)


def _make_tracker(session_id: str) -> TokenTracker:
    cfg = LLMConfig.from_env()
    return TokenTracker(
        session_id=session_id,
        provider=cfg.provider,
        model=cfg.model,
        label="plan",
    )


# ─────────────────────────── lifecycle ───────────────────────────────────────


@cl.on_chat_start
async def on_chat_start() -> None:
    """Initialise a new code-assistant session."""
    session_id = cl.user_session.get("id") or str(uuid.uuid4())
    tracker = _make_tracker(session_id)
    cl.user_session.set("tracker", tracker)
    cl.user_session.set("mode", Mode.PLAN)
    cl.user_session.set("history", [])

    # ── Try default workspace from env ──
    ws: Workspace | None = None
    if _DEFAULT_WS:
        ws = await _open_workspace(_DEFAULT_WS)

    # ── Show saved workspaces if any ──
    saved = _REGISTRY.list_all()
    if not ws and saved:
        names = ", ".join(f"`{w['name']}`" for w in saved[:5])
        intro = (
            "👋 **Welcome to Jarvis Code Assistant!**\n\n"
            f"You have saved workspaces: {names}\n\n"
            "Type a saved workspace name or paste a full directory path to open one.\n"
            "_(Or type `/switch <path>` at any time to switch workspace)_"
        )
        await cl.Message(content=intro, author="Jarvis Code").send()
    elif not ws:
        intro = (
            "👋 **Welcome to Jarvis Code Assistant!**\n\n"
            "To get started, type the **full path** to the folder you want to work in.\n\n"
            "Example:\n"
            "```\nD:\\Projects\\my-app\n```\n"
            "_I'll read that folder and you can ask me anything about the code._"
        )
        await cl.Message(content=intro, author="Jarvis Code").send()

    if ws:
        cl.user_session.set("workspace", ws)
        brain = _make_brain(ws, tracker)
        cl.user_session.set("brain", brain)
        await _send_file_tree(ws)

    await _update_sidebar(ws, Mode.PLAN, tracker)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Route each user message."""
    content = message.content.strip()

    # ── Slash commands ──
    if content.startswith("/switch"):
        path = content[len("/switch"):].strip()
        if not path:
            await cl.Message(content="Usage: `/switch <path>`").send()
            return
        await _handle_workspace_input(path)
        return

    if content.startswith("/usage"):
        await _handle_usage_command()
        return

    if content.startswith("/clear"):
        cl.user_session.set("history", [])
        await cl.Message(content="✅ Conversation history cleared.").send()
        return

    # ── Workspace picker (if no workspace set yet) ──
    ws = _get_ws()
    if not ws:
        # Try interpreting the message as a path or workspace name
        await _handle_workspace_input(content)
        return

    # ── Normal chat → brain ──
    await _handle_chat(content)


async def _handle_workspace_input(raw: str) -> None:
    """Accept a path or saved workspace name, open it, update session."""
    # Check saved names first
    saved_ws = _REGISTRY.get(raw)
    if saved_ws:
        path_str = str(saved_ws.root)
    else:
        path_str = raw

    ws = await _open_workspace(path_str)
    if not ws:
        return

    tracker = _get_tracker()
    cl.user_session.set("workspace", ws)
    brain = _make_brain(ws, tracker)
    cl.user_session.set("brain", brain)
    await _send_file_tree(ws)
    await _update_sidebar(ws, _get_mode(), tracker)


async def _handle_usage_command() -> None:
    tracker = _get_tracker()
    if not tracker:
        await cl.Message(content="No usage data for this session.").send()
        return

    history = tracker.history(limit=20)
    if not history:
        await cl.Message(content="No LLM calls recorded yet.").send()
        return

    rows = ["| Turn | Model | In | Out | Cost |", "|------|-------|----|-----|------|"]
    for i, turn in enumerate(history, 1):
        cost = f"${turn['cost_usd']:.6f}" if turn["cost_usd"] > 0 else "free"
        rows.append(
            f"| {i} | `{turn['model']}` | {turn['input_tokens']:,} | {turn['output_tokens']:,} | {cost} |"
        )

    snap = tracker.snapshot()
    total_cost = f"${snap['cost_usd']:.6f}" if snap["cost_usd"] > 0 else "free (local)"
    summary = (
        f"\n**Session totals** — {snap['turns']} turns, "
        f"{snap['total_tokens']:,} tokens, {total_cost}"
    )
    await cl.Message(content="\n".join(rows) + summary).send()


async def _handle_chat(content: str) -> None:
    """Run the brain and stream the response."""
    ws = _get_ws()
    brain = _get_brain()
    tracker = _get_tracker()
    mode = _get_mode()
    history = _get_history()

    if not brain or not ws:
        await cl.Message(
            content="⚠️ No workspace is open. Please enter a directory path first."
        ).send()
        return

    # Append the user turn to history
    history.append(HumanMessage(content=content))

    # ── Create the response message (streaming) ──
    response_msg = cl.Message(content="", author="Jarvis Code")
    await response_msg.send()

    tool_elements: list[cl.Element] = []
    full_response = ""
    active_tool: str | None = None

    async for event in brain.stream(history, mode):
        etype = event.get("type")
        data = event.get("data")

        if etype == "token":
            token_text = data or ""
            full_response += token_text
            await response_msg.stream_token(token_text)

        elif etype == "tool_start":
            tool_name = data.get("name", "tool")
            args = data.get("args", {})
            active_tool = tool_name
            # Format a readable args summary
            args_str = _format_tool_args(args)
            chip_text = f"🔧 `{tool_name}({args_str})`"
            # Stream the chip inline so the user sees tool activity
            await response_msg.stream_token(f"\n{chip_text}\n")
            full_response += f"\n{chip_text}\n"

        elif etype == "tool_end":
            tool_name = data.get("name", "tool")
            output = data.get("output", "")
            # Add as a collapsible side element
            tool_elements.append(
                cl.Text(
                    name=f"tool_{tool_name}_{len(tool_elements)}",
                    content=f"**{tool_name}** output:\n\n```\n{output}\n```",
                    display="side",
                )
            )
            active_tool = None

        elif etype == "usage":
            # Update the token card in the sidebar
            await _update_sidebar(ws, mode, tracker)

        elif etype == "error":
            await response_msg.stream_token(f"\n\n⚠️ {data}")
            full_response += f"\n\n⚠️ {data}"

    # Finalise the response message with any tool output elements
    response_msg.elements = tool_elements
    await response_msg.update()

    # Append the assistant turn to history
    history.append(AIMessage(content=full_response))
    cl.user_session.set("history", history)


def _format_tool_args(args: dict) -> str:
    """Compact single-line summary of tool arguments for the chip."""
    if not args:
        return ""
    parts = []
    for k, v in args.items():
        v_str = str(v)
        if len(v_str) > 40:
            v_str = v_str[:37] + "…"
        parts.append(f"{k}={v_str!r}")
    result = ", ".join(parts)
    return result if len(result) <= 80 else result[:77] + "…"


# ─────────────────────────── entrypoint ──────────────────────────────────────

# Run this application using:
# python code_ui.py

if __name__ == "__main__":
    import uvicorn
    from chainlit.server import app as server_app

    port = int(os.environ.get("CODE_UI_PORT", 8500))
    host = os.environ.get("CODE_UI_HOST", "0.0.0.0")

    logger.info("Starting Code Assistant UI on http://%s:%d", host, port)
    uvicorn.run(
        server_app, host=host, port=port, timeout_graceful_shutdown=5
    )

