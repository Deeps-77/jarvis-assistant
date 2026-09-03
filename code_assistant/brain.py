"""CodeBrain — the ReAct agent for the Jarvis Code Assistant (Phase 1).

``CodeBrain`` wraps a LangGraph ``create_react_agent`` and exposes a single
async generator ``stream(messages, mode)`` that yields typed events the
Chainlit UI can render without knowing anything about LangGraph internals.

Event types (``BrainEvent``)
----------------------------
``{"type": "token",     "data": str}``        — streamed LLM text token
``{"type": "tool_start","data": {name, args}}`` — tool call started
``{"type": "tool_end",  "data": {name, output}}`` — tool call finished
``{"type": "usage",     "data": {input, output, total, cost_usd}}`` — per-turn totals
``{"type": "error",     "data": str}``         — recoverable error message

Design notes
------------
- The agent is **re-built** each call (``create_react_agent`` is cheap). This
  avoids stale state between turns and lets the mode change freely.
- The workspace context is injected into the system message on every call so
  the model always knows which folder it is operating in.
- ``GraphRecursionError`` is caught and surfaced as an ``error`` event so the
  UI can tell the user to rephrase rather than showing a raw traceback.
- The ``TokenTracker`` callback captures token counts; ``snapshot()`` is
  emitted as a ``usage`` event at the end of each stream.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages import BaseMessage
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from llm_provider import LLMConfig, build_provider
from token_usage import TokenTracker
from code_assistant.modes import Mode, SYSTEM_PROMPTS, filter_tools
from code_assistant.tools import make_read_tools
from code_assistant.workspace import Workspace, count_workspace_contents

logger = logging.getLogger(__name__)

# Maximum ReAct rounds before we abort and tell the user.
_MAX_TOOL_ROUNDS = 8

# Type alias for the event dicts the UI consumes.
BrainEvent = dict[str, Any]


class CodeBrain:
    """Stateless wrapper around a LangGraph ReAct agent.

    Parameters
    ----------
    workspace:
        The current workspace. Injected into every system prompt.
    llm_config:
        ``LLMConfig`` built from ``LLMConfig.from_env()`` or a YAML config.
    tracker:
        ``TokenTracker`` shared across the session. Its snapshot is emitted
        as a ``usage`` event at the end of each ``stream()`` call.
    """

    def __init__(
        self,
        workspace: Workspace,
        llm_config: LLMConfig,
        tracker: TokenTracker,
    ) -> None:
        self.workspace = workspace
        self.llm_config = llm_config
        self.tracker = tracker
        self._all_tools = make_read_tools(workspace)

    # ─────────────────────────────── public API ───────────────────────────────

    async def stream(
        self,
        messages: list[BaseMessage],
        mode: Mode = Mode.PLAN,
    ) -> AsyncIterator[BrainEvent]:
        """Run the agent and yield ``BrainEvent`` dicts.

        Parameters
        ----------
        messages:
            Conversation history (HumanMessage / AIMessage / ToolMessage).
            The system message is prepended here; callers should NOT include one.
        mode:
            Current mode. Controls which tools are available and which system
            prompt is used.
        """
        tools = filter_tools(self._all_tools, mode)
        system_msg = self._build_system_message(mode)

        _, llm = build_provider(self.llm_config, callbacks=[self.tracker])

        agent = create_react_agent(
            llm,
            tools=tools,
            prompt=system_msg,
        )

        full_messages = [system_msg] + list(messages)

        try:
            async for event in self._run_agent(agent, full_messages):
                yield event
        except GraphRecursionError:
            yield {
                "type": "error",
                "data": (
                    f"The agent exceeded {_MAX_TOOL_ROUNDS} tool-call rounds. "
                    "Try a more specific question or break the task into smaller steps."
                ),
            }
        except Exception as e:
            logger.exception("CodeBrain.stream error")
            yield {"type": "error", "data": f"Agent error: {e}"}

        # Always emit usage at the end of a stream
        snap = self.tracker.snapshot()
        yield {
            "type": "usage",
            "data": {
                "input": snap["input_tokens"],
                "output": snap["output_tokens"],
                "total": snap["total_tokens"],
                "cost_usd": snap["cost_usd"],
                "turns": snap["turns"],
            },
        }

    # ─────────────────────────── internal helpers ─────────────────────────────

    def _build_system_message(self, mode: Mode) -> SystemMessage:
        """Compose the system prompt with workspace context prepended."""
        try:
            counts = count_workspace_contents(self.workspace)
            ws_context = (
                f"Workspace: {self.workspace.root}\n"
                f"Name: {self.workspace.name}\n"
                f"Contents: ~{counts['files']} files, ~{counts['dirs']} directories\n"
                f"Mode: {mode.value.upper()}\n\n"
            )
        except Exception:
            ws_context = f"Workspace: {self.workspace.root}\nMode: {mode.value.upper()}\n\n"

        base_prompt = SYSTEM_PROMPTS[mode]
        return SystemMessage(content=ws_context + base_prompt)

    async def _run_agent(
        self,
        agent: Any,
        messages: list[BaseMessage],
    ) -> AsyncIterator[BrainEvent]:
        """Stream raw events from the LangGraph agent and normalise them."""
        config = {"recursion_limit": _MAX_TOOL_ROUNDS * 2 + 2}

        # LangGraph astream_events emits granular events we can map to BrainEvents.
        async for raw_event in agent.astream_events(
            {"messages": messages},
            config=config,
            version="v2",
        ):
            event_name = raw_event.get("event", "")
            data = raw_event.get("data", {})

            if event_name == "on_chat_model_stream":
                chunk = data.get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    content = chunk.content
                    if isinstance(content, str):
                        yield {"type": "token", "data": content}
                    elif isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                yield {"type": "token", "data": part["text"]}

            elif event_name == "on_tool_start":
                tool_name = raw_event.get("name", "unknown_tool")
                tool_input = data.get("input", {})
                yield {
                    "type": "tool_start",
                    "data": {
                        "name": tool_name,
                        "args": tool_input,
                    },
                }

            elif event_name == "on_tool_end":
                tool_name = raw_event.get("name", "unknown_tool")
                tool_output = data.get("output", "")
                # Truncate very long outputs in the event (full output goes to
                # the LLM via ToolMessage; UI only needs a preview).
                preview = str(tool_output)
                if len(preview) > 500:
                    preview = preview[:500] + " …[truncated in chip]"
                yield {
                    "type": "tool_end",
                    "data": {
                        "name": tool_name,
                        "output": preview,
                    },
                }


__all__ = ["CodeBrain", "BrainEvent", "Mode"]
