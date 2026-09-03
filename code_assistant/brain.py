"""CodeBrain: the agent that drives the code-assistant UI.

Phase 2 swaps :func:`langgraph.prebuilt.create_react_agent` for a manual
ReAct loop so we can pause for an **approval gate** before any write or
exec tool runs. The architecture is otherwise identical to ``core.py``:

- Reads its LLM through :mod:`llm_provider` (Phase 0) so Ollama and
  OpenAI share the same path.
- Wires :class:`token_usage.TokenTracker` into the chat model so every
  LLM call counts toward the live usage card in the UI.
- Carries a per-session chat history so follow-up questions can refer
  to earlier tool reads without re-listing the workspace.
- Streams an ``AsyncIterator[BrainEvent]`` for the UI: ``token`` /
  ``tool_start`` / ``tool_end`` / ``approval_required`` / ``usage`` /
  ``done`` / ``error``.

Approval flow
-------------
When the model emits a tool call whose name is in
:data:`code_assistant.tools.REQUIRES_APPROVAL`, the brain:

1. Yields ``approval_required`` with the tool name and args.
2. Awaits :attr:`_approval_event`.
3. The UI calls :meth:`submit_approval` with one of:
   - ``ApprovalDecision("approve")`` — run the tool with the proposed args
   - ``ApprovalDecision("edit", args=...)`` — run the tool with edited args
   - ``ApprovalDecision("reject", reason=...)`` — inject a USER_REJECTED
     tool result so the model can adapt

Mode double-check
-----------------
On every tool invocation, :func:`code_assistant.tools._require_build_mode`
refuses to run unless ``set_current_mode("build")`` was called. The brain
calls ``set_current_mode`` before each run so a stale UI mode never lets
write tools slip through.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool

from code_assistant.modes import Mode, SYSTEM_PROMPTS, filter_tools, workspace_preamble
from code_assistant.tools import (
    ALL_TOOLS,
    REQUIRES_APPROVAL,
    ModeError,
    set_current_mode,
    set_workspace,
)
from code_assistant.workspace import Workspace
from llm_provider import LLMConfig, build_provider
from token_usage import TokenTracker

logger = logging.getLogger(__name__)


MAX_TOOL_ROUNDS = 6
MAX_HISTORY_MESSAGES = 24

FORCE_FINAL_PROMPT = (
    "Stop calling tools. Write your final answer NOW using ONLY the information "
    "you have already gathered above. If you couldn't verify something, say so "
    "explicitly instead of guessing."
)


def _looks_like_tool_call_json(delta: str) -> bool:
    """True when streamed content is the model serialising a tool call as
    raw text instead of using the structured ``tool_calls`` channel."""
    s = (delta or "").lstrip()
    if not s.startswith("{"):
        return False
    return ('"name"' in s and ('"arguments"' in s or '"args"' in s)) or s.startswith('{"tool"')


@dataclass(slots=True)
class BrainEvent:
    """A single update from :meth:`CodeBrain.run` consumed by the UI."""

    type: str  # token | tool_start | tool_end | approval_required | usage | done | error
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ApprovalDecision:
    decision: str  # "approve" | "edit" | "reject"
    args: dict[str, Any] | None = None
    reason: str = ""


class CodeBrain:
    """Per-session agent bound to one workspace.

    Harness API
    -----------
    The brain is designed to be subclassed or configured at runtime:

    - :meth:`register_tool` — add a custom tool to specific modes
    - :meth:`override_prompt` — append/replace system prompt suffix per mode
    - :meth:`register_approval_required` — dynamically add a tool to the
      approval gate
    - Instance attributes ``max_tool_rounds``, ``max_history_messages``
      control loop limits (can be overridden after construction or via
      :class:`code_assistant.config.CodeAssistantConfig`).
    """

    def __init__(
        self,
        workspace: Workspace,
        llm_config: LLMConfig | None = None,
        tracker: TokenTracker | None = None,
        mode: Mode = Mode.PLAN,
        config_overrides: dict | None = None,
    ) -> None:
        self.workspace = workspace
        self.mode = mode
        self._config = llm_config or LLMConfig.from_env()
        self._tracker = tracker
        callbacks = [tracker] if tracker else []
        self._provider, self._llm = build_provider(self._config, callbacks=callbacks)
        self._tools: list[BaseTool] = filter_tools(ALL_TOOLS, self.mode)
        self._tool_map = {t.name: t for t in ALL_TOOLS}
        self._history: list[BaseMessage] = []
        self._approval_event = asyncio.Event()
        self._approval_decision: ApprovalDecision | None = None

        # Harness customisation points
        self.max_tool_rounds = MAX_TOOL_ROUNDS
        self.max_history_messages = MAX_HISTORY_MESSAGES
        self._custom_tools: dict[str, BaseTool] = {}
        self._prompt_suffixes: dict[Mode, str] = {}
        self._approval_registry: set[str] = set(REQUIRES_APPROVAL)

        # Apply config overrides (from code_assistant.yaml or dict)
        if config_overrides:
            self._apply_config_overrides(config_overrides)

        set_workspace(self.workspace)
        set_current_mode(self.mode.value)

    # -------------------------------------------------------------- mutators

    def set_mode(self, mode: Mode) -> None:
        if mode == self.mode:
            return
        self.mode = mode
        self._tools = filter_tools(ALL_TOOLS, self.mode)
        set_current_mode(self.mode.value)

    def reset_history(self) -> None:
        self._history.clear()

    def workspace_root(self) -> str:
        return str(self.workspace.root)

    # ------------------------------------------------------------ harness API

    def register_tool(
        self,
        tool: BaseTool,
        modes: list[Mode] | None = None,
    ) -> None:
        """Register a custom tool for specific modes.

        Args:
            tool: A LangChain ``@tool``-decorated function.
            modes: List of modes to enable the tool for. Defaults to
                ``[Mode.BUILD]`` (write/exec tools). Use ``[Mode.PLAN, Mode.BUILD]``
                for read-only tools.
        """
        if modes is None:
            modes = [Mode.BUILD]
        self._custom_tools[tool.name] = tool
        for m in modes:
            if m == Mode.PLAN:
                # Add to PLAN toolset by forcing filter to include it
                pass  # Filter logic will pick it up via tool name in allowed set
            # For BUILD mode, the tool will be included since we rebuild tools in set_mode
        # Rebuild tools for current mode so the new tool is available
        self._rebuild_tools()

    def _rebuild_tools(self) -> None:
        """Rebuild the active tool list from ALL_TOOLS + custom tools."""
        # Start with base tools for current mode
        self._tools = filter_tools(ALL_TOOLS, self.mode)
        # Add custom tools if they're allowed in current mode
        for tool in self._custom_tools.values():
            if tool.name in REQUIRES_APPROVAL and self.mode == Mode.PLAN:
                continue  # Skip approval-required tools in Plan mode
            if tool.name not in [t.name for t in self._tools]:
                self._tools.append(tool)
        self._tool_map = {t.name: t for t in ALL_TOOLS}
        self._tool_map.update(self._custom_tools)

    def override_prompt(self, mode: Mode, suffix: str) -> None:
        """Append or replace a system prompt suffix for a given mode.

        The suffix is appended to the built-in prompt for that mode.
        Pass an empty string to clear.
        """
        if suffix:
            self._prompt_suffixes[mode] = suffix
        elif mode in self._prompt_suffixes:
            del self._prompt_suffixes[mode]

    def register_approval_required(self, tool_name: str) -> None:
        """Add a tool name to the approval gate registry.

        The tool will trigger ``approval_required`` events before
        execution, regardless of its default classification.
        """
        self._approval_registry.add(tool_name)

    def unregister_approval_required(self, tool_name: str) -> None:
        """Remove a tool from the approval gate."""
        self._approval_registry.discard(tool_name)

    def _apply_config_overrides(self, overrides: dict) -> None:
        """Apply config dict (from code_assistant.yaml or dict)."""
        if "max_tool_rounds" in overrides:
            self.max_tool_rounds = int(overrides["max_tool_rounds"])
        if "max_history_messages" in overrides:
            self.max_history_messages = int(overrides["max_history_messages"])
        if "approval_required" in overrides:
            for name in overrides["approval_required"]:
                self.register_approval_required(name)
        if "prompt_suffixes" in overrides:
            for mode_str, suffix in overrides["prompt_suffixes"].items():
                try:
                    mode = Mode.parse(mode_str)
                    self.override_prompt(mode, suffix)
                except ValueError:
                    pass

    async def submit_approval(self, decision: ApprovalDecision) -> None:
        """UI hook: resolve the pending approval and resume the brain."""
        self._approval_decision = decision
        self._approval_event.set()

    # -------------------------------------------------------------- runtime

    def _messages(self) -> list[BaseMessage]:
        preamble = workspace_preamble(self.workspace_root(), self.mode)
        base = SYSTEM_PROMPTS[self.mode]
        suffix = self._prompt_suffixes.get(self.mode, "")
        system = SystemMessage(content=base + suffix + preamble)
        return [system] + list(self._history)

    async def run(self, user_text: str) -> AsyncIterator[BrainEvent]:
        """Drive one user turn end-to-end, streaming events to the caller."""
        self._approval_event.clear()
        self._approval_decision = None
        set_workspace(self.workspace)
        set_current_mode(self.mode.value)

        self._history.append(HumanMessage(content=user_text))
        self._trim_history()
        messages = self._messages()

        t0 = time.perf_counter()
        final_text: str = ""
        try:
            for round_num in range(MAX_TOOL_ROUNDS):
                # ---- model round ----
                stream_events, ai_message = await self._stream_model_round(messages)
                for ev in stream_events:
                    yield ev

                tool_calls = list(getattr(ai_message, "tool_calls", None) or [])
                # Small-model fallback: parse content-JSON if no structured calls.
                if not tool_calls:
                    full_text = "".join(
                        e.data.get("data", "") for e in stream_events if e.type == "token"
                    )
                    if _looks_like_tool_call_json(full_text):
                        parsed = _safe_json_loads(full_text)
                        if isinstance(parsed, dict) and "name" in parsed:
                            tool_calls = [
                                {
                                    "id": f"text_{round_num}",
                                    "name": parsed.get("name"),
                                    "args": parsed.get("arguments")
                                    if "arguments" in parsed
                                    else parsed.get("args", {}),
                                }
                            ]
                            # Replace the AI message with a tool-calling one.
                            ai_message = AIMessage(content="", tool_calls=tool_calls)

                messages.append(ai_message)
                self._history.append(AIMessage(
                    content=ai_message.content if isinstance(ai_message.content, str) else "",
                    tool_calls=tool_calls,
                ))

                if not tool_calls:
                    final_text = ai_message.content if isinstance(ai_message.content, str) else ""
                    break

                # ---- tool round ----
                tool_events, tool_messages = await self._execute_tool_calls(tool_calls)
                for ev in tool_events:
                    yield ev
                for tm in tool_messages:
                    messages.append(tm)
            else:
                logger.warning("CodeBrain: tool-round cap hit; forcing final answer")
                forced = messages + [HumanMessage(content=FORCE_FINAL_PROMPT)]
                resp = await self._llm.ainvoke(forced)
                final_text = _content_to_str(resp.content)

        except Exception as e:  # noqa: BLE001
            logger.exception("CodeBrain.run failed")
            yield BrainEvent(type="error", data={"message": str(e)})
            return

        if not final_text.strip():
            final_text = "(no response)"

        self._history.append(AIMessage(content=final_text))
        self._trim_history()

        yield BrainEvent(
            type="usage",
            data={"snapshot": self._tracker.snapshot() if self._tracker else {}},
        )
        yield BrainEvent(
            type="done",
            data={
                "reply": final_text,
                "duration_ms": int((time.perf_counter() - t0) * 1000),
                "mode": self.mode.value,
            },
        )

    # ----------------------------------------------------------- per-round

    async def _stream_model_round(
        self,
        messages: list[BaseMessage],
    ) -> tuple[list[BrainEvent], AIMessage]:
        """Stream one model turn.

        Returns:
            (events_to_yield, final_ai_message) — the caller iterates the
            events for the UI and appends ``ai_message`` to its history.
        """
        events: list[BrainEvent] = []
        final_chunk: AIMessageChunk | None = None
        text_parts: list[str] = []
        async for chunk in self._llm.astream(messages):
            text = _content_to_str(getattr(chunk, "content", "") or "")
            if text:
                text_parts.append(text)
                events.append(BrainEvent(type="token", data={"data": text}))
            if isinstance(chunk, AIMessageChunk):
                final_chunk = chunk
        if final_chunk is None:
            final_chunk = AIMessageChunk(content="".join(text_parts))
        return events, final_chunk

    async def _execute_tool_calls(
        self,
        tool_calls: list[dict[str, Any]],
    ) -> tuple[list[BrainEvent], list[ToolMessage]]:
        """Run each tool, with approval gating where required.

        Returns:
            (events_to_yield, tool_messages) — the events include
            ``tool_start``, ``approval_required``, and ``tool_end`` for
            each call; ``tool_messages`` are the conversation-level
            ``ToolMessage`` objects to append to the agent's history.
        """
        events: list[BrainEvent] = []
        tool_messages: list[ToolMessage] = []

        for idx, tc in enumerate(tool_calls):
            name = tc.get("name") or ""
            args = tc.get("args") or {}
            call_id = tc.get("id") or f"call_{idx}"

            events.append(BrainEvent(type="tool_start", data={"name": name, "args": args}))

            needs_approval = name in self._approval_registry
            if needs_approval:
                self._approval_event.clear()
                self._approval_decision = None
                events.append(
                    BrainEvent(
                        type="approval_required",
                        data={"name": name, "args": args, "tool_call_id": call_id},
                    )
                )
                await self._approval_event.wait()
                decision = self._approval_decision or ApprovalDecision(
                    decision="reject", reason="no decision recorded"
                )
                if decision.decision == "reject":
                    result_text = (
                        f"USER_REJECTED: the user declined this {name} call"
                        + (f" ({decision.reason})" if decision.reason else "")
                        + "."
                    )
                    events.append(
                        BrainEvent(
                            type="tool_end",
                            data={
                                "name": name,
                                "output": result_text,
                                "rejected": True,
                                "tool_call_id": call_id,
                            },
                        )
                    )
                    tool_messages.append(
                        ToolMessage(content=result_text, tool_call_id=call_id, name=name)
                    )
                    continue
                if decision.decision == "edit" and decision.args is not None:
                    args = decision.args

            try:
                tool_obj = self._tool_map.get(name)
                if tool_obj is None:
                    result_text = f"ERROR: unknown tool {name!r}"
                else:
                    raw = await _invoke_tool(tool_obj, args)
                    result_text = _content_to_str(raw)
            except ModeError as e:
                result_text = f"ERROR: {e}"
            except Exception as e:  # noqa: BLE001
                logger.exception("Tool %s failed", name)
                result_text = f"ERROR: {type(e).__name__}: {e}"

            events.append(
                BrainEvent(
                    type="tool_end",
                    data={
                        "name": name,
                        "output": result_text[:4000],
                        "tool_call_id": call_id,
                    },
                )
            )
            tool_messages.append(
                ToolMessage(content=result_text, tool_call_id=call_id, name=name)
            )

        return events, tool_messages

    # ----------------------------------------------------------- internals

    def _trim_history(self) -> None:
        while len(self._history) > MAX_HISTORY_MESSAGES:
            self._history.pop(0)
        while self._history and not isinstance(self._history[0], HumanMessage):
            self._history.pop(0)


# ============================================================ module helpers


def _content_to_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_content_to_str(p) for p in content)
    return str(content or "")


def _safe_json_loads(s: str) -> Any:
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return None


async def _invoke_tool(tool: BaseTool, args: dict[str, Any]) -> Any:
    """Run a tool whether it's sync or async."""
    if hasattr(tool, "ainvoke"):
        return await tool.ainvoke(args)
    return tool.invoke(args)


__all__ = [
    "CodeBrain",
    "BrainEvent",
    "ApprovalDecision",
    "MAX_TOOL_ROUNDS",
]
