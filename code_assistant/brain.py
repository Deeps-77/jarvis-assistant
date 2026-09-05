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


def _strip_thinking(text: str) -> str:
    """Remove Qwen3 ``<think>...</think>`` blocks; tool JSON comes after them."""
    if "<think>" in text and "</think>" in text:
        parts = text.split("</think>", 1)
        return parts[1] if len(parts) > 1 else text
    return text


def _looks_like_tool_call_json(delta: str) -> bool:
    """True when streamed content is the model serialising a tool call as
    raw text instead of using the structured ``tool_calls`` channel.

    Handles Qwen3/Hermes styles: bare JSON, ```json fences,
    ``<tool_call>`` / ``<function_call>`` / ``[TOOL_CALLS]`` wrappers.
    """
    s = _strip_thinking(delta or "").strip()
    if not s:
        return False
    markers = ('"name"', '"function"', '"tool"', "<tool_call", "<function_call", "[TOOL_CALLS]")
    return any(m in s for m in markers)


def _candidate_json_blocks(text: str) -> list[str]:
    """Pull candidate JSON payloads out of fences, tags, and brace spans."""
    import re
    import uuid as _uuid  # local to avoid touching module imports

    out: list[str] = []
    s = _strip_thinking(text or "")
    # 1. Fenced blocks ```json ... ``` or ``` ... ```
    for m in re.finditer(r"```(?:json)?\s*(.*?)```", s, re.DOTALL | re.IGNORECASE):
        out.append(m.group(1).strip())
    # 2. <tool_call>...</tool_call> and <function_call>...</function_call>
    for m in re.finditer(r"<(?:tool_call|function_call)>(.*?)</(?:tool_call|function_call)>", s, re.DOTALL | re.IGNORECASE):
        out.append(m.group(1).strip())
    # 3. Whole-text array [ {...}, {...} ]
    stripped = s.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        out.append(stripped)
    # 4. Balanced-brace spans containing a tool marker
    for m in re.finditer(r"\{[^{}]*\"(?:name|function|tool)\"[^{}]*\}", s, re.DOTALL):
        out.append(m.group(0))
    # 5. Raw text as last resort
    out.append(s.strip())
    # De-dupe, keep order
    seen: set[str] = set()
    uniq: list[str] = []
    for b in out:
        if b and b not in seen:
            seen.add(b)
            uniq.append(b)
    return uniq


def _normalise_one_call(item: Any) -> dict[str, Any] | None:
    """Normalise one parsed dict into {id, name, args} or None if invalid."""
    import uuid
    if not isinstance(item, dict):
        return None
    # Hermes style: {"function": {"name": ..., "arguments": ...}}
    if "function" in item and isinstance(item["function"], dict):
        fn = item["function"]
        name = fn.get("name")
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {}
    elif "tool" in item and isinstance(item["tool"], str) and "name" not in item:
        # {"tool": "read_file", "args"/"parameters": {...}}
        name = item.get("tool")
        args = item.get("args", item.get("parameters", item.get("input", {})))
    else:
        name = item.get("name")
        args = item.get("arguments", item.get("args", item.get("parameters", {})))
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {}
    if not name or not isinstance(name, str):
        return None
    if not isinstance(args, dict):
        args = {}
    return {"id": f"content_{uuid.uuid4().hex[:8]}", "name": name.strip(), "args": args}


def _extract_tool_calls_from_content(content: str) -> list[dict[str, Any]]:
    """Extract tool calls from model content when they are serialized as JSON.

    Handles:
    - {"name": "...", "arguments": {...}} / {"args": {...}}
    - [{"name": ...}, ...] and newline-delimited JSON objects
    - ```json fences, <tool_call>/<function_call> tags (Qwen3/Hermes)
    - {"function": {"name": ..., "arguments": ...}} nesting
    """
    if not content or not _looks_like_tool_call_json(content):
        return []
    calls: list[dict[str, Any]] = []
    for block in _candidate_json_blocks(content):
        # Try whole block as JSON first
        parsed = _safe_json_loads(block)
        if isinstance(parsed, dict):
            one = _normalise_one_call(parsed)
            if one:
                calls.append(one)
                continue
        elif isinstance(parsed, list):
            for item in parsed:
                one = _normalise_one_call(item)
                if one:
                    calls.append(one)
            if calls:
                continue
        # Try newline-delimited JSON objects inside the block
        for line in block.splitlines():
            line = line.strip().rstrip(",")
            if not (line.startswith("{") and line.endswith("}")):
                continue
            one = _normalise_one_call(_safe_json_loads(line))
            if one:
                calls.append(one)
        if calls:
            break
    # De-dupe identical (name, args) pairs, keep order
    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for c in calls:
        key = json.dumps({"name": c["name"], "args": c["args"]}, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def _accumulate_tool_call_chunks(chunks: list[dict]) -> list[dict]:
    """Accumulate tool_call_chunks from streamed chunks into complete tool calls."""
    if not chunks:
        return []
    
    # Group by tool call index
    tool_calls: dict[int, dict] = {}
    for chunk in chunks:
        idx = chunk.get("index", 0)
        if idx not in tool_calls:
            tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
        
        tc = tool_calls[idx]
        if "id" in chunk and chunk["id"]:
            tc["id"] = chunk["id"]
        if "name" in chunk and chunk["name"]:
            tc["name"] += chunk["name"]
        if "arguments" in chunk and chunk["arguments"]:
            tc["arguments"] += chunk["arguments"]
    
    import uuid
    result = []
    for idx in sorted(tool_calls.keys()):
        tc = tool_calls[idx]
        if tc["name"]:
            raw_args = tc["arguments"]
            if isinstance(raw_args, dict):
                args = raw_args
            else:
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except (json.JSONDecodeError, TypeError):
                    args = {}
                if not isinstance(args, dict):
                    args = {}
            result.append({
                "id": tc["id"] or f"tool_{idx}_{uuid.uuid4().hex[:6]}",
                "name": tc["name"],
                "args": args,
            })
    return result


@dataclass(slots=True)
class BrainEvent:
    """A single update from :meth:`CodeBrain.run` consumed by the UI."""

    # token | thinking | tool_start | tool_end | approval_required | usage | done | error
    type: str
    data: dict[str, Any] = field(default_factory=dict)


#: Cap on streamed reasoning chars per model round (thinking traces are verbose).
MAX_REASONING_CHARS = 4000


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
        self._llm_with_tools = self._bind_current_tools()
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

    def _bind_current_tools(self):
        """Bind the active toolset to the LLM; fall back to raw LLM on failure."""
        try:
            return self._llm.bind_tools(self._tools)
        except Exception:
            logger.warning("bind_tools failed; continuing without structured tool binding")
            return self._llm

    def set_mode(self, mode: Mode) -> None:
        if mode == self.mode:
            return
        self.mode = mode
        self._tools = filter_tools(ALL_TOOLS, self.mode)
        self._llm_with_tools = self._bind_current_tools()
        set_current_mode(self.mode.value)

    def reset_history(self) -> None:
        self._history.clear()

    def export_history(self) -> list[BaseMessage]:
        """Return a copy of the conversation history for session persistence."""
        return list(self._history)

    def import_history(self, messages: list[BaseMessage]) -> None:
        """Replace history with previously exported messages (session resume)."""
        self._history = list(messages or [])
        self._trim_history()

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
        self._llm_with_tools = self._bind_current_tools()

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
            for round_num in range(self.max_tool_rounds):
                # ---- model round ----
                stream_events, ai_message = await self._stream_model_round(messages)
                for ev in stream_events:
                    yield ev

                tool_calls = list(getattr(ai_message, "tool_calls", None) or [])
                # Belt and suspenders: _stream_model_round already tries
                # content-JSON, but retry here in case tokens arrived without
                # triggering it (e.g. thinking tags interleaved).
                if not tool_calls:
                    full_text = "".join(
                        e.data.get("data", "") for e in stream_events if e.type == "token"
                    )
                    if _looks_like_tool_call_json(full_text):
                        tool_calls = _extract_tool_calls_from_content(full_text)
                        if tool_calls:
                            # Replace the AI message with a tool-calling one.
                            ai_message = AIMessage(content="", tool_calls=tool_calls)

                messages.append(ai_message)
                self._history.append(AIMessage(
                    content=ai_message.content if isinstance(ai_message.content, str) else "",
                    tool_calls=tool_calls,
                ))

                if not tool_calls:
                    final_text = ai_message.content if isinstance(ai_message.content, str) else ""
                    if final_text.strip():
                        break
                    # Empty round (thinking model emitted nothing usable):
                    # nudge once instead of surrendering with "(no response)".
                    # Counts against max_tool_rounds so it always terminates.
                    nudge = HumanMessage(
                        content=(
                            "Your last response contained neither text nor a tool call. "
                            "Reply now: either call one of the listed tools with valid "
                            "arguments, or answer the user's request in plain prose."
                        )
                    )
                    messages.append(nudge)
                    logger.warning("CodeBrain: empty model round %d; nudging", round_num)
                    continue

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
        tool_call_chunks: list[dict] = []
        streamed_calls: dict[str, dict] = {}
        streamed_invalid: list[dict] = []
        reasoning_chars = 0

        model = getattr(self, "_llm_with_tools", None) or self._llm
        async for chunk in model.astream(messages):
            text = _content_to_str(getattr(chunk, "content", "") or "")
            if text:
                text_parts.append(text)
                events.append(BrainEvent(type="token", data={"data": text}))

            # Thinking trace (only present when the provider is built with
            # reasoning=True and the model thinks). Streamed so the UI can
            # show progress during long generations instead of sitting silent.
            addl = getattr(chunk, "additional_kwargs", None) or {}
            thinking = addl.get("reasoning_content") or ""
            if thinking and reasoning_chars < MAX_REASONING_CHARS:
                room = MAX_REASONING_CHARS - reasoning_chars
                piece = thinking[:room]
                reasoning_chars += len(piece)
                events.append(BrainEvent(type="thinking", data={"data": piece}))
            
            # Collect structured calls from EVERY chunk (LangChain may surface
            # the complete call on any chunk, not just the last one).
            for tc in getattr(chunk, "tool_calls", None) or []:
                try:
                    cid = tc.get("id") or f"stream_{len(streamed_calls)}"
                    streamed_calls[cid] = {
                        "id": cid,
                        "name": tc.get("name") or "",
                        "args": tc.get("args") if isinstance(tc.get("args"), dict) else {},
                    }
                except (AttributeError, TypeError):
                    continue
            # Keep malformed calls too — the Parable/Qwen3 finetune sometimes
            # emits bad JSON args; we repair them below instead of going silent.
            for bad in getattr(chunk, "invalid_tool_calls", None) or []:
                try:
                    streamed_invalid.append(dict(bad))
                except (TypeError, ValueError):
                    continue

            # Collect tool_call_chunks from streamed chunks (for models that support it)
            tcc = getattr(chunk, "tool_call_chunks", None) or []
            for tc_chunk in tcc:
                # Convert LangChain's tool_call_chunk to dict
                tc_dict = {
                    "index": getattr(tc_chunk, "index", 0),
                    "id": getattr(tc_chunk, "id", None),
                    "name": getattr(tc_chunk, "name", None),
                    "arguments": getattr(tc_chunk, "args", None) or getattr(tc_chunk, "arguments", None),
                }
                # Filter out None values
                tc_dict = {k: v for k, v in tc_dict.items() if v is not None}
                if tc_dict:
                    tool_call_chunks.append(tc_dict)
            
            if isinstance(chunk, AIMessageChunk):
                final_chunk = chunk
        
        # Build the final AI message
        full_text = "".join(text_parts)

        # 1. Structured calls seen on any streamed chunk (most reliable).
        tool_calls = [c for c in streamed_calls.values() if c.get("name")]

        # 2. Malformed calls: repair what we can (known name + coerced args).
        if not tool_calls and streamed_invalid:
            for bad in streamed_invalid:
                name = bad.get("name") or ""
                raw_args = bad.get("args")
                args: dict[str, Any] = {}
                if isinstance(raw_args, dict):
                    args = raw_args
                elif isinstance(raw_args, str) and raw_args.strip():
                    candidate = _normalise_one_call({"name": name, "arguments": raw_args})
                    if candidate:
                        args = candidate["args"]
                if name:
                    import uuid
                    tool_calls.append({
                        "id": bad.get("id") or f"repaired_{uuid.uuid4().hex[:6]}",
                        "name": name,
                        "args": args,
                    })

        # 3. Text-serialised JSON (small-model fallback).
        if not tool_calls and _looks_like_tool_call_json(full_text):
            tool_calls = _extract_tool_calls_from_content(full_text)

        # 4. Accumulated tool_call_chunks.
        if not tool_calls and tool_call_chunks:
            accumulated = _accumulate_tool_call_chunks(tool_call_chunks)
            if accumulated:
                tool_calls = accumulated

        # 5. Final-chunk fallback (covers providers that only set the last chunk).
        if not tool_calls and final_chunk is not None:
            tool_calls = list(getattr(final_chunk, "tool_calls", None) or [])
        
        # Create the final AIMessage (override with extracted calls if found).
        if final_chunk is None:
            ai_message = AIMessage(content=full_text, tool_calls=tool_calls)
        else:
            ai_message = AIMessage(
                content=full_text,
                tool_calls=tool_calls if tool_calls else getattr(final_chunk, "tool_calls", None) or [],
            )

        # Note: we don't yield here; the caller will iterate the returned events
        return events, ai_message

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

        allowed_names = {t.name for t in self._tools}
        for idx, tc in enumerate(tool_calls):
            name = tc.get("name") or ""
            args = tc.get("args") or {}
            call_id = tc.get("id") or f"call_{idx}"

            events.append(BrainEvent(type="tool_start", data={"name": name, "args": args}))

            # Pre-approval validation: fail fast without prompting the user.
            if not isinstance(args, dict):
                result_text = f"ERROR: invalid args for {name!r}: must be an object."
                events.append(
                    BrainEvent(
                        type="tool_end",
                        data={"name": name, "output": result_text, "tool_call_id": call_id},
                    )
                )
                tool_messages.append(
                    ToolMessage(content=result_text, tool_call_id=call_id, name=name)
                )
                continue
            if name not in self._tool_map:
                result_text = f"ERROR: unknown tool {name!r}"
                events.append(
                    BrainEvent(
                        type="tool_end",
                        data={"name": name, "output": result_text, "tool_call_id": call_id},
                    )
                )
                tool_messages.append(
                    ToolMessage(content=result_text, tool_call_id=call_id, name=name)
                )
                continue
            if name not in allowed_names:
                result_text = (
                    f"ERROR: {name} is not available in {self.mode.value.upper()} mode. "
                    f"Switch to BUILD with `/mode build` for write/exec tools."
                )
                events.append(
                    BrainEvent(
                        type="tool_end",
                        data={"name": name, "output": result_text, "tool_call_id": call_id},
                    )
                )
                tool_messages.append(
                    ToolMessage(content=result_text, tool_call_id=call_id, name=name)
                )
                continue

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
        while len(self._history) > self.max_history_messages:
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
    "MAX_REASONING_CHARS",
]
