"""CodeBrain: the ReAct agent that drives the code-assistant UI.

Mirrors the structure of ``core.py`` (the chat brain) but:

- Reads its LLM through :mod:`llm_provider` (Phase 0) so a single class
  powers both Ollama and OpenAI.
- Wires :class:`token_usage.TokenTracker` into the chat model so every
  LLM call — including tool-calling rounds — counts toward the live
  usage card in the UI.
- Carries a per-session chat history under ``chat_histories[session_key]``
  so follow-up questions can reference earlier tool reads without
  re-listing the workspace.
- Streams an ``AsyncIterator[BrainEvent]`` for the UI rather than returning
  one big string. Events let the UI show tool chips, retract intermediate
  chatter when the model calls a tool, and surface token usage per turn.

Phase 1 ships read-only tools; Phase 2 will add a ``BrainEvent`` kind
``approval_required`` and a resume hook for the approval gate.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from code_assistant.modes import Mode, SYSTEM_PROMPTS, filter_tools, workspace_preamble
from code_assistant.tools import ALL_TOOLS, set_workspace
from code_assistant.workspace import Workspace
from llm_provider import LLMConfig, build_provider
from token_usage import TokenTracker

logger = logging.getLogger(__name__)


# Cap on the ReAct tool-calling rounds. Mirrors core.MAX_TOOL_ROUNDS.
MAX_TOOL_ROUNDS = 6
MAX_HISTORY_MESSAGES = 24

FORCE_FINAL_PROMPT = (
    "Stop calling tools. Write your final answer NOW using ONLY the information "
    "you have already gathered above. If you couldn't verify something, say so "
    "explicitly instead of guessing."
)


def _looks_like_tool_call_json(delta: str) -> bool:
    """True when a streamed content delta is the model serialising a tool
    call as raw text.

    Smaller local models occasionally emit tool calls in the content channel
    instead of the structured ``tool_calls`` channel. When that happens the
    UI would otherwise show raw JSON to the user. We strip those chunks so
    only the structured tool_start event surfaces in the UI.
    """
    s = (delta or "").lstrip()
    if not s.startswith("{"):
        return False
    # Cheap structural check — avoid json.loads() per token for speed.
    return ('"name"' in s and ('"arguments"' in s or '"args"' in s)) or s.startswith('{"tool"')


@dataclass(slots=True)
class BrainEvent:
    """A single update from :meth:`CodeBrain.run` consumed by the UI."""

    type: str  # "token" | "tool_start" | "tool_end" | "usage" | "done" | "error" | "approval_required"
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "data": self.data}


class CodeBrain:
    """Per-session ReAct agent bound to one workspace."""

    def __init__(
        self,
        workspace: Workspace,
        llm_config: LLMConfig | None = None,
        tracker: TokenTracker | None = None,
        mode: Mode = Mode.PLAN,
    ) -> None:
        self.workspace = workspace
        self.mode = mode
        self._config = llm_config or LLMConfig.from_env()
        self._tracker = tracker
        callbacks = [tracker] if tracker else []
        self._provider, self._llm = build_provider(self._config, callbacks=callbacks)
        self._tools = filter_tools(ALL_TOOLS, self.mode)
        self._agent = create_react_agent(self._llm, self._tools)
        self._history: list[BaseMessage] = []
        # Inject workspace into the tool module so the LangChain tools see it.
        set_workspace(self.workspace)

    # -------------------------------------------------------------- mutators

    def set_mode(self, mode: Mode) -> None:
        if mode == self.mode:
            return
        self.mode = mode
        self._tools = filter_tools(ALL_TOOLS, self.mode)
        # Rebuild the ReAct graph with the new toolbelt.
        self._agent = create_react_agent(self._llm, self._tools)

    def reset_history(self) -> None:
        self._history.clear()

    def workspace_root(self) -> str:
        return str(self.workspace.root)

    # -------------------------------------------------------------- runtime

    def _messages(self, user_text: str) -> list[BaseMessage]:
        preamble = workspace_preamble(self.workspace_root(), self.mode)
        system = SystemMessage(content=SYSTEM_PROMPTS[self.mode] + preamble)
        return [system] + list(self._history) + [HumanMessage(content=user_text)]

    async def run(self, user_text: str) -> AsyncIterator[BrainEvent]:
        """Drive one user turn end-to-end, streaming events to the caller.

        Yields:
            - ``token``: model text delta (may be retracted if a tool follows)
            - ``tool_start``: a tool call is being made
            - ``tool_end``: a tool call finished (success or error string)
            - ``usage``: post-turn snapshot of TokenTracker
            - ``done``: terminal event with the final reply
            - ``error``: terminal event if the turn failed
        """
        messages = self._messages(user_text)
        self._history.append(HumanMessage(content=user_text))
        self._trim_history()

        config = {"recursion_limit": MAX_TOOL_ROUNDS * 2 + 4}
        t0 = time.perf_counter()
        generated: list[BaseMessage] = []
        final_text: str = ""
        try:
            async for ev in self._run_streaming(messages, config, generated):
                yield ev
            # Find the LAST AIMessage — the final model answer. The list may
            # end on a ToolMessage if the agent loop closed right after a tool
            # returned without a final synthesis turn.
            final_text = ""
            for msg in reversed(generated):
                if isinstance(msg, AIMessage):
                    c = msg.content
                    final_text = c if isinstance(c, str) else " ".join(str(p) for p in c)
                    if final_text.strip():
                        break
        except GraphRecursionError:
            logger.warning("CodeBrain: tool-round cap hit; forcing final answer")
            forced = messages + generated + [HumanMessage(content=FORCE_FINAL_PROMPT)]
            try:
                resp = await self._llm.ainvoke(forced)
                final_text = resp.content if isinstance(resp.content, str) else str(resp.content)
                yield BrainEvent("token", {"data": final_text})
            except Exception as e:  # noqa: BLE001
                yield BrainEvent("error", {"message": str(e)})
                return
        except Exception as e:  # noqa: BLE001
            logger.exception("CodeBrain.run failed")
            yield BrainEvent("error", {"message": str(e)})
            return

        if not final_text:
            final_text = "(no response)"

        self._history.append(AIMessage(content=final_text))
        self._trim_history()

        yield BrainEvent(
            "usage",
            {"snapshot": self._tracker.snapshot() if self._tracker else {}},
        )
        yield BrainEvent(
            "done",
            {
                "reply": final_text,
                "duration_ms": int((time.perf_counter() - t0) * 1000),
                "mode": self.mode.value,
            },
        )

    # ----------------------------------------------------------- internals

    async def _run_streaming(
        self,
        messages: list[BaseMessage],
        config: dict,
        generated: list[BaseMessage],
    ) -> AsyncIterator[BrainEvent]:
        """Walk the LangGraph event stream and re-emit as BrainEvents.

        Handles two ways models can emit tool calls:

        1. Structured ``tool_calls`` field on the AIMessage (the OpenAI-style
           happy path). LangChain recognises it; we re-emit as ``tool_start``
           events and emit a ``retract`` so the UI can clear any streamed
           chatter from the previous round.

        2. Raw JSON tool call serialised into ``content`` text (what small
           local models like qwen2.5-coder:3b do — the streamed ``content``
           chunks are literally characters of ``{"name": ..., "arguments":
           ...}``). We accumulate, recognise, and translate the same way.
        """
        # Re-inject the workspace on every run because set_workspace is
        # module-global and other brains (in tests) may have stomped it.
        set_workspace(self.workspace)
        content_buf: list[str] = []
        async for event in self._agent.astream_events(
            {"messages": messages}, config=config, version="v2"
        ):
            kind = event.get("event")
            data = event.get("data") or {}
            if kind == "on_chat_model_stream":
                chunk = data.get("chunk")
                delta = ""
                if chunk is not None:
                    c = getattr(chunk, "content", "")
                    delta = c if isinstance(c, str) else " ".join(str(p) for p in c)
                if delta:
                    content_buf.append(delta)
                    yield BrainEvent("token", {"data": delta})
            elif kind == "on_chat_model_end":
                msg = data.get("output")
                if msg is not None:
                    generated.append(msg)
                    full_content = "".join(content_buf).strip()
                    content_buf = []
                    structured_calls = getattr(msg, "tool_calls", None) or []
                    text_calls = []
                    if not structured_calls and full_content and _looks_like_tool_call_json(full_content):
                        try:
                            parsed = json.loads(full_content)
                            if isinstance(parsed, dict):
                                text_calls = [
                                    {
                                        "name": parsed.get("name"),
                                        "args": parsed.get("arguments")
                                        if "arguments" in parsed
                                        else parsed.get("args", {}),
                                    }
                                ]
                        except json.JSONDecodeError:
                            pass
                    if structured_calls or text_calls:
                        # Tool-call chatter — retract anything we streamed so
                        # the UI presents a clean slate for the next round.
                        yield BrainEvent("retract", {"reason": "tool_call"})
                        for tc in structured_calls:
                            yield BrainEvent(
                                "tool_start",
                                {
                                    "name": tc.get("name"),
                                    "args": tc.get("args", {}),
                                },
                            )
                        for tc in text_calls:
                            yield BrainEvent(
                                "tool_start",
                                {"name": tc.get("name"), "args": tc.get("args") or {}},
                            )
            elif kind == "on_tool_end":
                out = data.get("output")
                if out is not None:
                    generated.append(out)
                    # ToolMessage.content is what the agent will see next.
                    content = getattr(out, "content", "")
                    text = content if isinstance(content, str) else " ".join(str(p) for p in content)
                    yield BrainEvent(
                        "tool_end",
                        {
                            "name": getattr(out, "name", "tool"),
                            "output": text[:4000],  # cap UI payload
                        },
                    )
                    partial = []
            elif kind == "on_tool_end":
                out = data.get("output")
                if out is not None:
                    generated.append(out)
                    # ToolMessage.content is what the agent will see next.
                    content = getattr(out, "content", "")
                    text = content if isinstance(content, str) else " ".join(str(p) for p in content)
                    yield BrainEvent(
                        "tool_end",
                        {
                            "name": getattr(out, "name", "tool"),
                            "output": text[:4000],  # cap UI payload
                        },
                    )

    def _trim_history(self) -> None:
        while len(self._history) > MAX_HISTORY_MESSAGES:
            self._history.pop(0)
        # History must start with a Human turn so the ReAct prompt is well-formed.
        while self._history and not isinstance(self._history[0], HumanMessage):
            self._history.pop(0)


__all__ = ["CodeBrain", "BrainEvent", "MAX_TOOL_ROUNDS"]
