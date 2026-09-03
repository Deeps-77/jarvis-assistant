"""Token observation for the code assistant.

A LangChain ``BaseCallbackHandler`` that:

- Reads ``(prompt_tokens, completion_tokens)`` from any LLM response, using
  :meth:`LLMProvider.extract_usage` so Ollama and OpenAI are handled by the
  same code path.
- Estimates USD cost via :func:`llm_provider.estimate_cost_usd`.
- Appends every turn to a per-session JSONL file (``logs/code_tokens.jsonl``)
  so a UI page can rebuild a full history without holding everything in RAM.
- Exposes running totals (input / output / turns / cost) for live display.

Usage:

    from llm_provider import LLMConfig, build_provider
    from token_usage import TokenTracker

    cfg = LLMConfig.from_env()
    tracker = TokenTracker(session_id="user:abc", provider=cfg.provider, model=cfg.model)
    provider, model = build_provider(cfg, callbacks=[tracker])
    await model.ainvoke("hello")
    print(tracker.snapshot())

The handler is safe to share across many concurrent invocations on the same
session; a lock guards the in-memory counters and the append-mode JSONL handle.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from llm_provider import LLMProvider, estimate_cost_usd

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path(__file__).parent / "logs" / "code_tokens.jsonl"


@dataclass(slots=True)
class TurnUsage:
    """One LLM call's worth of usage, persisted as one JSONL row."""

    session_id: str
    ts: float
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    duration_ms: int = 0
    label: str = ""  # optional UI-side tag, e.g. "plan" / "build" / "summarize"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TokenTotals:
    """Running aggregate over a session."""

    input_tokens: int = 0
    output_tokens: int = 0
    turns: int = 0
    cost_usd: float = 0.0
    first_ts: float = 0.0
    last_ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TokenTracker(BaseCallbackHandler):
    """Records per-turn token usage and keeps live totals.

    Pass an instance to ``build_provider(config, callbacks=[tracker])`` and the
    handler will be invoked on every LLM call made by any LangChain component
    bound to the resulting chat model (including tool-calling rounds in a
    ReAct agent).
    """

    # LangChain BaseCallbackHandler metadata — keep this honest so the
    # framework doesn't rate-limit or duplicate events.
    run_inline = False

    def __init__(
        self,
        session_id: str,
        provider: str,
        model: str,
        log_path: Path | None = None,
        label: str = "",
    ) -> None:
        super().__init__()
        self.session_id = session_id
        self.provider = provider
        self.model = model
        self.label = label
        self.log_path = Path(log_path) if log_path else DEFAULT_LOG_PATH
        self._lock = threading.Lock()
        self._turn_starts: dict[str, float] = {}
        self.totals = TokenTotals()
        self._turns: list[TurnUsage] = []
        self._max_memory = int(os.environ.get("CODE_TOKEN_HISTORY", "1000"))
        self._ensure_log_file()

    def _ensure_log_file(self) -> None:
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.log_path.exists():
                self.log_path.touch()
        except OSError as e:
            logger.warning("TokenTracker cannot open %s: %s", self.log_path, e)

    # ------------------------------------------------------------------ utils

    def snapshot(self) -> dict[str, Any]:
        """Cheap, lock-free summary for the UI to display every turn."""
        with self._lock:
            t = self.totals
            avg_in = t.input_tokens // t.turns if t.turns else 0
            avg_out = t.output_tokens // t.turns if t.turns else 0
            return {
                "session_id": self.session_id,
                "provider": self.provider,
                "model": self.model,
                "turns": t.turns,
                "input_tokens": t.input_tokens,
                "output_tokens": t.output_tokens,
                "total_tokens": t.input_tokens + t.output_tokens,
                "avg_input": avg_in,
                "avg_output": avg_out,
                "cost_usd": round(t.cost_usd, 6),
                "first_ts": t.first_ts,
                "last_ts": t.last_ts,
            }

    def history(self, limit: int = 100) -> list[dict[str, Any]]:
        """Last ``limit`` turns as plain dicts (most recent first)."""
        with self._lock:
            rows = [asdict(t) for t in self._turns[-limit:]]
        rows.reverse()
        return rows

    def reset(self) -> None:
        """Clear in-memory counters. JSONL file is preserved for audit."""
        with self._lock:
            self.totals = TokenTotals()
            self._turns.clear()

    # ------------------------------------------------------- callback hooks

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        run_id = self._run_id(kwargs)
        self._turn_starts[run_id] = time.perf_counter()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        run_id = self._run_id(kwargs)
        start = self._turn_starts.pop(run_id, None)
        duration_ms = int((time.perf_counter() - start) * 1000) if start else 0

        input_tokens, output_tokens = LLMProvider.extract_usage(response)
        if not input_tokens and not output_tokens:
            # No metadata available (some local servers omit counts). Skip but
            # log so silent gaps are visible in diagnostics.
            logger.debug("No token usage reported for run %s", run_id)
            return

        cost = estimate_cost_usd(self.provider, self.model, input_tokens, output_tokens)
        turn = TurnUsage(
            session_id=self.session_id,
            ts=time.time(),
            provider=self.provider,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
            duration_ms=duration_ms,
            label=self.label,
        )

        with self._lock:
            t = self.totals
            t.input_tokens += input_tokens
            t.output_tokens += output_tokens
            t.cost_usd += cost
            t.turns += 1
            if not t.first_ts:
                t.first_ts = turn.ts
            t.last_ts = turn.ts
            self._turns.append(turn)
            if len(self._turns) > self._max_memory:
                # Drop oldest from memory; JSONL log still has the full record.
                del self._turns[: len(self._turns) - self._max_memory]

        self._append_jsonl(turn)

    def on_llm_error(self, error: BaseException | Exception, **kwargs: Any) -> None:
        run_id = self._run_id(kwargs)
        self._turn_starts.pop(run_id, None)
        logger.debug("LLM error in run %s: %s", run_id, error)

    # ----------------------------------------------------------- async mirror

    async def on_llm_end_async(self, response: LLMResult, **kwargs: Any) -> None:
        # LangChain invokes the sync handler when ``run_inline=False`` even from
        # async contexts; this hook is here for explicit async pipelines.
        self.on_llm_end(response, **kwargs)

    # -------------------------------------------------------------- internals

    @staticmethod
    def _run_id(kwargs: dict[str, Any]) -> str:
        run = kwargs.get("run_id")
        if run is None:
            return "_anon"
        return getattr(run, "uuid", None) or str(run)

    def _append_jsonl(self, turn: TurnUsage) -> None:
        try:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(turn.to_dict(), ensure_ascii=False) + "\n")
        except OSError as e:
            logger.warning("TokenTracker append failed: %s", e)


__all__ = ["TokenTracker", "TokenTotals", "TurnUsage", "DEFAULT_LOG_PATH"]
