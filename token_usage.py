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

from llm_provider import CODEX_TIER_LABEL, LLMProvider, cloud_equivalent_cost_usd, estimate_cost_usd
from paths import LOG_DIR

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = LOG_DIR / "code_tokens.jsonl"


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
            cloud = cloud_equivalent_cost_usd(t.input_tokens, t.output_tokens)
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
                "cloud_cost_usd": round(cloud, 6),
                "saved_usd": round(max(cloud - t.cost_usd, 0.0), 6),
                "first_ts": t.first_ts,
                "last_ts": t.last_ts,
            }

    def lifetime_saved_usd(self) -> float:
        """Total API spend avoided across all sessions in the JSONL log.

        Sums the Codex-tier equivalent per row minus recorded actual cost.
        Estimate only (server-reported counts); local power/hardware excluded.
        """
        if not self.log_path.exists():
            return 0.0
        saved = 0.0
        try:
            with self.log_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    try:
                        in_tok = int(row.get("input_tokens", 0))
                        out_tok = int(row.get("output_tokens", 0))
                        actual = float(row.get("cost_usd", 0.0))
                    except (TypeError, ValueError):
                        continue
                    saved += max(cloud_equivalent_cost_usd(in_tok, out_tok) - actual, 0.0)
        except OSError as e:
            logger.warning("TokenTracker lifetime_saved read failed: %s", e)
            return 0.0
        return round(saved, 6)

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

    def rehydrate(self, session_id: str) -> int:
        """Rebuild in-memory totals from JSONL rows for ``session_id``.

        Used when reopening a saved chat so ``/usage`` continues where the
        session left off. Returns the number of rows loaded.
        """
        loaded = 0
        if not self.log_path.exists():
            return 0
        try:
            with self.log_path.open("r", encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh if line.strip()]
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("TokenTracker rehydrate read failed: %s", e)
            return 0
        with self._lock:
            self.totals = TokenTotals()
            self._turns.clear()
            for row in rows:
                if not isinstance(row, dict) or row.get("session_id") != session_id:
                    continue
                try:
                    turn = TurnUsage(
                        session_id=session_id,
                        ts=float(row.get("ts", 0)),
                        provider=str(row.get("provider", "")),
                        model=str(row.get("model", "")),
                        input_tokens=int(row.get("input_tokens", 0)),
                        output_tokens=int(row.get("output_tokens", 0)),
                        cost_usd=float(row.get("cost_usd", 0.0)),
                        duration_ms=int(row.get("duration_ms", 0)),
                        label=str(row.get("label", "") or ""),
                    )
                except (TypeError, ValueError):
                    continue
                t = self.totals
                t.input_tokens += turn.input_tokens
                t.output_tokens += turn.output_tokens
                t.cost_usd += turn.cost_usd
                t.turns += 1
                if not t.first_ts:
                    t.first_ts = turn.ts
                t.last_ts = turn.ts
                self._turns.append(turn)
                loaded += 1
            if len(self._turns) > self._max_memory:
                del self._turns[: len(self._turns) - self._max_memory]
        return loaded

    def purge_session(self, session_id: str) -> int:
        """Delete all JSONL rows for ``session_id`` + reset matching memory.

        Used by session delete (history + usage both purged). Returns the
        number of rows removed. Other sessions' rows are preserved via an
        atomic rewrite.
        """
        removed = 0
        if self.log_path.exists():
            try:
                with self.log_path.open("r", encoding="utf-8") as fh:
                    lines = fh.readlines()
            except OSError as e:
                logger.warning("TokenTracker purge read failed: %s", e)
                lines = []
            kept: list[str] = []
            for line in lines:
                try:
                    row = json.loads(line) if line.strip() else None
                except json.JSONDecodeError:
                    kept.append(line)
                    continue
                if isinstance(row, dict) and row.get("session_id") == session_id:
                    removed += 1
                else:
                    kept.append(line)
            if removed:
                try:
                    tmp = self.log_path.with_suffix(".jsonl.tmp")
                    tmp.write_text("".join(kept), encoding="utf-8")
                    tmp.replace(self.log_path)
                except OSError as e:
                    logger.warning("TokenTracker purge rewrite failed: %s", e)
                    removed = 0
        with self._lock:
            if self.session_id == session_id:
                self.totals = TokenTotals()
                self._turns.clear()
        return removed

    def daily_summary(self, days: int = 7) -> list[dict[str, Any]]:
        """Read the JSONL log and aggregate usage per local day.

        Returns a list of dicts, most recent day first, each with::

            {
                "date": "2026-09-03",
                "turns": N,
                "input_tokens": N,
                "output_tokens": N,
                "cost_usd": float,
                "models": {"qwen2.5-coder:3b": {"turns": N, "tokens": N, "cost_usd": float}, ...}
            }

        Reads the on-disk log so totals survive restarts. Caps each
        model's per-day token bucket to avoid unbounded memory.
        """
        import datetime as _dt
        import collections

        if not self.log_path.exists():
            return []
        rows_by_day: dict[str, dict[str, Any]] = {}
        models_by_day: dict[str, dict[str, dict[str, Any]]] = {}
        try:
            with self.log_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts = float(row.get("ts", 0))
                    if not ts:
                        continue
                    day = _dt.datetime.fromtimestamp(ts).astimezone().strftime("%Y-%m-%d")
                    model = str(row.get("model", "unknown"))
                    in_tok = int(row.get("input_tokens", 0))
                    out_tok = int(row.get("output_tokens", 0))
                    cost = float(row.get("cost_usd", 0.0))

                    bucket = rows_by_day.setdefault(day, {
                        "date": day,
                        "turns": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cost_usd": 0.0,
                        "models": {},
                    })
                    bucket["turns"] += 1
                    bucket["input_tokens"] += in_tok
                    bucket["output_tokens"] += out_tok
                    bucket["cost_usd"] += cost
                    m_bucket = bucket["models"].setdefault(model, {
                        "turns": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                    })
                    m_bucket["turns"] += 1
                    m_bucket["input_tokens"] += in_tok
                    m_bucket["output_tokens"] += out_tok
                    m_bucket["cost_usd"] += cost
                    bucket["cost_usd"] = round(bucket["cost_usd"], 6)
                    m_bucket["cost_usd"] = round(m_bucket["cost_usd"], 6)
        except OSError as e:
            logger.warning("TokenTracker daily_summary read failed: %s", e)
            return []

        # Sort by date desc and cap to ``days``
        ordered = sorted(rows_by_day.values(), key=lambda r: r["date"], reverse=True)
        return ordered[:days]

    def render_usage_card(self, history_limit: int = 10) -> str:
        """Render a human-readable usage card for ``/usage``.

        Includes:
        - Session totals
        - Daily roll-up (last 7 days from JSONL)
        - Last N turns as a markdown table
        """
        snap = self.snapshot()
        lines: list[str] = []
        lines.append("### Token usage — current session")
        lines.append(f"- Session: `{snap['session_id']}`")
        lines.append(f"- Model: `{snap['provider']}:{snap['model']}`")
        lines.append(f"- Turns: **{snap['turns']:,}**")
        lines.append(f"- Input: **{snap['input_tokens']:,}**  ·  Output: **{snap['output_tokens']:,}**  ·  Total: **{snap['total_tokens']:,}**")
        lines.append(f"- Avg in/out per turn: **{snap['avg_input']:,}** / **{snap['avg_output']:,}**")
        lines.append(f"- Est. cost: **${snap['cost_usd']:.4f}**")
        lines.append(
            f"- 💰 API spend avoided (this session, vs {CODEX_TIER_LABEL}): "
            f"**${snap['saved_usd']:.4f}** (est.)"
        )
        lifetime = self.lifetime_saved_usd()
        if lifetime > 0:
            lines.append(f"- 💰 Lifetime avoided (all sessions): **${lifetime:.4f}** (est.)")
        if snap["first_ts"]:
            import datetime as _dt
            f = _dt.datetime.fromtimestamp(snap["first_ts"]).astimezone().strftime("%Y-%m-%d %H:%M:%S")
            l = _dt.datetime.fromtimestamp(snap["last_ts"]).astimezone().strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"- First turn: {f}  ·  Last turn: {l}")

        # Daily summary
        daily = self.daily_summary(7)
        if daily:
            lines.append("")
            lines.append("### Last 7 days (from JSONL log)")
            lines.append("| Date | Turns | In | Out | Cost |")
            lines.append("|---|---:|---:|---:|---:|")
            for d in daily:
                lines.append(
                    f"| {d['date']} | {d['turns']} | {d['input_tokens']:,} | "
                    f"{d['output_tokens']:,} | ${d['cost_usd']:.4f} |"
                )

        # Recent turns
        history = self.history(history_limit)
        if history:
            lines.append("")
            lines.append(f"### Last {len(history)} turns")
            lines.append("| Time | In | Out | Cost | Duration | Label |")
            lines.append("|---|---:|---:|---:|---:|---|")
            import datetime as _dt
            for row in history:
                ts = row.get("ts", 0)
                when = _dt.datetime.fromtimestamp(ts).astimezone().strftime("%H:%M:%S") if ts else "?"
                in_t = int(row.get("input_tokens", 0))
                out_t = int(row.get("output_tokens", 0))
                cost = float(row.get("cost_usd", 0.0))
                dur = int(row.get("duration_ms", 0))
                label = row.get("label", "") or ""
                lines.append(f"| {when} | {in_t:,} | {out_t:,} | ${cost:.4f} | {dur}ms | {label} |")

        return "\n".join(lines)

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
