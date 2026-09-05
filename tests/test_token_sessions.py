"""TokenTracker session scope: purge isolation + rehydrate totals."""

import json
import time

from token_usage import TokenTracker, TurnUsage


def _append(tr, sid, n, it=10, ot=5):
    for _ in range(n):
        tr._append_jsonl(
            TurnUsage(
                session_id=sid,
                ts=time.time(),
                provider="ollama",
                model="m",
                input_tokens=it,
                output_tokens=ot,
                cost_usd=0.0,
            )
        )


def test_purge_keeps_other_sessions(tmp_path):
    log = tmp_path / "t.jsonl"
    tr = TokenTracker(session_id="code:AAA", provider="ollama", model="m", log_path=log)
    _append(tr, "code:AAA", 3)
    _append(tr, "code:BBB", 2)
    assert tr.purge_session("code:AAA") == 3
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all(r["session_id"] == "code:BBB" for r in rows)


def test_purge_resets_live_tracker_and_savings(tmp_path):
    log = tmp_path / "t.jsonl"
    tr = TokenTracker(session_id="code:AAA", provider="ollama", model="m", log_path=log)
    _append(tr, "code:AAA", 2)
    tr.rehydrate("code:AAA")
    assert tr.snapshot()["turns"] == 2
    assert tr.snapshot()["saved_usd"] > 0
    tr.purge_session("code:AAA")
    assert tr.snapshot()["turns"] == 0
    assert tr.lifetime_saved_usd() == 0.0


def test_rehydrate_rebuilds_totals(tmp_path):
    log = tmp_path / "t.jsonl"
    tr = TokenTracker(session_id="code:BBB", provider="ollama", model="m", log_path=log)
    _append(tr, "code:BBB", 2, it=10, ot=5)
    fresh = TokenTracker(session_id="code:BBB", provider="ollama", model="m", log_path=log)
    assert fresh.rehydrate("code:BBB") == 2
    snap = fresh.snapshot()
    assert (snap["turns"], snap["input_tokens"], snap["output_tokens"]) == (2, 20, 10)


def test_rehydrate_missing_log_is_zero(tmp_path):
    tr = TokenTracker(
        session_id="x", provider="ollama", model="m", log_path=tmp_path / "nope.jsonl"
    )
    assert tr.rehydrate("x") == 0
