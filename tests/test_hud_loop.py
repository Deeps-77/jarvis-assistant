"""Shared HUD asyncio loop: identity, results, errors, shutdown.

The whole point: Ollama's pooled httpx client must share one loop for the
app lifetime. Per-turn ``asyncio.run`` closed the loop under the pool and
killed every HUD turn with ``RuntimeError: Event loop is closed``.
"""

import asyncio
import pathlib

from hud import loop as loop_mod


def test_shared_loop_reused():
    a = loop_mod.ensure()
    b = loop_mod.ensure()
    assert a is b
    assert a.is_running()


async def _whoami():
    return asyncio.get_running_loop()


def test_submit_runs_on_shared_loop():
    loop = loop_mod.ensure()
    assert loop_mod.submit(_whoami()).result(timeout=10) is loop


def test_call_returns_and_propagates():
    async def ok():
        await asyncio.sleep(0)
        return 42

    assert loop_mod.call(ok(), timeout=10) == 42

    async def boom():
        raise ValueError("nope")

    try:
        loop_mod.call(boom(), timeout=10)
    except ValueError as e:
        assert str(e) == "nope"
    else:
        raise AssertionError("exception did not propagate")


def test_shutdown_idempotent_and_restartable():
    loop_mod.ensure()
    loop_mod.shutdown()
    loop_mod.shutdown()  # second call must not raise
    fresh = loop_mod.ensure()
    assert fresh.is_running()


def test_no_asyncio_run_in_voice_path():
    """TurnWorker/STT/switch must never create per-turn loops again."""
    root = pathlib.Path(__file__).resolve().parent.parent / "hud"
    for name in ("workers.py", "window.py"):
        src = (root / name).read_text(encoding="utf-8")
        assert "asyncio.run(" not in src, f"{name} still uses asyncio.run"


def test_turn_worker_uses_shared_loop(monkeypatch):
    import hud.workers as workers_mod

    calls = {}

    def fake_submit(coro):
        calls["loop"] = loop_mod.ensure()
        coro.close()
        import concurrent.futures

        fut: concurrent.futures.Future = concurrent.futures.Future()
        fut.set_result(None)
        return fut

    # start_turn does "from .loop import submit", so patch the loop module.
    monkeypatch.setattr(loop_mod, "submit", fake_submit)

    class FakeSpeaker:
        async def synthesize(self, text):
            return b""

    w = workers_mod.TurnWorker(FakeSpeaker(), "hud:test", "normal")
    w.start_turn("hi", "en")
    assert calls.get("loop") is loop_mod.ensure()
    w.cancel()  # completed future: harmless no-op
