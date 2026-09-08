"""Timeout / stop tracking for Telegram handlers. No network needed."""

import asyncio
import os

from langchain_core.messages import HumanMessage

import main


def test_respond_timeout_default_and_override(monkeypatch):
    monkeypatch.delenv("TELEGRAM_RESPOND_TIMEOUT", raising=False)
    assert main._respond_timeout() == 90.0
    monkeypatch.setenv("TELEGRAM_RESPOND_TIMEOUT", "45")
    assert main._respond_timeout() == 45.0
    monkeypatch.setenv("TELEGRAM_RESPOND_TIMEOUT", "bogus")
    assert main._respond_timeout() == 90.0
    monkeypatch.setenv("TELEGRAM_RESPOND_TIMEOUT", "1")
    assert main._respond_timeout() == 10.0  # floor


def test_inflight_track_cancel_untrack():
    chat_id = 999001
    main._inflight.pop(chat_id, None)

    async def go():
        assert main._track_inflight(chat_id) is not None
        assert chat_id in main._inflight
        # A second task simulating /stop from another update cancels us.
        async def stopper():
            await asyncio.sleep(0.05)
            return main._cancel_inflight(chat_id)

        cancelled = await stopper()
        assert cancelled == 1
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            main._untrack_inflight(chat_id)
            raise
        return False

    task = asyncio.run(_run_cancelled(go()))
    assert task == "cancelled"
    assert chat_id not in main._inflight


async def _run_cancelled(coro):
    task = asyncio.ensure_future(coro)

    async def canceller():
        await asyncio.sleep(0.2)
        if not task.done():
            task.cancel()

    # go() self-cancels via _cancel_inflight; just await and map outcome.
    try:
        await task
        return "finished"
    except asyncio.CancelledError:
        return "cancelled"


def test_pop_last_exchange_rollback():
    import core

    key = "test-timeout-rollback"
    core.chat_histories.pop(key, None)
    core.chat_histories[key] = [HumanMessage(content="pending q")]
    assert core.pop_last_exchange(key) is True
    assert core.chat_histories[key] == []
    # Nothing to pop -> False, and AIMessage tail is left alone.
    assert core.pop_last_exchange(key) is False
    core.chat_histories.pop(key, None)
