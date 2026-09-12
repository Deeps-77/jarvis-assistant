"""Shared asyncio loop for the HUD process.

Why this exists: every HUD turn used to call ``asyncio.run()`` in a fresh
QThread, i.e. a brand-new event loop per turn. The Ollama httpx client and
its connection pool belong to the shared global ``llm`` — pool teardown
callbacks then landed on an already-closed loop and every voice/text turn
died with ``RuntimeError: Event loop is closed``. One persistent loop for
the app lifetime (like uvicorn gives the :8600 console) makes pool and
loop agree forever. It also gives future barge-in a cancellable task.

Qt signals remain the thread-safe bridge back to the GUI thread.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import threading
from concurrent.futures import Future

logger = logging.getLogger(__name__)

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_lock = threading.Lock()


def _serve(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    loop.run_forever()


def ensure() -> asyncio.AbstractEventLoop:
    """Return the shared loop, starting its thread on first use."""
    global _loop, _thread
    with _lock:
        if _loop is not None and _loop.is_running():
            return _loop
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=_serve, args=(loop,), name="hud-asyncio", daemon=True
        )
        thread.start()
        # Wait until run_forever is actually spinning.
        spin = 0
        while not loop.is_running() and spin < 500:
            import time

            time.sleep(0.005)
            spin += 1
        _loop = loop
        _thread = thread
        return loop


def submit(coro) -> Future:
    """Schedule a coroutine on the shared loop from any thread."""
    loop = ensure()
    return asyncio.run_coroutine_threadsafe(coro, loop)


def call(coro, timeout: float | None = None):
    """Blocking call: run coro on the shared loop and return its result."""
    return submit(coro).result(timeout)


def shutdown() -> None:
    """Stop the shared loop. Idempotent; safe to call twice."""
    global _loop, _thread
    with _lock:
        loop, thread = _loop, _thread
        _loop, _thread = None, None
    if loop is not None and loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
    if thread is not None and thread.is_alive():
        thread.join(timeout=5)


atexit.register(shutdown)


__all__ = ["ensure", "submit", "call", "shutdown"]
