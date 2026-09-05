"""Voice mode: VAD endpointing, Piper graceful-degradation, ephemeral turns.

All offline, no Ollama/mic needed (synthetic PCM, mocked agent/memory).
"""

import asyncio
import math
import struct

import speech
from speech import PiperSpeaker, _clean_for_speech, piper_available
from voice import VoiceConfig, VoiceTurnTaker


def _cfg(**kw):
    base = dict(
        sample_rate=24000, frame_ms=30, threshold=500.0,
        silence_ms=300, max_turn_ms=30000, min_speech_ms=200,
    )
    base.update(kw)
    return VoiceConfig(**base)


def _tone(ms, amp=8000, hz=440):
    n = int(24000 * ms / 1000)
    return struct.pack("<%dh" % n, *[int(amp * math.sin(2 * math.pi * hz * i / 24000)) for i in range(n)])


def _silence(ms=600):
    return b"\x00" * int(24000 * ms / 1000) * 2


def test_silence_yields_nothing():
    t = VoiceTurnTaker(_cfg())
    assert t.feed(_silence()) is None
    assert t.flush() is None


def test_speech_then_silence_endpoints():
    t = VoiceTurnTaker(_cfg())
    assert t.feed(_tone(600)) is None  # still speaking
    u = t.feed(_silence())
    assert u and len(u) > 0


def test_mic_bump_dropped():
    t = VoiceTurnTaker(_cfg())
    assert t.feed(_tone(50)) is None
    assert t.feed(_silence()) is None  # < min_speech_ms


def test_max_turn_force_cuts():
    t = VoiceTurnTaker(_cfg(max_turn_ms=900))
    u = t.feed(_tone(1500))
    assert u and len(u) > 0


def test_flush_collects_partial():
    t = VoiceTurnTaker(_cfg())
    t.feed(_tone(600))
    u = t.flush()
    assert u and len(u) > 0
    assert t.flush() is None  # state reset afterwards


def test_piper_missing_degrades_to_empty(monkeypatch):
    monkeypatch.setenv("VOICE_PIPER_EXE", r"C:\nonexistent\piper.exe")
    ok, reason = piper_available()
    assert not ok and "not found" in reason
    wav = asyncio.run(PiperSpeaker().synthesize("hello"))
    assert wav == b""


def test_speech_text_cleaned():
    out = _clean_for_speech("See [docs](http://x) and `code`.\n```py\nprint(1)\n``` Done!")
    assert "[" not in out and "```" not in out and "`" not in out
    assert "docs" in out and "Done!" in out
    long = "Sentence one. " + "x" * 2000
    assert len(_clean_for_speech(long)) <= speech.MAX_SYNTH_CHARS + 1
    assert _clean_for_speech("   ") == ""


def test_ephemeral_skips_memory():
    import core

    calls = []

    class FakeMemory:
        enabled = True

        async def search(self, *a):
            calls.append("search")
            return []

        async def learn_from_exchange(self, *a):
            calls.append("learn")

    async def fake_run_agent(messages, **kw):
        return "hello there", []

    old_mem, old_run, old_hist = core.memory_store, core.run_agent, dict(core.chat_histories)
    core.memory_store = FakeMemory()
    core.run_agent = fake_run_agent
    try:
        body, _, _ = asyncio.run(core.respond("voice:test", "hi", ephemeral=True))
        assert body == "hello there"
        assert calls == [], calls  # nothing retained
        body2, _, _ = asyncio.run(core.respond("k2", "hi"))
        assert calls == ["search", "learn"], calls  # normal path unaffected
    finally:
        core.memory_store, core.run_agent = old_mem, old_run
        core.chat_histories.clear()
        core.chat_histories.update(old_hist)
