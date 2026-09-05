"""Voice mode: VAD endpointing, Piper graceful-degradation, ephemeral turns.

All offline, no Ollama/mic needed (synthetic PCM, mocked agent/memory).
"""

import asyncio
import math
import struct

import speech
from speech import PiperSpeaker, _clean_for_speech, piper_available
from voice import VoiceConfig, VoiceTurnTaker


RATE = 48000  # browser mic delivery rate


def _cfg(**kw):
    base = dict(
        input_rate=RATE, frame_ms=30, threshold=500.0,
        silence_ms=300, max_turn_ms=30000, min_speech_ms=200,
    )
    base.update(kw)
    return VoiceConfig(**base)


def _tone(ms, amp=8000, hz=440, rate=RATE):
    n = int(rate * ms / 1000)
    return struct.pack("<%dh" % n, *[int(amp * math.sin(2 * math.pi * hz * i / rate)) for i in range(n)])


def _silence(ms=600, rate=RATE):
    return b"\x00" * int(rate * ms / 1000) * 2


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


def _dominant_hz(pcm, rate):
    import struct as _st

    n = len(pcm) // 2
    samples = [s / 32768.0 for (s,) in _st.iter_unpack("<h", pcm[: n * 2])]
    # zero-crossing estimate (fine for a pure test tone)
    crossings = sum(1 for a, b in zip(samples, samples[1:]) if (a < 0) != (b < 0))
    seconds = n / rate
    return crossings / 2 / seconds if seconds else 0.0


def test_resample_48k_to_16k():
    from voice import TARGET_RATE, resample_pcm

    assert TARGET_RATE == 16000
    tone48 = _tone(600, rate=48000)
    out = resample_pcm(tone48, 48000, 16000)
    assert len(out) == len(tone48) // 3  # exact 3:1 fast path
    assert abs(_dominant_hz(out, 16000) - 440) < 30  # pitch preserved


def test_resample_passthrough_and_empty():
    from voice import resample_pcm

    tone16 = _tone(200, rate=16000)
    assert resample_pcm(tone16, 16000, 16000) == tone16
    assert resample_pcm(b"", 48000, 16000) == b""


def test_voice_threads_hidden_and_purged(tmp_path):
    import asyncio as _asyncio

    from chainlit.types import Pagination, ThreadFilter

    from datalayer import SQLiteDataLayer

    layer = SQLiteDataLayer(tmp_path / "t.db")

    async def go():
        await layer.update_thread("voice-1", user_id="u", metadata={"is_voice": True})
        await layer.update_thread("chat-1", user_id="u", metadata={})

    _asyncio.run(go())

    async def listed():
        page = await layer.list_threads(Pagination(first=10), ThreadFilter(userId="u"))
        return [t["id"] for t in page.data]

    ids = _asyncio.run(listed())
    assert "chat-1" in ids and "voice-1" not in ids

    async def purged():
        return await layer.purge_voice_threads()

    assert _asyncio.run(purged()) == 1
    assert _asyncio.run(listed()) == ["chat-1"]


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
