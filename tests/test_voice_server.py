"""voice_server: sentence pipeline, endpoint validation, WS loop, barge-in.

Everything external (mic, LLM, Piper, Silero download) is faked or
monkeypatched — no network, no GPU needed.
"""

import asyncio
import math
import struct

import voice_server.tts as tts_mod
from voice_server.protocol import server_msg
from voice_server.tts import SentenceSynth, split_sentences


def test_split_sentences():
    assert split_sentences("Hello world. How are you? Fine!") == [
        "Hello world.", "How are you?", "Fine!",
    ]
    assert split_sentences("no punctuation") == ["no punctuation"]
    assert split_sentences("   ") == []


def test_server_msg_rejects_unknown():
    import pytest

    with pytest.raises(ValueError):
        server_msg("nope")


class FakeSpeaker:
    def __init__(self, delays=None):
        self.calls = []
        self.delays = delays or {}

    async def synthesize(self, text):
        self.calls.append(text)
        await asyncio.sleep(self.delays.get(text, 0))
        return b"WAV:" + text.encode()


def test_prefetch_emits_in_order_despite_timing():
    async def go():
        out = []

        async def cb(i, w):
            out.append((i, w))

        synth = SentenceSynth(FakeSpeaker(delays={"First.": 0.05}), cb)
        for tok in "First. Second. Third.".split(" "):
            await synth.feed_token(tok + " ")
        await synth.drain()
        return [i for i, _ in out]

    assert asyncio.run(go()) == [0, 1, 2]


def test_cancel_drops_pending():
    async def go():
        out = []
        synth = SentenceSynth(FakeSpeaker(), lambda i, w: out.append(i))
        await synth.feed_token("Hello world. ")
        synth.cancel()
        await synth.drain()
        return out

    assert asyncio.run(go()) == []


def test_endpoint_validated_before_stt(monkeypatch):
    from voice_server import vad as vad_mod
    from voice import VoiceConfig, VoiceTurnTaker

    calls = []
    monkeypatch.setattr(vad_mod, "has_speech", lambda pcm, **kw: calls.append(True) or True)
    ep = vad_mod.VoiceEndpoint(VoiceTurnTaker(VoiceConfig(input_rate=16000)))
    n = int(16000 * 0.03)
    loud = struct.pack("<%dh" % n, *[8000] * n)
    quiet = b"\x00" * len(loud)
    assert ep.feed(loud * 20) is None
    out = ep.feed(quiet * 40)
    assert out and calls  # endpointed AND silero-validated


def test_endpoint_rejects_non_speech(monkeypatch):
    from voice_server import vad as vad_mod
    from voice import VoiceConfig, VoiceTurnTaker

    monkeypatch.setattr(vad_mod, "has_speech", lambda pcm, **kw: False)
    ep = vad_mod.VoiceEndpoint(VoiceTurnTaker(VoiceConfig(input_rate=16000)))
    n = int(16000 * 0.03)
    loud = struct.pack("<%dh" % n, *[8000] * n)
    quiet = b"\x00" * len(loud)
    ep.feed(loud * 20)
    assert ep.feed(quiet * 40) is None


def _tone_16k(ms, hz=440, amp=8000):
    n = int(16000 * ms / 1000)
    return struct.pack("<%dh" % n, *[int(amp * math.sin(2 * math.pi * hz * i / 16000)) for i in range(n)])


def _silence_16k(ms=1200):
    return b"\x00" * int(16000 * ms / 1000) * 2


def _ws_client(monkeypatch):
    """TestClient over a voice app with STT/LLM/TTS all faked."""
    from starlette.testclient import TestClient

    import core
    import voice_server.stt as stt_mod
    import voice_server.vad as vad_mod

    async def fake_transcribe(wav, filename="u.wav"):
        return "hello jarvis"

    async def fake_respond(session_key, text, owner=None, on_token=None,
                           on_retry=None, ephemeral=False):
        assert ephemeral and session_key.startswith("voice:")
        for tok in ["Hi ", "there."]:
            if on_token:
                await on_token(tok)
        return "Hi there.", [], False

    class FakeSpeaker:
        async def synthesize(self, text):
            return b"WAV:" + text.encode()[:20]

    monkeypatch.setattr(stt_mod, "transcribe_utterance", fake_transcribe)
    monkeypatch.setattr(core, "respond", fake_respond)
    monkeypatch.setattr(vad_mod, "has_speech", lambda pcm, **kw: True)

    import voice_server.server as server_mod

    app = server_mod.create_app()
    # Lifespan doesn't run outside a context manager; inject the fake directly.
    app.state.speaker = FakeSpeaker()
    return TestClient(app)


def test_ws_full_turn_mp(monkeypatch):
    client = _ws_client(monkeypatch)
    with client.websocket_connect("/ws/voice") as ws:
        assert ws.receive_json()["type"] == "state"
        ws.send_json({"type": "start"})
        assert ws.receive_json() == {"type": "state", "state": "listening"}
        # speak: tone then trailing silence to endpoint
        ws.send_bytes(_tone_16k(800))
        for _ in range(6):
            ws.send_bytes(_silence_16k(300))
        kinds = []
        got_audio, got_reply = False, False
        for _ in range(30):
            msg = ws.receive()
            if "bytes" in msg:
                got_audio = True
            else:
                import json

                data = json.loads(msg["text"])
                kinds.append(data["type"])
                if data["type"] == "reply":
                    got_reply = True
                    assert "Hi there" in data["text"]
                    break
        assert got_audio, f"no audio frames, saw {kinds}"
        assert got_reply


def test_ws_barge_and_text(monkeypatch):
    client = _ws_client(monkeypatch)
    with client.websocket_connect("/ws/voice") as ws:
        ws.receive_json()  # idle
        ws.send_json({"type": "start"})
        ws.receive_json()  # listening
        ws.send_json({"type": "barge"})  # no active turn: harmless
        assert ws.receive_json() == {"type": "state", "state": "listening"}
        ws.send_json({"type": "text", "text": "good morning"})
        kinds = []
        for _ in range(30):
            msg = ws.receive()
            if "bytes" in msg:
                kinds.append("audio")
            else:
                import json

                kinds.append(json.loads(msg["text"])["type"])
            if "reply" in kinds:
                break
        assert "reply" in kinds, kinds


def test_no_chainlit_in_voice_path():
    import pathlib

    for path in (pathlib.Path("voice_server")).rglob("*.py"):
        src = path.read_text(encoding="utf-8")
        assert "import chainlit" not in src and "from chainlit" not in src, path
