"""Desktop HUD: persona, language routing, TTS switch, window smoke.

Qt runs offscreen — no display, GPU, Ollama, or network needed.
"""

import asyncio
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from hud.persona import STATES, PersonaState
from hud.router import (
    decide,
    guess_text_lang,
    needs_multilingual_model,
    suggest_tamil_model,
)


def test_persona_states_and_bounds():
    p = PersonaState()
    assert p.state == "idle"
    assert p.set_state("bogus") == "idle"
    for s in STATES:
        assert p.set_state(s) == s
        g = p.tick(0.05)
        assert 0.0 <= g <= 1.0
        assert len(p.color()) == 3
    assert p.set_level(99) == 1.0
    assert p.set_level(-1) == 0.0
    assert p.set_level("bogus") == 0.0
    assert p.tick(-5) >= 0.0


def test_persona_energy_brightens():
    p = PersonaState()
    p.set_state("listening")
    dim = PersonaState()
    dim.set_state("listening")
    dim._phase = p._phase
    p.set_level(1.0)
    assert p.glow() >= dim.glow()


def test_router_decide():
    assert decide("ta")["engine"] == "edge"
    assert decide("ta")["voice"] == "ta-IN-ValluvarNeural"
    assert decide("ta-IN")["engine"] == "edge"
    assert decide("en")["engine"] == "piper"
    assert decide("")["engine"] == "piper"


def test_guess_text_lang():
    assert guess_text_lang("வணக்கம் நண்பா") == "ta"
    assert guess_text_lang("hello there") == "en"
    assert guess_text_lang("") == "en"
    assert guess_text_lang("Price in சென்னை?") == "ta"


def test_needs_multilingual_model():
    assert needs_multilingual_model("ta", "hf.co/openbmb/MiniCPM5-2B-GGUF:Q4_K_M") is True
    assert needs_multilingual_model("ta", "qwen2.5:7b") is False
    assert needs_multilingual_model("ta", "hf.co/unsloth/gemma-4-E2B-it-GGUF:Q4_K_M") is False
    assert needs_multilingual_model("en", "MiniCPM") is False
    assert needs_multilingual_model("", "MiniCPM") is False


def test_suggest_tamil_model(monkeypatch):
    import core

    async def fake_list():
        return [{"name": "MiniCPM:2b"}, {"name": "qwen2.5:7b"}]

    monkeypatch.setattr(core, "list_ollama_models", fake_list)
    assert asyncio.run(suggest_tamil_model()) == "qwen2.5:7b"

    async def fake_both():
        # gemma wins over qwen even when listed later (measured better Tamil)
        return [{"name": "qwen3.5:2b"}, {"name": "gemma-4-E2B-it"}]

    monkeypatch.setattr(core, "list_ollama_models", fake_both)
    assert asyncio.run(suggest_tamil_model()) == "gemma-4-E2B-it"

    async def fake_none():
        return [{"name": "MiniCPM:2b"}]

    monkeypatch.setattr(core, "list_ollama_models", fake_none)
    assert asyncio.run(suggest_tamil_model()) is None


def test_hud_speaker_english_offline_chain(monkeypatch):
    """EN stays offline: Kokoro -> Piper -> pyttsx3. Edge must not run."""
    from hud import tts as tts_mod

    async def no_kokoro(text, lang="en-us"):
        return None

    monkeypatch.setattr(tts_mod, "_kokoro_synthesize", no_kokoro)

    calls = []

    class FakePiper:
        async def synthesize(self, text):
            calls.append(text)
            return b"WAV"

    class BoomComm:
        def __init__(self, *a, **k):
            raise AssertionError("edge must not run for English")
        async def save(self, path):
            pass

    import edge_tts

    monkeypatch.setattr(edge_tts, "Communicate", BoomComm)
    sp = tts_mod.HudSpeaker(piper=FakePiper())
    assert asyncio.run(sp.synthesize("hello", "en")) == b"WAV"
    assert calls == ["hello"]
    assert asyncio.run(sp.synthesize("   ", "en")) == b""


def test_hud_speaker_english_kokoro_first(monkeypatch):
    """Warm Kokoro serves English; Piper stays quiet on success."""
    from hud import tts as tts_mod

    async def fake_kokoro(text, lang="en-us"):
        return b"KOKORO"

    async def boom(*a, **k):
        raise AssertionError("Piper must not run when Kokoro succeeds")

    monkeypatch.setattr(tts_mod, "_kokoro_synthesize", fake_kokoro)
    monkeypatch.setattr(tts_mod, "_KOKORO_PIPELINES", {"a": object()})

    class FakePiper:
        async def synthesize(self, text):
            return await boom()

    sp = tts_mod.HudSpeaker(piper=FakePiper())
    assert asyncio.run(sp.synthesize("hello", "en")) == b"KOKORO"


def test_hud_speaker_piper_covers_cold_kokoro(monkeypatch):
    """Cold Kokoro must not stall sentence one: Piper answers instead."""
    from hud import tts as tts_mod

    async def boom_kokoro(text, lang="en-us"):
        raise AssertionError("cold Kokoro must not even be attempted")

    monkeypatch.setattr(tts_mod, "_kokoro_synthesize", boom_kokoro)
    monkeypatch.setattr(tts_mod, "_KOKORO_PIPELINES", {})

    class FakePiper:
        async def synthesize(self, text):
            return b"PIPER"

    sp = tts_mod.HudSpeaker(piper=FakePiper())
    assert asyncio.run(sp.synthesize("hello", "en")) == b"PIPER"


def test_warmup_kokoro_reports_readiness(monkeypatch):
    from hud import tts as tts_mod

    monkeypatch.setattr(tts_mod, "_KOKORO_PIPELINES", {})

    async def fake_build(text, lang="en-us"):
        tts_mod._KOKORO_PIPELINES["a"] = object()
        return b"WAV"

    monkeypatch.setattr(tts_mod, "_kokoro_synthesize", fake_build)
    assert asyncio.run(tts_mod.warmup_kokoro()) is True
    assert tts_mod.kokoro_ready() is True


def test_kokoro_pipeline_cached(monkeypatch):
    """Pipelines build once per language, not per sentence."""
    import sys
    import types

    import hud.tts as tts_mod

    builds = []

    class FakePipeline:
        def __init__(self, lang_code="a", device=None):
            builds.append((lang_code, device))

        def __call__(self, text, voice="af_heart", speed=1.0):
            import numpy as np

            yield None, None, np.zeros(100, dtype=np.float32)

    fake_mod = types.ModuleType("kokoro")
    fake_mod.KPipeline = FakePipeline
    monkeypatch.setitem(sys.modules, "kokoro", fake_mod)
    monkeypatch.setattr(tts_mod, "_KOKORO_PIPELINES", {})

    import soundfile

    def fake_write(buf, arr, rate, format=None):
        buf.write(b"WAV")

    monkeypatch.setattr(soundfile, "write", fake_write)

    assert asyncio.run(tts_mod._kokoro_synthesize("one", "en")) == b"WAV"
    assert asyncio.run(tts_mod._kokoro_synthesize("two", "en")) == b"WAV"
    assert builds == [("a", "cpu")]  # built once (CPU-pinned), reused


def test_hud_speaker_tamil_edge_and_fallback(monkeypatch):
    import edge_tts

    from hud import tts as tts_mod

    saved = []

    class FakeComm:
        def __init__(self, text, voice):
            saved.append((text, voice))

        async def save(self, path):
            with open(path, "wb") as fh:
                fh.write(b"MP3")

    monkeypatch.setattr(edge_tts, "Communicate", FakeComm)

    class FakePiper:
        def __init__(self):
            self.calls = []

        async def synthesize(self, text):
            self.calls.append(text)
            return b"PIPER"

    piper = FakePiper()
    sp = tts_mod.HudSpeaker(piper=piper)
    out = asyncio.run(sp.synthesize("வணக்கம்", "ta"))
    assert out == b"MP3"
    assert saved and saved[0][1] == "ta-IN-ValluvarNeural"

    class BoomComm:
        def __init__(self, *a, **k):
            pass

        async def save(self, path):
            raise RuntimeError("offline")

    monkeypatch.setattr(edge_tts, "Communicate", BoomComm)
    assert asyncio.run(sp.synthesize("வணக்கம்", "ta")) == b"PIPER"
    assert piper.calls and "internet" in piper.calls[-1].lower()


_APP = None


def _tone_pcm(ms=300, hz=440, rate=16000, amp=8000):
    import math as _math
    import struct as _st

    n = int(rate * ms / 1000)
    return _st.pack(
        "<%dh" % n,
        *[int(amp * _math.sin(2 * _math.pi * hz * i / rate)) for i in range(n)],
    )


def _qapp():
    """Process-wide QApplication, kept alive (dropping the last reference
    while widgets live crashes Qt at teardown)."""
    global _APP
    from PyQt6.QtWidgets import QApplication

    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def test_endpoint_auto_responds_while_recording(monkeypatch):
    """Hands-free: each endpointed utterance starts a turn immediately —
    no toggle-off needed. Busy turns queue via _start_turn."""
    _qapp()
    import hud.stt as stt_mod
    import hud.window as window_mod
    from hud.window import MainWindow

    seen_wavs = []

    async def fake_transcribe(wav, filename="hud.wav"):
        seen_wavs.append(bytes(wav))
        return ("hello world", "en")

    monkeypatch.setattr(stt_mod, "transcribe_with_lang", fake_transcribe)
    monkeypatch.setattr("voice_server.vad.has_speech", lambda pcm: True)

    def run_now(host, fn, on_done, on_failed=None):
        on_done(fn())  # synchronous: no Qt event loop in tests

    monkeypatch.setattr(window_mod, "_run_in_thread", run_now)

    begun = []
    win = MainWindow()
    try:
        win._begin_turn = lambda text, lang: begun.append((text, lang))
        win.mic.recording = True  # toggle held on: still answers per pause
        c1 = _tone_pcm()
        win._on_mic_utterance(c1)
        assert begun == [("hello world", "en")]
        assert win._pending_pcm == []
        assert len(seen_wavs) == 1
        assert len(seen_wavs[0]) >= len(c1)
    finally:
        win.close()
        win.deleteLater()


def test_toggle_off_with_empty_flush_is_quiet(monkeypatch):
    """Toggle-off with an empty taker flush must not crash or double-fire."""
    _qapp()
    import hud.window as window_mod
    from hud.window import MainWindow

    monkeypatch.setattr(
        window_mod, "_run_in_thread",
        lambda host, fn, on_done, on_failed=None: None,
    )

    win = MainWindow()
    try:
        begun = []
        win._begin_turn = lambda text, lang: begun.append((text, lang))
        win.mic.recording = False
        win._on_mic_toggled(False)  # real MicWorker.stop: empty flush, no-op
        assert begun == []
        assert win._pending_pcm == []
    finally:
        win.close()
        win.deleteLater()


def test_silero_drop_is_silent(monkeypatch):
    """Silero runs in the STT worker (never the GUI thread); non-speech
    drops without transcript spam."""
    _qapp()
    import hud.window as window_mod
    from hud.window import MainWindow

    monkeypatch.setattr("voice_server.vad.has_speech", lambda pcm: False)
    calls = []

    def run_now(host, fn, on_done, on_failed=None):
        calls.append(True)
        on_done(fn())

    monkeypatch.setattr(window_mod, "_run_in_thread", run_now)
    win = MainWindow()
    try:
        win._on_mic_utterance(_tone_pcm())
        assert calls  # STT worker ran (gate lives there now)
        plain = win.transcript.toPlainText()
        assert "Didn't catch" not in plain
        assert win._pending_turns == []
    finally:
        win.close()
        win.deleteLater()


def test_busy_queue_drains_joined(monkeypatch):
    _qapp()
    from PyQt6.QtCore import QObject

    from hud.window import MainWindow

    win = MainWindow()
    try:
        begun = []
        win._begin_turn = lambda text, lang: begun.append((text, lang))
        win._busy = True
        win._start_turn("first", "en")
        win._start_turn("second", "ta")
        assert len(win._pending_turns) == 2
        assert begun == []
        win._busy = False
        win._on_turn_done(QObject())
        assert begun == [("first\nsecond", "en")]
        assert win._pending_turns == []
    finally:
        win.close()
        win.deleteLater()


def test_token_streams_to_live_line():
    _qapp()
    from hud.window import MainWindow

    win = MainWindow()
    try:
        win._on_turn_token("Hello ")
        win._on_turn_token("there")
        assert "Hello" in win.live.text()
        win._on_turn_reply("Hello there")
        assert win.live.text() == ""
        assert "Hello there" in win.transcript.toPlainText()
    finally:
        win.close()
        win.deleteLater()


def test_theme_accent_and_stylesheet():
    import hud.theme as theme

    old_accent, old_dim = theme.ACCENT, theme.ACCENT_DIM
    try:
        assert theme.set_accent_hue(0).startswith("#")
        assert theme.ACCENT != old_accent
        assert theme.set_accent_hue("bogus") == theme.ACCENT  # invalid ignored
        css = theme.build_stylesheet()
        assert theme.ACCENT in css and "QMainWindow" in css
        _qapp()
        theme.apply_theme(__import__("PyQt6.QtWidgets", fromlist=["QApplication"]).QApplication.instance())
    finally:
        theme.ACCENT, theme.ACCENT_DIM = old_accent, old_dim


def test_metrics_snapshot_shape():
    from hud.metrics import Snapshot, format_snapshot, snapshot

    snap = snapshot()
    assert isinstance(snap, Snapshot)
    assert snap.gpu >= -1.0
    text = format_snapshot(snap)
    assert "CPU" in text and "MEM" in text and "GPU" in text and "NET" in text
    assert "N/A" in format_snapshot(Snapshot(cpu=1, mem=2, net_mbps=0.5, gpu=-1.0))


def test_window_layout_and_extras():
    _qapp()
    from PyQt6.QtWidgets import QSplitter

    from hud.window import MainWindow

    win = MainWindow()
    try:
        assert win.findChild(QSplitter) is not None
        assert win.cam_view is None  # camera never auto-starts
        # Accent slider recolors live.
        import hud.theme as theme

        before = theme.ACCENT
        try:
            win.accent_slider.setValue(0)
            assert theme.ACCENT != before
        finally:
            theme.set_accent_hue(187)
        # Camera toggle with no device: notice, no crash, stays off.
        from PyQt6.QtMultimedia import QMediaDevices

        win._on_camera_toggled(True)
        if not QMediaDevices.videoInputs():
            assert "No camera" in win.transcript.toPlainText()
            assert win.cam_view is None
        else:
            win._on_camera_toggled(False)
            assert win.cam_view is None or not win.cam_view.isVisible()
        win._refresh_metrics()
        assert "CPU" in win.metrics.text()
        # Paint onto a test image directly: exceptions surface as clean
        # failures (an exception escaping a real paintEvent aborts the
        # whole pytest process, and bare repaint() without a running event
        # loop paints nothing — this combo once hid a NameError).
        from PyQt6.QtGui import QImage, QPainter

        win.show()
        for state, level in (("idle", 0.0), ("speaking", 0.8), ("thinking", 0.3)):
            win.orb.persona.set_state(state)
            win.orb.persona.set_level(level)
            for _ in range(5):
                win.orb._tick()
            img = QImage(300, 300, QImage.Format.Format_ARGB32)
            img.fill(0)
            painter = QPainter(img)
            try:
                win.orb.render(painter, 300, 300)
            finally:
                painter.end()
            assert img.pixelColor(150, 100).alpha() > 0
    finally:
        try:
            win._on_camera_toggled(False)
        except Exception:
            pass
        win.close()
        win.deleteLater()


def test_vad_slider_retunes_taker():
    _qapp()
    from hud.window import MainWindow

    win = MainWindow()
    try:
        win.vad_slider.setValue(300)
        assert win.mic._taker.config.threshold == 300.0
    finally:
        win.close()
        win.deleteLater()


def test_player_error_surfaces():
    _qapp()
    from hud.window import MainWindow

    win = MainWindow()
    try:
        win._on_player_error(None, "boom")
        assert "Speaker problem" in win.transcript.toPlainText()
    finally:
        win.close()
        win.deleteLater()


def test_collect_mic_sample_never_raises():
    from hud.window import _collect_mic_sample

    result = _collect_mic_sample(0.3)
    assert set(result) >= {"ok"}
    if result["ok"]:
        assert {"device", "rate", "bytes", "peak"} <= set(result)


def test_self_test_reports_and_plays(monkeypatch):
    _qapp()
    import hud.window as window_mod
    from hud.window import MainWindow

    monkeypatch.setattr(
        window_mod,
        "_collect_mic_sample",
        lambda seconds=2.5: {
            "ok": True, "device": "FakeMic", "rate": 16000,
            "bytes": 32000, "peak": 1200.0,
        },
    )

    async def fake_synth(self, text, lang="en"):
        return b"FAKEWAV"

    from hud import tts as tts_mod

    monkeypatch.setattr(tts_mod.HudSpeaker, "synthesize", fake_synth)

    def run_now(host, fn, on_done, on_failed=None):
        on_done(fn())

    monkeypatch.setattr(window_mod, "_run_in_thread", run_now)

    win = MainWindow()
    try:
        played = []
        win._play_next = lambda: played.append(bytes(win._audio_queue.pop(0)))
        win._on_self_test()
        plain = win.transcript.toPlainText()
        assert "FakeMic" in plain and "heard you" in plain
        assert "Playing speaker check" in plain
        assert played == [b"FAKEWAV"]
    finally:
        win.close()
        win.deleteLater()


def test_self_test_quiet_mic_hint(monkeypatch):
    _qapp()
    import hud.window as window_mod
    from hud.window import MainWindow

    monkeypatch.setattr(
        window_mod,
        "_collect_mic_sample",
        lambda seconds=2.5: {
            "ok": True, "device": "QuietMic", "rate": 16000,
            "bytes": 32000, "peak": 50.0,
        },
    )

    async def fake_synth(self, text, lang="en"):
        return b""

    from hud import tts as tts_mod

    monkeypatch.setattr(tts_mod.HudSpeaker, "synthesize", fake_synth)
    monkeypatch.setattr(
        window_mod, "_run_in_thread", lambda host, fn, on_done, on_failed=None: on_done(fn())
    )

    win = MainWindow()
    try:
        win._on_self_test()
        plain = win.transcript.toPlainText()
        assert "too quiet" in plain and "VAD slider" in plain
        assert "returned nothing" in plain
    finally:
        win.close()
        win.deleteLater()


def test_run_in_thread_delivers():
    """Real QThread delivery (regression: an unparented worker was GC'd
    before started fired, so fn() silently never ran — no logs, no crash)."""
    _qapp()
    from PyQt6.QtCore import QEventLoop, QTimer

    import hud.window as window_mod
    from hud.window import MainWindow

    win = MainWindow()
    try:
        got = []
        # Drop all local refs immediately after spawning, exactly like
        # production call sites do (this is what exposed the GC bug).
        window_mod._run_in_thread(win, lambda: 41 + 1, got.append)
        # Pump GUI events until delivered or timeout.
        import time as _time

        from PyQt6.QtWidgets import QApplication

        t0 = _time.monotonic()
        while not got and _time.monotonic() - t0 < 5:
            QApplication.instance().processEvents()
            _time.sleep(0.01)
        assert got == [42]
    finally:
        win.close()
        win.deleteLater()
    _qapp()
    from hud.window import MainWindow

    win = MainWindow()
    try:
        win.refresh_models(["model-a:latest"])
        assert win.model_combo.count() >= 1
        win.mode_combo.setCurrentText("research")
        assert win._mode == "research"
        win._on_mode_changed("bogus")
        assert win._mode == "normal"  # invalid falls back
        win._set_state("thinking")
        assert win.status.text().startswith("thinking")
        win.orb.persona.tick(0.05)
        # Busy send is refused without spawning threads/Ollama.
        win._busy = True
        win.text_in.setText("hello")
        win._on_send_text()
        assert "Still answering" in win.transcript.toPlainText()
        win._busy = False
        win.text_in.setText("   ")
        win._on_send_text()  # empty: no-op, no crash
    finally:
        win.close()
        win.deleteLater()


def test_preload_voice_ml_idempotent():
    import sys

    from hud.app import preload_voice_ml

    preload_voice_ml()
    preload_voice_ml()  # second call harmless
    assert "faster_whisper" in sys.modules
    assert "faster_whisper.vad" in sys.modules


def test_concurrent_voice_imports_no_deadlock():
    """Regression: concurrent first-imports from worker threads tripped
    3.14 import-lock deadlock detection; the loser then latched STT off
    for the whole process (0 chars forever). Post-preload imports safe."""
    import sys
    import threading

    from hud.app import preload_voice_ml

    preload_voice_ml()
    errors: list = []

    def do_import():
        try:
            import faster_whisper.vad  # noqa: F401
            import edge_tts  # noqa: F401
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=do_import) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not [t for t in threads if t.is_alive()], "import hung"
    assert errors == []
    assert "faster_whisper.vad" in sys.modules


def test_turn_streams_tokens_into_synth(monkeypatch):
    """Regression: on_token must feed SentenceSynth or no voice ever plays."""
    import hud.loop as loop_mod
    import hud.workers as workers_mod

    async def fake_respond(session_key, text, owner=None, on_token=None,
                           on_retry=None, ephemeral=False, mode="normal"):
        for tok in ["Hello there. ", "How are you? "]:
            await on_token(tok)
        return "Hello there. How are you?", [], False

    import core

    monkeypatch.setattr(core, "respond", fake_respond)

    calls = []

    class FakeSpeaker:
        # Same interface as HudSpeaker: (text, lang).
        async def synthesize(self, text, lang="en"):
            calls.append((text, lang))
            return b"WAV:" + text.encode()[:10]

    heard = []
    worker = workers_mod.TurnWorker(FakeSpeaker(), "hud:test-synth", "normal")
    # DirectConnection: no Qt event loop runs in tests, so queued cross-thread
    # delivery would never fire (production GUI loop delivers normally).
    from PyQt6.QtCore import Qt

    worker.audio.connect(heard.append, Qt.ConnectionType.DirectConnection)
    loop_mod.call(worker._go(), timeout=30)
    assert calls == [("Hello there.", "en"), ("How are you?", "en")]
    assert len(heard) == 2
    assert all(h.startswith(b"WAV:") for h in heard)


def _fake_transcriber(first, forced=None):
    """Fake core.speech_transcriber. first=(text,lang,prob);
    forced maps language -> (text,lang,prob). Counts transcribe calls."""
    import asyncio
    from types import SimpleNamespace

    calls = []

    class FakeModel:
        def transcribe(self, tmp, **kwargs):
            calls.append(kwargs.get("language"))
            if kwargs.get("language") and forced and kwargs["language"] in forced:
                text, lang, prob = forced[kwargs["language"]]
            else:
                text, lang, prob = first
            segs = [SimpleNamespace(text=text)]
            info = SimpleNamespace(language=lang, language_probability=prob)
            return segs, info

    tr = SimpleNamespace(
        enabled=True, _model=FakeModel(), _transcribe_lock=asyncio.Lock()
    )
    return tr, calls


def test_stt_accepts_clean_ta_en(monkeypatch):
    import asyncio

    import core
    from hud import stt as stt_mod

    tr, calls = _fake_transcriber(("vanakkam", "ta", 0.92))
    monkeypatch.setattr(core, "speech_transcriber", tr)
    monkeypatch.setattr(core, "transcribe_audio", None, raising=False)
    text, lang = asyncio.run(stt_mod.transcribe_with_lang(b"\x01\x02" * 500))
    assert (text, lang) == ("vanakkam", "ta")
    assert calls == [None]  # single pass, no runoff


def test_stt_runoff_picks_ta_over_ml(monkeypatch):
    """Tamil heard as Malayalam (live case: ml 0.66) must resolve to ta."""
    import asyncio

    import core
    from hud import stt as stt_mod

    tr, calls = _fake_transcriber(
        ("malayalam garbage", "ml", 0.66),
        forced={"ta": ("vanakkam nanba", "ta", 0.81),
                "en": ("hello friend", "en", 0.22)},
    )
    monkeypatch.setattr(core, "speech_transcriber", tr)
    text, lang = asyncio.run(stt_mod.transcribe_with_lang(b"\x01\x02" * 500))
    assert lang == "ta"
    assert text == "vanakkam nanba"
    assert calls == [None, "ta", "en"]


def test_stt_runoff_picks_en(monkeypatch):
    import asyncio

    import core
    from hud import stt as stt_mod

    tr, calls = _fake_transcriber(
        ("bengali garbage", "bn", 0.39),
        forced={"ta": ("tamil garbage", "ta", 0.31),
                "en": ("what time is it", "en", 0.74)},
    )
    monkeypatch.setattr(core, "speech_transcriber", tr)
    text, lang = asyncio.run(stt_mod.transcribe_with_lang(b"\x01\x02" * 500))
    assert lang == "en"
    assert text == "what time is it"


def test_theme_has_tamil_font_fallback():
    import hud.theme as theme_mod

    css = theme_mod.build_stylesheet()
    assert "Nirmala UI" in css


def test_no_cpp_comments_in_hud():
    """Guard: `//` is not a Python comment — stray ones are SyntaxError
    risks (or dead text) that editors don't flag."""
    import pathlib
    import re

    bad = []
    for path in (pathlib.Path(__file__).resolve().parent.parent / "hud").glob("*.py"):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if re.match(r"\s*//", line):
                bad.append(f"{path.name}:{i}")
    assert bad == []


def test_chat_markdown_keeps_line_breaks_outside_pre():
    from hud.markdown import render_chat_html

    out = render_chat_html("Top picks of 2026:\n1. First film\n2. Second film")
    assert "20261." not in out  # the reported collapse
    assert "<br" in out
    assert "First film" in out and "Second film" in out


def test_chat_markdown_preserves_pre_blocks():
    from hud.markdown import render_chat_html

    out = render_chat_html("```py\nline1\nline2\n```")
    assert "<pre" in out
    # newlines inside <pre> stay literal for QTextEdit's monospace block
    inner = out.split("<pre", 1)[1]
    assert "\n" in inner


def test_say_bubbles_and_markdown():
    _qapp()
    from hud.window import MainWindow

    win = MainWindow()
    try:
        win._say("You", "hello there")
        html = win.transcript.toHtml()
        assert 'align="right"' in html
        assert "hello there" in win.transcript.toPlainText()
        win._say("Jarvis", "**bold** answer with `code`")
        html = win.transcript.toHtml()
        assert "#0d1b26" in html  # jarvis bubble (user bubble is #12324a)
        assert "bold" in win.transcript.toPlainText()
        # raw markdown syntax must not leak through rendered output
        assert "**bold**" not in html
        # single newlines (softbreaks) must break lines in Qt, not collapse
        win._say("Jarvis", "Top picks of 2026:\n1. First film\n2. Second film")
        html = win.transcript.toHtml().replace("<br />", "<br/>")
        assert "Top picks of 2026:<br/><br/>1. First film<br/>2. Second film" in html
        plain = win.transcript.toPlainText()
        assert "First film" in plain and "Second film" in plain
        win._say("Jarvis", "_ephemeral notice_")
        assert "ephemeral notice" in win.transcript.toPlainText()
    finally:
        win.close()
        win.deleteLater()


def test_audio_queue_and_media_status():
    _qapp()
    from hud.window import MainWindow

    win = MainWindow()
    try:
        win._busy = True  # stay busy so EndOfMedia doesn't flip state
        win._on_turn_audio(b"\x01\x02" * 100)
        # Either queued or already consumed by offscreen playback; both fine.
        from PyQt6.QtMultimedia import QMediaPlayer

        win._on_media_status(QMediaPlayer.MediaStatus.EndOfMedia)
        win._busy = False
        win._audio_queue.clear()
        win._on_media_status(QMediaPlayer.MediaStatus.EndOfMedia)
        assert win.status.text().startswith("idle")
    finally:
        win.close()
        win.deleteLater()
