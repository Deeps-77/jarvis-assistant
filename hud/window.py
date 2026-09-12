"""HUD main window: persona stage, transcript, mic, model/mode controls.

All blocking work (mic I/O除外) runs in QThreads; the GUI thread only
paints and shuffles signals. Constructor is side-effect free (no network,
no model loads) so offscreen unit tests can instantiate it.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import (
    QBuffer,
    QByteArray,
    QPointF,
    QRectF,
    Qt,
    QThread,
    QTimer,
    pyqtSignal,
    QObject,
)
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSlider,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .persona import PersonaState
from .workers import MicWorker, TurnWorker

logger = logging.getLogger(__name__)


class OrbWidget(QWidget):
    """Animated persona orb: halo + spinning rings + particles + waveform."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.persona = PersonaState()
        self.setMinimumSize(220, 220)
        self._levels: list[float] = []
        self._particles: list[list[float]] = []  # [angle, radius_frac, speed, life]
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(16)  # ~60fps

    def _tick(self) -> None:
        import random

        glow = self.persona.tick(0.016)
        del glow
        self._levels.append(self.persona.level)
        del self._levels[:-36]
        active = self.persona.state in ("listening", "speaking", "endpointing")
        if active and random.random() < 0.25 and len(self._particles) < 60:
            self._particles.append(
                [random.uniform(0, 6.283), 0.30, random.uniform(0.15, 0.5), 1.0]
            )
        alive = []
        for angle, radius, speed, life in self._particles:
            radius += speed * 0.016
            life -= 0.02
            if life > 0 and radius < 0.95:
                alive.append([angle + speed * 0.03, radius, speed, life])
        self._particles = alive
        self.update()

    def _accent(self) -> tuple[int, int, int]:
        import hud.theme as theme

        hexcol = theme.ACCENT.lstrip("#")
        try:
            return int(hexcol[0:2], 16), int(hexcol[2:4], 16), int(hexcol[4:6], 16)
        except ValueError:
            return self.persona.color()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        self.render(p, rect.width(), rect.height())

    def render(self, p, width: int, height: int) -> None:
        """Paint the orb onto any painter (widget or test QImage)."""
        import math

        import hud.theme as theme
        from PyQt6.QtGui import QPolygonF

        p.fillRect(0, 0, width, height, QColor(theme.BG))
        cx, cy = width / 2, (height - 34) / 2
        base = min(width, height - 34) / 2 - 10
        if base <= 0:
            return
        r, g, b = self._accent()
        glow = self.persona.glow()
        phase = self.persona._phase

        # Hexagonal Frame (Jarvis style)
        hex_r = base * 1.15
        poly = QPolygonF()
        for i in range(6):
            angle = math.pi / 2 + i * math.pi / 3
            poly.append(QPointF(cx + hex_r * math.cos(angle), cy + hex_r * math.sin(angle)))
        p.setPen(QPen(QColor(r, g, b, int(40 + 40 * glow)), 1.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPolygon(poly)

        # Halo: stacked faint rings breathing with glow.
        for i in range(self.persona.rings()):
            frac = 1.55 - i * 0.09
            alpha = max(0, min(255, int(glow * 34 * (1.0 - i / 8))))
            p.setPen(QPen(QColor(r, g, b, alpha), 1.5))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(
                QRectF(cx - base * frac, cy - base * frac, base * frac * 2, base * frac * 2)
            )
        # Spinning arc rings (faster when thinking/speaking).
        for idx, (frac, ring_w, arc_len) in enumerate(((0.86, 3, 110), (0.68, 2, 80), (0.52, 1, 55))):
            p.setPen(QPen(QColor(r, g, b, max(0, min(255, int(60 + 150 * glow)))), ring_w))
            p.setBrush(Qt.BrushStyle.NoBrush)
            start = int((phase * (1.0 + idx * 0.4) + idx * 2.1) * 57.3 * 16)
            rr = base * frac
            p.drawArc(QRectF(cx - rr, cy - rr, rr * 2, rr * 2), start, int(arc_len * 16))
        # Core orb.
        for i, frac in enumerate((1.0, 0.72, 0.45)):
            alpha = int(40 + 150 * glow * (0.4 + 0.3 * i))
            p.setBrush(QColor(r, g, b, max(0, min(255, alpha))))
            p.setPen(Qt.PenStyle.NoPen)
            radius = base * 0.42 * frac * (0.92 + 0.16 * glow)
            p.drawEllipse(int(cx - radius), int(cy - radius), int(radius * 2), int(radius * 2))
        # Particles drifting outward.
        for angle, radius, _speed, life in self._particles[:self.persona.particles()]:
            alpha = max(0, min(255, int(life * 220)))
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(r, g, b, alpha))
            px, py = cx + math.cos(angle) * base * radius, cy + math.sin(angle) * base * radius
            p.drawEllipse(QPointF(px, py), 2.5, 2.5)
        # Waveform bars from recent mic/playback energy.
        n, bw = 32, 7
        total_w = n * bw
        x0 = cx - total_w / 2
        wy = cy + base + 8
        for i, level in enumerate(self._levels[-n:]):
            hgt = max(2.0, min(26.0, level * 30.0))
            p.fillRect(QRectF(x0 + i * bw, wy + 26 - hgt, bw - 1, hgt), QColor(r, g, b, 200))


class _FnWorker(QObject):
    """One-shot: run a sync/async callable in a thread, emit result."""

    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn, parent=None) -> None:
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            result = self._fn()
            self.done.emit(result)
        except Exception as e:
            logger.exception("HUD background task failed")
            self.failed.emit(str(e)[:300])


def _collect_mic_sample(seconds: float = 2.5) -> dict:
    """One-shot mic probe (no signals needed): open, poll-read, close.

    Returns {"ok", "device", "rate", "bytes", "peak"} where peak is RMS
    int16 energy. Used by the self-test button for a plain-English diagnosis.
    """
    import struct
    import time

    from PyQt6.QtCore import QThread
    from PyQt6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices

    dev = QMediaDevices.defaultAudioInput()
    if dev.isNull():
        logger.warning("HUD mic probe: no input device found")
        return {"ok": False, "reason": "no input device found"}
    logger.info("HUD mic probe: sampling %s @ %gs", dev.description(), seconds)
    fmt = QAudioFormat()
    fmt.setSampleRate(16000)
    fmt.setChannelCount(1)
    fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
    if not dev.isFormatSupported(fmt):
        fmt = dev.nearestFormat(fmt)
    src = QAudioSource(dev, fmt)
    io = src.start()
    buf = bytearray()
    peak = 0.0
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < seconds:
            QThread.msleep(50)
            avail = io.bytesAvailable()
            if avail > 0:
                # NB: QIODevice.read(n) returns bytes in PyQt6 (no .data()),
                # while readAll() returns QByteArray (has .data()).
                chunk = io.read(avail)
                chunk = bytes(chunk.data()) if hasattr(chunk, "data") else bytes(chunk)
                buf.extend(chunk)
                n = len(chunk) // 2
                if n:
                    total = 0
                    for (sample,) in struct.iter_unpack("<h", chunk[: n * 2]):
                        total += sample * sample
                    peak = max(peak, (total / n) ** 0.5)
    finally:
        try:
            src.stop()
        except Exception:
            pass
    logger.info("HUD mic probe done: %d bytes, peak %.0f", len(buf), peak)
    return {
        "ok": True,
        "device": dev.description(),
        "rate": fmt.sampleRate(),
        "bytes": len(buf),
        "peak": round(peak, 1),
    }


def _run_in_thread(host: QWidget, fn, on_done, on_failed=None) -> QThread:
    """Run fn() in a throwaway QThread owned by host. Returns the thread."""
    thread = QThread(host)
    worker = _FnWorker(fn)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.done.connect(on_done)
    if on_failed is not None:
        worker.failed.connect(on_failed)
    worker.done.connect(thread.quit)
    worker.failed.connect(thread.quit)
    worker.done.connect(worker.deleteLater)
    worker.failed.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread


class MainWindow(QMainWindow):
    """Jarvis desktop HUD."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Jarvis")
        self.resize(900, 700)
        self._chat_key = "hud:main"
        self._mode = "normal"
        self._busy = False
        self._pending_pcm: list[bytes] = []  # mic audio collected while toggle is on
        self._pending_turns: list[tuple[str, str]] = []  # (text, lang) queued while busy
        self._live_text = ""  # streamed reply tokens for the live line
        self._audio_queue: list[bytes] = []
        self._audio_buf = None  # keep QBuffer alive during playback
        self._speaker_obj = None  # HudSpeaker, created lazily (imports speech)

        root = QWidget(self)
        layout = QVBoxLayout(root)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        left = QWidget(self)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        self.header = QLabel("J.A.R.V.I.S.", self)
        self.header.setObjectName("header")
        self.header.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        left_layout.addWidget(self.header)
        
        self.orb = OrbWidget(self)
        left_layout.addWidget(self.orb, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.status = QLabel("idle", self)
        self.status.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        left_layout.addWidget(self.status)
        self.live = QLabel("", self)
        self.live.setObjectName("live")
        self.live.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        self.live.setWordWrap(True)
        left_layout.addWidget(self.live)

        self.vu = QProgressBar(self)
        self.vu.setRange(0, 100)
        self.vu.setTextVisible(False)
        self.vu.setMaximumHeight(10)
        left_layout.addWidget(self.vu)

        self.metrics = QLabel("", self)
        self.metrics.setObjectName("metrics")
        self.metrics.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        left_layout.addWidget(self.metrics)

        self.cam_view = None  # QVideoWidget, created on first camera toggle
        splitter.addWidget(left)

        self.transcript = QTextEdit(self)
        self.transcript.setReadOnly(True)
        splitter.addWidget(self.transcript)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 460])
        layout.addWidget(splitter, stretch=1)

        row = QHBoxLayout()
        self.model_combo = QComboBox(self)
        self.mode_combo = QComboBox(self)
        self.mode_combo.addItems(["normal", "quick", "research", "docs"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        row.addWidget(QLabel("Model:", self))
        row.addWidget(self.model_combo, stretch=1)
        row.addWidget(QLabel("Mode:", self))
        row.addWidget(self.mode_combo)
        layout.addLayout(row)

        entry = QHBoxLayout()
        self.mic_btn = QPushButton("🎤 Talk", self)
        self.mic_btn.setCheckable(True)
        self.mic_btn.toggled.connect(self._on_mic_toggled)
        self.text_in = QLineEdit(self)
        self.text_in.setPlaceholderText("Type here, or press Talk and speak…")
        self.text_in.returnPressed.connect(self._on_send_text)
        self.send_btn = QPushButton("Send", self)
        self.send_btn.clicked.connect(self._on_send_text)
        entry.addWidget(self.mic_btn)
        entry.addWidget(self.text_in, stretch=1)
        entry.addWidget(self.send_btn)
        layout.addLayout(entry)

        tune = QHBoxLayout()
        tune.addWidget(QLabel("Accent:", self))
        self.accent_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.accent_slider.setRange(0, 359)
        self.accent_slider.setValue(187)  # default cyan
        self.accent_slider.setMaximumWidth(160)
        self.accent_slider.valueChanged.connect(self._on_accent_changed)
        tune.addWidget(self.accent_slider)
        tune.addWidget(QLabel("VAD:", self))
        self.vad_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.vad_slider.setRange(100, 3000)
        self.vad_slider.setSingleStep(50)
        self.vad_slider.setValue(500)
        self.vad_slider.setMaximumWidth(120)
        self.vad_slider.setToolTip("Mic sensitivity: lower hears quieter speech")
        self.vad_slider.valueChanged.connect(self._on_vad_changed)
        tune.addWidget(self.vad_slider)
        self.test_btn = QPushButton("🔧 Self-test", self)
        self.test_btn.setToolTip("Probe mic + speaker, report plain-English results")
        self.test_btn.clicked.connect(self._on_self_test)
        tune.addWidget(self.test_btn)
        self.cam_btn = QPushButton("📷 Camera: off", self)
        self.cam_btn.setCheckable(True)
        self.cam_btn.toggled.connect(self._on_camera_toggled)
        tune.addWidget(self.cam_btn)
        tune.addStretch()
        layout.addLayout(tune)

        self.setCentralWidget(root)

        self._metrics_timer = QTimer(self)
        self._metrics_timer.timeout.connect(self._refresh_metrics)
        self._metrics_timer.start(1500)
        self._camera = None
        self._capture = None

        # Mic capture (main-thread, signal driven).
        self.mic = MicWorker(self)
        self.mic.level.connect(self._on_mic_level)
        self.mic.utterance.connect(self._on_mic_utterance)
        self.vad_slider.setValue(int(self.mic._taker.config.threshold))

        # Playback chain for EdgeTTS MP3 / Piper WAV alike.
        self.player = QMediaPlayer(self)
        self._audio_out = QAudioOutput(self)
        self.player.setAudioOutput(self._audio_out)
        self.player.mediaStatusChanged.connect(self._on_media_status)
        self.player.errorOccurred.connect(self._on_player_error)

        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        
        from PyQt6.QtGui import QShortcut, QKeySequence
        QShortcut(QKeySequence("Space"), self, activated=self._on_shortcut_space)
        QShortcut(QKeySequence("Esc"), self, activated=self._on_shortcut_esc)
        
        self._set_state("idle")

    # ------------------------------------------------------------ UI state

    def _on_shortcut_space(self) -> None:
        if not self.text_in.hasFocus():
            self.mic_btn.toggle()
            
    def _on_shortcut_esc(self) -> None:
        if self.mic.recording:
            self.mic_btn.setChecked(False)

    def _set_state(self, state: str) -> None:
        self.orb.persona.set_state(state)
        self.status.setText(f"{state} · {self._mode} · {self.model_combo.currentText()}")

    def _say(self, who: str, text: str) -> None:
        self.transcript.append(f"<b>{who}:</b> {text}")

    def refresh_models(self, names: list[str]) -> None:
        import core

        if core.MODEL_NAME not in names:
            names = [core.MODEL_NAME] + list(names)
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(names)
        self.model_combo.setCurrentText(core.MODEL_NAME)
        self.model_combo.blockSignals(False)
        self._set_state(self.orb.persona.state)

    # ------------------------------------------------------------ controls

    def _on_mode_changed(self, mode: str) -> None:
        import core

        self._mode = core.normalize_chat_mode(mode)
        self._say("Jarvis", f"_Mode set to **{self._mode}**._")
        self._set_state(self.orb.persona.state)

    def _on_model_changed(self, name: str) -> None:
        import core

        from .loop import call as _call

        if not name or name == core.MODEL_NAME:
            return
        self._say("Jarvis", f"_Switching to `{name}`…_")

        def _switch():
            return _call(core.set_chat_model(name), timeout=120)

        def _done(old: object):
            self._say("Jarvis", f"✅ Switched `{old}` → `{name}`.")
            self._set_state(self.orb.persona.state)

        def _fail(err: str):
            self._say("Jarvis", f"⚠️ Switch failed: {err}")
            self.refresh_models([core.MODEL_NAME])

        _run_in_thread(self, _switch, _done, _fail)

    def _on_mic_toggled(self, on: bool) -> None:
        if on:
            self._pending_pcm.clear()
            self.live.setText("")
            if self.mic.start():
                self.mic_btn.setText("■ Stop")
                self._set_state("listening")
            else:
                self.mic_btn.setChecked(False)
                self._say("Jarvis", "⚠️ Mic unavailable — check microphone access.")
        else:
            self.mic.stop()  # flush emits a tail utterance when one exists
            self.mic_btn.setText("🎤 Talk")
            # The flush is empty exactly when collection worked (endpoints
            # already consumed the taker buffer), so drain explicitly.
            # Otherwise the pending audio strands and the turn never starts.
            self._finish_collected()

    def _on_mic_level(self, level: float) -> None:
        self.vu.setValue(int(level * 100))
        self.orb.persona.set_level(level)

    def _on_send_text(self) -> None:
        from .router import guess_text_lang

        text = self.text_in.text().strip()
        if not text:
            return
        self.text_in.clear()
        self._say("You", text)
        self._start_turn(text, guess_text_lang(text))

    # ------------------------------------------------------------ turn flow

    def _on_mic_utterance(self, pcm: bytes) -> None:
        # Push-to-talk-collect: while the toggle is on, pauses must NOT split
        # the query — accumulate everything, process once on toggle-off.
        self._pending_pcm.append(bytes(pcm))
        if self.mic.recording:
            secs = sum(len(c) for c in self._pending_pcm) / 32000.0
            self.status.setText(f"collecting… {secs:.1f}s (pauses won't split)")
            return
        self._finish_collected()

    def _finish_collected(self) -> None:
        """Process all collected mic audio as one utterance, if any."""
        if not self._pending_pcm:
            return
        pcm = b"".join(self._pending_pcm)
        self._pending_pcm.clear()
        logger.info("HUD collected turn (%d bytes)", len(pcm))
        self._process_utterance(pcm)

    def _process_utterance(self, pcm: bytes) -> None:
        from voice_server.stt import wav_bytes

        from .loop import call as _call
        from .stt import transcribe_with_lang

        if not pcm:
            self._set_state("listening" if self.mic.recording else "idle")
            return
        logger.info("HUD utterance queued for STT (%d bytes)", len(pcm))
        self._set_state("endpointing")
        self.status.setText("transcribing…")
        wav = wav_bytes(bytes(pcm))

        def _stt():
            # Silero validation lives here (worker thread): first use loads
            # torch and must never freeze the GUI thread. NOTE: never touch
            # Qt widgets from this thread — GUI updates are UB and crash
            # intermittently. Stage visibility comes from the GUI-thread
            # status set before spawn ("transcribing…") plus INFO logs.
            try:
                from voice_server.vad import has_speech
                ok = has_speech(bytes(pcm))
            except Exception:
                logger.debug("HUD silero gate failed open", exc_info=True)
                ok = True
            logger.info("HUD silero verdict: %s", "speech" if ok else "non-speech")
            if not ok:
                return (None, "")  # silent drop: no transcript spam
            logger.info("HUD whisper starting")
            try:
                out = _call(transcribe_with_lang(wav), timeout=180)
                logger.info(
                    "HUD whisper done: %d chars",
                    len(out[0] if out and out[0] else "")
                )
                return out
            except TimeoutError:
                logger.error("HUD whisper TIMED OUT after 180s — model may be stuck")
                raise
            except Exception:
                logger.exception("HUD whisper call failed")
                raise

        def _done(result: object):
            text, lang = result if isinstance(result, tuple) else ("", "")
            if text is None:
                # Silero non-speech gate: stay quiet, keep listening.
                self._set_state("listening" if self.mic.recording else "idle")
                return
            if not (text or "").strip():
                self._say("Jarvis", "_Didn't catch that — try again._")
                self._set_state("listening" if self.mic.recording else "idle")
                return
            self._say("You", f"{text} `[{lang or '?'}]`")
            self._start_turn(text, lang or "en")

        def _fail(err: str):
            logger.error("HUD STT thread failed: %s", err)
            self._say("Jarvis", f"⚠️ Transcription failed: {err}")
            self._set_state("idle")

        _run_in_thread(self, _stt, _done, _fail)

    def _get_speaker(self):
        if self._speaker_obj is None:
            from .tts import HudSpeaker

            self._speaker_obj = HudSpeaker()
        return self._speaker_obj

    def _start_turn(self, text: str, lang: str) -> None:
        if self._busy:
            # Queue instead of dropping: drained as one combined turn when
            # the current turn finishes.
            self._pending_turns.append((text, lang))
            self._say("Jarvis", f"_Still answering — queued ({len(self._pending_turns)})._")
            return
        self._begin_turn(text, lang)

    def _begin_turn(self, text: str, lang: str) -> None:
        self._busy = True
        self._live_text = ""
        self.live.setText("")
        self._set_state("thinking")
        # Worker lives on the GUI thread; its coroutine runs on the shared
        # loop (hud.loop), so no per-turn QThread/asyncio.run anymore.
        worker = TurnWorker(self._get_speaker(), self._chat_key, self._mode, self)
        worker.token.connect(self._on_turn_token)
        worker.notice.connect(lambda msg: self._say("Jarvis", f"_{msg}_"))
        worker.audio.connect(self._on_turn_audio)
        worker.reply.connect(self._on_turn_reply)
        worker.state.connect(self._set_state)
        worker.finished.connect(lambda: self._on_turn_done(worker))
        worker.start_turn(text, lang)

    def _on_turn_token(self, token: str) -> None:
        self._live_text += token
        tail = self._live_text.strip()[-160:]
        self.live.setText(f"“{tail}…”" if tail else "")

    def _on_turn_reply(self, msg: str) -> None:
        self._live_text = ""
        self.live.setText("")
        self._say("Jarvis", msg)

    def _on_turn_done(self, worker: QObject) -> None:
        self._busy = False
        worker.deleteLater()
        if self._pending_turns:
            items = list(self._pending_turns)
            self._pending_turns.clear()
            joined = "\n".join(t for t, _ in items)
            self._begin_turn(joined, items[0][1])
            return
        if not self._audio_queue and not self._playing():
            self._set_state("listening" if self.mic.recording else "idle")
            self.orb.persona.set_level(0.0)
            self.vu.setValue(0)

    # ------------------------------------------------------------ extras

    def _on_vad_changed(self, value: int) -> None:
        self.mic._taker.config.threshold = float(value)
        logger.debug("HUD VAD threshold tuned to %d", value)

    def _on_player_error(self, error, error_string: str = "") -> None:
        # Surface silent playback failures (the "no voice output" class).
        try:
            desc = str(error_string or error)
        except Exception:
            desc = "unknown playback error"
        logger.warning("HUD playback error: %s", desc)
        self._say("Jarvis", f"🔇 Speaker problem: {desc}")

    def _on_self_test(self) -> None:
        self._say("Jarvis", "_Self-test running: 2.5s mic probe, then a speaker check…_")

        def _test():
            from .loop import call as _call

            sample = _collect_mic_sample(2.5)
            wav = _call(self._get_speaker().synthesize("Speaker test.", "en"), timeout=60)
            return sample, bytes(wav or b"")

        def _done(result: object):
            sample, wav = result
            thr = self.mic._taker.config.threshold
            if not sample.get("ok"):
                self._say("Jarvis", f"🎤 Mic: **FAILED** — {sample.get('reason')}.")
            elif sample["bytes"] == 0:
                self._say("Jarvis", "🎤 Mic: opened but delivered **no data** — check Windows sound settings / default input.")
            else:
                verdict = "heard you ✅" if sample["peak"] >= thr else "too quiet for the VAD ⚠️ — lower the VAD slider"
                self._say(
                    "Jarvis",
                    f"🎤 Mic `{sample['device']}` @ {sample['rate']}Hz: "
                    f"{sample['bytes']} bytes, peak {sample['peak']} (gate {thr:.0f}) — {verdict}.",
                )
            if wav:
                self._say("Jarvis", "_Playing speaker check…_")
                self._on_turn_audio(wav)
            else:
                self._say("Jarvis", "🔇 Speaker: synthesis returned nothing — check Piper setup.")

        def _fail(err: str):
            self._say("Jarvis", f"⚠️ Self-test failed: {err}")

        _run_in_thread(self, _test, _done, _fail)

    def _on_accent_changed(self, hue: int) -> None:
        import hud.theme as theme
        from PyQt6.QtWidgets import QApplication

        theme.set_accent_hue(hue)
        theme.apply_theme(QApplication.instance())
        self.orb.update()

    def _refresh_metrics(self) -> None:
        from .metrics import format_snapshot, snapshot

        try:
            self.metrics.setText(format_snapshot(snapshot()))
        except Exception:
            pass

    def _on_camera_toggled(self, on: bool) -> None:
        if not on:
            try:
                if self._camera is not None:
                    self._camera.stop()
            except Exception:
                pass
            self._camera = None
            self._capture = None
            if self.cam_view is not None:
                self.cam_view.hide()
            self.cam_btn.setText("📷 Camera: off")
            return
        try:
            from PyQt6.QtMultimedia import QCamera, QMediaCaptureSession, QMediaDevices
            from PyQt6.QtMultimediaWidgets import QVideoWidget

            inputs = QMediaDevices.videoInputs()
            if not inputs:
                self.cam_btn.setChecked(False)
                self._say("Jarvis", "_No camera found on this machine._")
                return
            if self.cam_view is None:
                self.cam_view = QVideoWidget(self)
                self.cam_view.setMinimumHeight(160)
                # Insert above the transcript-adjacent left pane bottom.
                self.centralWidget().layout().insertWidget(1, self.cam_view)
            self._camera = QCamera(inputs[0])
            self._capture = QMediaCaptureSession()
            self._capture.setCamera(self._camera)
            self._capture.setVideoOutput(self.cam_view)
            self.cam_view.show()
            self._camera.start()
            self.cam_btn.setText("📷 Camera: on")
        except Exception as e:
            logger.exception("camera start failed")
            self.cam_btn.setChecked(False)
            self._say("Jarvis", f"⚠️ Camera failed: {e}")

    # ------------------------------------------------------------ playback

    def _playing(self) -> bool:
        from PyQt6.QtMultimedia import QMediaPlayer

        return self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    def _on_turn_audio(self, data: bytes) -> None:
        if data:
            self._audio_queue.append(bytes(data))
            if not self._playing():
                self._play_next()

    def _play_next(self) -> None:
        if not self._audio_queue:
            return
        chunk = self._audio_queue.pop(0)
        self._audio_buf = QBuffer(self)
        self._audio_buf.setData(QByteArray(chunk))
        self._audio_buf.open(QBuffer.OpenModeFlag.ReadOnly)
        from PyQt6.QtMultimedia import QMediaPlayer

        self.player.setSourceDevice(self._audio_buf)
        self.player.play()
        self._set_state("speaking")

    def _on_media_status(self, status) -> None:
        from PyQt6.QtMultimedia import QMediaPlayer

        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            if self._audio_queue:
                self._play_next()
            elif not self._busy:
                self._set_state("listening" if self.mic.recording else "idle")


__all__ = ["MainWindow", "OrbWidget"]
