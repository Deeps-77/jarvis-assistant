"""Qt workers: mic capture and chat turns off the GUI thread.

- :class:`MicWorker` — QAudioSource @16kHz mono int16 → VoiceTurnTaker
  (sub-frame buffering handles any chunk size) → level/utterance signals.
- :class:`TurnWorker` — one chat turn on the shared HUD asyncio loop
  (see :mod:`hud.loop`): STT text → language routing (Tamil model
  suggestion) → core.respond (persistent ``hud:*`` session, memory on)
  → sentence-prefetch TTS.
"""

from __future__ import annotations

import logging
import struct

from PyQt6.QtCore import QObject, pyqtSignal

logger = logging.getLogger(__name__)


def _rms_level(pcm: bytes) -> float:
    """0..1 energy estimate for VU/orb display."""
    n = len(pcm) // 2
    if n == 0:
        return 0.0
    total = 0
    for (sample,) in struct.iter_unpack("<h", pcm[: n * 2]):
        total += sample * sample
    rms = (total / n) ** 0.5
    return min(1.0, rms / 12000.0)


class MicWorker(QObject):
    """Microphone capture living on the Qt thread (signal-driven, no polling)."""

    level = pyqtSignal(float)
    utterance = pyqtSignal(bytes)
    started = pyqtSignal()
    stopped = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        from voice import VoiceConfig, VoiceTurnTaker

        self._taker = VoiceTurnTaker(VoiceConfig(input_rate=16000))
        self._audio = None
        self._io = None
        self._rate = 16000
        self.recording = False

    def start(self) -> bool:
        from PyQt6.QtMultimedia import QAudioFormat, QAudioSource, QMediaDevices

        if self.recording:
            return True
        fmt = QAudioFormat()
        fmt.setSampleRate(16000)
        fmt.setChannelCount(1)
        fmt.setSampleFormat(QAudioFormat.SampleFormat.Int16)
        try:
            device = QMediaDevices.defaultAudioInput()
            if device.isNull():
                logger.warning("mic start: no audio input device found")
                return False
            if not device.isFormatSupported(fmt):
                near = device.nearestFormat(fmt)
                logger.warning(
                    "16kHz mono int16 not supported; using %dHz ch=%d",
                    near.sampleRate(),
                    near.channelCount(),
                )
                fmt = near
            self._rate = fmt.sampleRate()
            self._audio = QAudioSource(device, fmt)
            self._io = self._audio.start()
            self._io.readyRead.connect(self._on_data)
            logger.info(
                "mic started: %s @ %dHz (VAD threshold %.0f)",
                device.description(),
                self._rate,
                self._taker.config.threshold,
            )
        except Exception:
            logger.exception("mic start failed")
            self._audio = None
            self._io = None
            return False
        self._taker.reset()
        self.recording = True
        self.started.emit()
        return True

    def _on_data(self) -> None:
        if not self.recording or self._io is None:
            return
        raw = bytes(self._io.readAll().data())
        if not raw:
            return
        if getattr(self, "_rate", 16000) != 16000:
            from voice import resample_pcm

            raw = resample_pcm(raw, self._rate, 16000)
            if not raw:
                return
        self.level.emit(_rms_level(raw))
        utterance = self._taker.feed(raw)
        if utterance:
            logger.info("mic utterance endpointed (%d bytes)", len(utterance))
            self.utterance.emit(bytes(utterance))

    def stop(self) -> None:
        """Stop capture; flush trailing speech as a final utterance, if any."""
        if not self.recording:
            return
        self.recording = False
        try:
            utterance = self._taker.flush()
        except Exception:
            utterance = None
        try:
            if self._audio is not None:
                self._audio.stop()
        except Exception:
            pass
        self._audio = None
        self._io = None
        if utterance:
            logger.info("mic flush collected %d bytes", len(utterance))
            self.utterance.emit(bytes(utterance))
        else:
            logger.info("mic stopped with no pending speech")
        self.stopped.emit()


class _LangSpeaker:
    """Adapt HudSpeaker to the SentenceSynth (text-only) interface."""

    def __init__(self, speaker, lang: str) -> None:
        self._speaker = speaker
        self._lang = lang

    async def synthesize(self, text: str) -> bytes:
        return await self._speaker.synthesize(text, self._lang)


class TurnWorker(QObject):
    """One HUD turn: respond → stream tokens → prefetch sentence audio.

    Lives on the GUI thread; the coroutine runs on the shared HUD asyncio
    loop (see :mod:`hud.loop`) so the Ollama connection pool always agrees
    with its loop. Signals cross threads via queued connections.
    """

    token = pyqtSignal(str)
    reply = pyqtSignal(str)
    audio = pyqtSignal(bytes)
    state = pyqtSignal(str)
    notice = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, speaker, chat_key: str, mode: str, parent=None) -> None:
        super().__init__(parent)
        self._speaker = speaker
        self._chat_key = chat_key
        self._mode = mode
        self._transcript = ""
        self._lang = "en"
        self._future = None

    def start_turn(self, transcript: str, lang: str = "en") -> None:
        from .loop import submit

        self._transcript = transcript
        self._lang = lang or "en"
        self._future = submit(self._go())

    def cancel(self) -> None:
        """Request cancellation (future barge-in). Best-effort."""
        fut, self._future = self._future, None
        if fut is not None and not fut.done():
            try:
                fut.cancel()
            except Exception:
                pass

    async def _go(self) -> None:
        import core

        from voice_server.tts import SentenceSynth

        from .router import decide, needs_multilingual_model, suggest_tamil_model

        route = decide(self._lang)
        self.state.emit("thinking")
        if needs_multilingual_model(self._lang, core.MODEL_NAME):
            suggestion = await suggest_tamil_model()
            if suggestion and suggestion != core.MODEL_NAME:
                try:
                    old = await core.set_chat_model(suggestion)
                    self.notice.emit(f"Tamil detected — model `{old}` → `{suggestion}`.")
                except Exception as e:
                    self.notice.emit(f"Model switch failed ({e}); continuing.")
        synth = SentenceSynth(_LangSpeaker(self._speaker, route["lang"]), self._on_sentence)
        streamed: list[str] = []

        async def on_token(token: str):
            streamed.append(token)
            self.token.emit(token)

        try:
            body, _sources, _failed = await core.respond(
                self._chat_key,
                self._transcript,
                owner="hud",
                on_token=on_token,
                mode=self._mode,
            )
        except Exception:
            logger.exception("HUD turn failed")
            body = "Sorry, something went wrong on my end."
            await synth.feed_token(body)
        finally:
            await synth.drain()
        full = "".join(streamed).strip() or body
        self.reply.emit(full)
        self.finished.emit()

    async def _on_sentence(self, _index: int, wav: bytes) -> None:
        if wav:
            self.audio.emit(bytes(wav))


__all__ = ["MicWorker", "TurnWorker", "_rms_level"]
