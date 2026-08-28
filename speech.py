import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path

import botlog

logger = logging.getLogger(__name__)


class SpeechTranscriber:
    def __init__(self):
        self.model_name = os.environ.get("WHISPER_MODEL", "small")
        self.device = os.environ.get("WHISPER_DEVICE", "auto")
        self.compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
        self.enabled = True
        self._model = None
        self._loaded = False
        self._load_lock = asyncio.Lock()
        self._transcribe_lock = asyncio.Lock()

    def _load_sync(self):
        from faster_whisper import WhisperModel

        attempts = []
        if self.device == "auto":
            attempts.append(("cuda", "float16"))
            attempts.append(("cpu", "int8"))
        else:
            attempts.append((self.device, self.compute_type))

        last_err = None
        for device, compute in attempts:
            try:
                t0 = time.perf_counter()
                logger.info(
                    "Loading whisper model '%s' (device=%s, compute=%s)...",
                    self.model_name, device, compute,
                )
                model = WhisperModel(self.model_name, device=device, compute_type=compute)
                logger.info(
                    "Whisper model ready in %.1fs (%s/%s)",
                    time.perf_counter() - t0, device, compute,
                )
                return model
            except Exception as e:
                last_err = e
                logger.warning("Whisper load failed on %s/%s: %s", device, compute, e)
        raise last_err

    async def _ensure_model(self):
        if not self._loaded:
            self._model = await asyncio.to_thread(self._load_sync)
            self._loaded = True

    def _load_cpu_sync(self):
        from faster_whisper import WhisperModel

        logger.info("Reloading whisper model '%s' on CPU (int8)", self.model_name)
        return WhisperModel(self.model_name, device="cpu", compute_type="int8")

    def _disable(self, reason: str):
        if not self.enabled:
            return
        self.enabled = False
        logger.warning("Speech transcription disabled after error: %s", reason)

    async def transcribe(self, data: bytes, filename: str) -> str:
        if not self.enabled:
            return ""
        suffix = Path(filename).suffix or ".oga"
        tmp_path = None
        try:
            async with self._load_lock:
                await self._ensure_model()
            fd, tmp_path = tempfile.mkstemp(suffix=suffix)
            with os.fdopen(fd, "wb") as f:
                f.write(data)

            def _run():
                segments, info = self._model.transcribe(tmp_path, vad_filter=True)
                return " ".join(s.text.strip() for s in segments).strip(), info

            t0 = time.perf_counter()
            # serialize inference on the shared model (faster-whisper is not
            # documented as thread-safe, and concurrent voice notes race here)
            async with self._transcribe_lock:
                try:
                    text, info = await asyncio.to_thread(_run)
                except Exception as e:
                    if self.device != "cpu":
                        logger.warning("Whisper inference failed (%s); retrying on CPU", e)
                        self._model = await asyncio.to_thread(self._load_cpu_sync)
                        self.device = "cpu"
                        text, info = await asyncio.to_thread(_run)
                    else:
                        raise
            audio_s = float(getattr(info, "duration", 0) or 0)
            engine = f"whisper-{self.model_name}:{self.device}"
            botlog.log_transcribe(audio_s, len(text), engine)
            logger.info(
                "Transcribed %s (%.1fs audio, lang=%s) -> %d chars in %.1fs",
                filename, getattr(info, "duration", 0), getattr(info, "language", "?"),
                len(text), time.perf_counter() - t0,
            )
            return text
        except Exception as e:
            self._disable(f"transcribe failed: {e}")
            return ""
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
