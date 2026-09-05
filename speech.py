import asyncio
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

import botlog

logger = logging.getLogger(__name__)

# Default vendored Piper locations (see README voice-mode section).
# The user downloads these manually once; afterwards synthesis is offline.
DEFAULT_PIPER_EXE = Path(__file__).parent / "vendor" / "piper" / "piper.exe"
DEFAULT_PIPER_MODEL = (
    Path(__file__).parent / "vendor" / "piper" / "voices" / "en_US-lessac-high.onnx"
)

# Hard cap per synthesis so a runaway reply can't wedge the voice loop.
MAX_SYNTH_CHARS = 1000


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


# ------------------------------------------------------------- Piper TTS


def piper_paths() -> tuple[Path, Path]:
    """Resolve (binary, voice model) from env with vendored defaults."""
    exe = Path(os.environ.get("VOICE_PIPER_EXE", str(DEFAULT_PIPER_EXE)))
    model = Path(os.environ.get("VOICE_PIPER_MODEL", str(DEFAULT_PIPER_MODEL)))
    return exe, model


def piper_available() -> tuple[bool, str]:
    """Cheap pre-flight check. Returns (ok, reason)."""
    exe, model = piper_paths()
    if not exe.exists():
        return False, f"Piper binary not found: {exe}"
    if not model.exists():
        return False, f"Piper voice model not found: {model}"
    if not Path(str(model) + ".json").exists():
        return False, f"Piper voice config missing: {model}.json"
    return True, "ok"


def _clean_for_speech(text: str) -> str:
    """Strip markdown/code artefacts TTS would read aloud literally."""
    import re

    text = re.sub(r"```.*?```", " code snippet omitted. ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[#*_>|-]{1,}", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_SYNTH_CHARS:
        # Cut at a sentence boundary so speech doesn't end mid-word.
        cut = text.rfind(". ", 0, MAX_SYNTH_CHARS)
        text = text[: cut + 1] if cut > 0 else text[:MAX_SYNTH_CHARS]
    return text


class PiperSpeaker:
    """Offline TTS via the vendored Piper binary (WAV bytes out)."""

    def __init__(self) -> None:
        self.speaker_id = os.environ.get("VOICE_PIPER_SPEAKER", "0")
        self.timeout = int(os.environ.get("VOICE_PIPER_TIMEOUT", "30"))
        self._lock = asyncio.Lock()

    async def synthesize(self, text: str) -> bytes:
        """Speak ``text``. Returns WAV bytes, or ``b""`` on any failure."""
        ok, reason = piper_available()
        if not ok:
            logger.warning("Piper unavailable (%s); reply will be text-only", reason)
            return b""
        clean = _clean_for_speech(text)
        if not clean:
            return b""
        exe, model = piper_paths()
        try:
            async with self._lock:
                return await asyncio.to_thread(self._run_sync, exe, model, clean)
        except Exception:
            logger.exception("Piper synthesis failed")
            return b""

    def _run_sync(self, exe: Path, model: Path, text: str) -> bytes:
        out_path = None
        try:
            fd, out_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            proc = subprocess.run(
                [
                    str(exe),
                    "--model", str(model),
                    "--output_file", out_path,
                    "--speaker", str(self.speaker_id),
                    "--quiet",
                ],
                input=text.encode("utf-8"),
                capture_output=True,
                timeout=self.timeout,
            )
            if proc.returncode != 0:
                logger.warning(
                    "Piper exited %d: %s", proc.returncode,
                    proc.stderr.decode("utf-8", "replace")[:300],
                )
                return b""
            return Path(out_path).read_bytes()
        except (OSError, subprocess.SubprocessError) as e:
            logger.warning("Piper run failed: %s", e)
            return b""
        finally:
            if out_path:
                try:
                    os.unlink(out_path)
                except OSError:
                    pass
