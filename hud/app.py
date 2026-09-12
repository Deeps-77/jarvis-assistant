"""Desktop HUD entry point. Run: ``python -m hud.app`` (or run.ps1 → hud)."""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def main() -> int:
    import core
    from paths import memory_db
    from paths import documents_dir

    core.load_histories()
    core.init_memory(memory_db())
    core.init_docs(memory_db())
    core.init_speech()
    documents_dir().mkdir(exist_ok=True)

    from PyQt6.QtWidgets import QApplication

    from . import theme
    from .window import MainWindow

    app = QApplication(sys.argv[:1])
    theme.apply_theme(app)
    win = MainWindow()

    async def _refresh_models():
        try:
            models = await core.list_ollama_models()
            win.refresh_models([m["name"] for m in models])
        except Exception:
            logger.warning("HUD model list failed; keeping current", exc_info=True)
            win.refresh_models([core.MODEL_NAME])

    from .loop import call as _call

    _call(_refresh_models(), timeout=60)
    win.show()

    def _warmup_voice():
        # Pay torch/whisper/Silero first-load cost upfront in the background
        # so the first real turn isn't the slow (or frozen-feeling) one.
        # NOTE: uses submit() (non-blocking) to avoid holding _transcribe_lock
        # across the daemon thread boundary, which was causing the voice pipeline
        # to appear frozen: the STT QThread would block waiting for the lock
        # that the warmup held, and no timeout would surface the error.
        import time
        time.sleep(3)  # let the app settle before competing for the lock
        try:
            from voice_server.stt import warmup as stt_warmup
            from voice_server.vad import has_speech
            from .loop import submit

            logger.info("HUD voice warmup starting")
            future = submit(stt_warmup())
            future.result(timeout=300)  # wait but in the daemon thread, not GUI
            has_speech(b"\x00" * 32000)  # load Silero once
            logger.info("HUD voice warmup done")
        except Exception:
            logger.exception("HUD voice warmup failed (non-fatal)")

    import threading

    threading.Thread(target=_warmup_voice, name="hud-warmup", daemon=True).start()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
