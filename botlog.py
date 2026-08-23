import logging
import sys
from collections import Counter
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

LOGS_DIR = Path(__file__).parent / "logs"
_activity = logging.getLogger("jarvis.activity")


def setup_logging() -> None:
    LOGS_DIR.mkdir(exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    )
    root.addHandler(console)

    file_handler = TimedRotatingFileHandler(
        LOGS_DIR / "jarvis.log", when="midnight", backupCount=30, encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    root.addHandler(file_handler)

    for noisy in ("httpx", "primp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    activity_file = TimedRotatingFileHandler(
        LOGS_DIR / "activity.log", when="midnight", backupCount=30, encoding="utf-8"
    )
    activity_file.setFormatter(logging.Formatter("%(message)s"))
    _activity.addHandler(activity_file)
    _activity.setLevel(logging.INFO)
    _activity.propagate = False


def log_startup(model: str, users_desc: str, memory_enabled: bool, tools_count: int):
    lines = [
        "─" * 60,
        "🤖 JARVIS ONLINE",
        f"   🧠 Model   : {model}",
        f"   👥 Users   : {users_desc}",
        f"   🧩 Memory  : {'enabled' if memory_enabled else 'disabled'}",
        f"   🔧 Tools   : {tools_count} loaded",
        "─" * 60,
    ]
    _activity.info("\n".join(lines))


def log_user_msg(name: str, user_id, chat_desc: str, text: str):
    preview = " ".join((text or "").split())[:100]
    _activity.info('👤 %s (%s) · %s\n    💬 "%s"', name, user_id, chat_desc, preview)


def log_tools(names: list[str]):
    if not names:
        return
    counts = Counter(names)
    summary = ", ".join(f"{n} x{c}" if c > 1 else n for n, c in counts.items())
    _activity.info("🔧 Tools used: %s", summary)


def log_reply(duration_s: float, sources_count: int, status: str):
    icon = "⚠️" if status != "ok" else "✅"
    _activity.info(
        "%s 💬 Reply in %.1fs · sources: %d · %s", icon, duration_s, sources_count, status
    )


def log_denied(name: str, user_id):
    _activity.warning("🛡️ DENIED %s (%s) — not on the allowlist", name, user_id)


def log_command(command: str, name: str, user_id):
    _activity.info("⌨️ /%s by %s (%s)", command, name, user_id)


def log_error_note(message: str):
    _activity.error("❌ %s", message)
