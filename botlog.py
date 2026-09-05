import json
import logging
import sys
from collections import Counter
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from paths import LOG_DIR as LOGS_DIR
ACTIVITY_FILE = LOGS_DIR / "activity.log"
DIAG_FILE = LOGS_DIR / "jarvis.log"
EVENTS_FILE = LOGS_DIR / "events.jsonl"
MAX_JSONL_BYTES = 10 * 1024 * 1024

_activity = logging.getLogger("jarvis.activity")


class _BenignSocketResetFilter(logging.Filter):
    """Suppress Windows Proactor shutdown-race noise.

    When a browser/socket.io client aborts a connection mid-teardown,
    asyncio's cleanup logs a full ERROR traceback for the already-reset
    socket (WinError 10054). Requests succeeded; the traceback is pure noise.
    Everything else - including any other ConnectionResetError - passes.
    """

    MAX_CHAIN_DEPTH = 6

    def filter(self, record: logging.LogRecord) -> bool:
        if record.exc_info:
            exc = record.exc_info[1]
            depth = 0
            while exc is not None and depth < self.MAX_CHAIN_DEPTH:
                if isinstance(exc, ConnectionResetError) and getattr(exc, "winerror", None) == 10054:
                    return False
                nxt = exc.__cause__ or exc.__context__
                if nxt is exc:
                    break
                exc = nxt
                depth += 1
        return True


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

    diag_file = TimedRotatingFileHandler(
        DIAG_FILE, when="midnight", backupCount=30, encoding="utf-8"
    )
    diag_file.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    root.addHandler(diag_file)
    logging.getLogger("asyncio").addFilter(_BenignSocketResetFilter())

    activity_file = TimedRotatingFileHandler(
        ACTIVITY_FILE, when="midnight", backupCount=30, encoding="utf-8"
    )
    activity_file.setFormatter(logging.Formatter("%(message)s"))
    _activity.addHandler(activity_file)
    _activity.setLevel(logging.INFO)
    _activity.propagate = False

    for noisy in ("httpx", "primp", "uvicorn.access", "uvicorn.error", "chainlit"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _jsonl(event: str, **fields) -> None:
    record = {"ts": datetime.now().isoformat(timespec="seconds"), "event": event}
    record.update(fields)
    try:
        if EVENTS_FILE.exists() and EVENTS_FILE.stat().st_size > MAX_JSONL_BYTES:
            EVENTS_FILE.replace(EVENTS_FILE.with_suffix(".jsonl.old"))
        with open(EVENTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


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
    _jsonl("startup", model=model, users=users_desc, memory=memory_enabled, tools=tools_count)


def log_user_msg(name: str, user_id, chat_desc: str, text: str, kind: str = "text"):
    preview = " ".join((text or "").split())[:100]
    icon = "🎤" if kind == "voice" else "👤"
    _activity.info('%s %s (%s) · %s\n    💬 "%s"', icon, name, user_id, chat_desc, preview)
    _jsonl("user_msg", user=name, user_id=str(user_id), chat=chat_desc, kind=kind, chars=len(preview))


def log_tools(names: list[str]):
    if not names:
        return
    counts = Counter(names)
    summary = ", ".join(f"{n} x{c}" if c > 1 else n for n, c in counts.items())
    _activity.info("🔧 Tools used: %s", summary)
    _jsonl("tools", tools=names)


def log_reply(duration_s: float, sources_count: int, status: str):
    icon = "⚠️" if status != "ok" else "✅"
    _activity.info(
        "%s 💬 Reply in %.1fs · sources: %d · %s", icon, duration_s, sources_count, status
    )
    _jsonl("reply", duration_s=round(duration_s, 1), sources=sources_count, status=status)


def log_denied(name: str, user_id):
    _activity.warning("🛡️ DENIED %s (%s) — not on the allowlist", name, user_id)
    _jsonl("denied", name=name, user_id=str(user_id))


def log_command(command: str, name: str, user_id):
    _activity.info("⌨️ /%s by %s (%s)", command, name, user_id)
    _jsonl("command", cmd=command, name=name, user_id=str(user_id))


def log_transcribe(audio_s: float, chars: int, engine: str):
    _activity.info("🎙️ Transcribed %.1fs audio -> %d chars [%s]", audio_s, chars, engine)
    _jsonl("transcribe", audio_s=round(audio_s, 1), chars=chars, engine=engine)


def log_doc(user_id, source: str, status: str, chunks: int = 0):
    if status == "added":
        _activity.info('📄 Indexed "%s" for %s (%d chunks)', source, user_id, chunks)
    elif status == "unchanged":
        _activity.info('📄 "%s" already indexed for %s (dedup)', source, user_id)
    else:
        _activity.info('📄 Doc index problem for %s: %s', user_id, source)
    _jsonl("doc", user_id=str(user_id), source=source, status=status, chunks=chunks)


def log_web_login(username: str):
    _activity.info("🔑 web login: %s", username)
    _jsonl("web_login", user=username)


def log_error_note(message: str):
    _activity.error("❌ %s", message)
    _jsonl("error", message=message[:300])


def get_stats() -> str:
    date = datetime.now().strftime("%Y-%m-%d")
    per_user: dict[str, int] = {}
    voice = 0
    replies = gated = docs_added = docs_dedup = denied = commands = 0
    tools: Counter = Counter()

    if EVENTS_FILE.exists():
        for line in EVENTS_FILE.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not str(rec.get("ts", "")).startswith(date):
                continue
            ev = rec.get("event")
            if ev == "user_msg":
                user = rec.get("user", "?")
                per_user[user] = per_user.get(user, 0) + 1
                if rec.get("kind") == "voice":
                    voice += 1
            elif ev == "reply":
                replies += 1
                if rec.get("status") != "ok":
                    gated += 1
            elif ev == "tools":
                for t in rec.get("tools", []):
                    tools[t] += 1
            elif ev == "doc":
                if rec.get("status") == "added":
                    docs_added += 1
                elif rec.get("status") == "unchanged":
                    docs_dedup += 1
            elif ev == "denied":
                denied += 1
            elif ev == "command":
                commands += 1

    out = [f"📊 Jarvis today ({date})"]
    if per_user:
        out.append(
            "👤 " + " · ".join(f"{u}: {c} msgs" for u, c in sorted(per_user.items(), key=lambda x: -x[1]))
        )
    out.append(f"💬 Replies: {replies} ({gated} gated) · 🎤 Voice notes: {voice}")
    if tools:
        top = ", ".join(f"{t} x{c}" for t, c in tools.most_common(5))
        out.append(f"🔧 Tools: {top}")
    out.append(f"📄 Docs indexed: {docs_added} (dedup skips: {docs_dedup})")
    out.append(f"🛡️ Denied: {denied} · ⌨️ Commands: {commands}")
    return "\n".join(out)