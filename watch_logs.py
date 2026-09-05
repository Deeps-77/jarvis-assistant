import argparse
import os
import sys
import time
from pathlib import Path

from paths import LOG_DIR as LOGS_DIR

POLL_INTERVAL = 0.5

_COLORS = {
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "grey": "\033[90m",
    "reset": "\033[0m",
}


def enable_ansi():
    if sys.platform == "win32":
        os.system("")


def colorize(line: str) -> str:
    low = line.lower()
    if "error" in low or line.startswith("❌"):
        c = "red"
    elif "warning" in low or line.startswith("⚠") or "gated-fallback" in line:
        c = "yellow"
    elif "denied" in low or "🛡" in line:
        c = "magenta"
    elif "👤" in line or "🔑" in line:
        c = "cyan"
    elif "💬 reply" in low or "✅" in line:
        c = "green"
    elif line.startswith(("📄", "🎙", "🔧")):
        c = "cyan"
    elif line.startswith("📊"):
        c = "grey"
    elif line.startswith("─"):
        c = "grey"
    else:
        return line
    return f"{_COLORS[c]}{line}{_COLORS['reset']}"


def _backfill(path: Path, n: int) -> tuple[list[str], int]:
    if not path.exists():
        return [], 0
    data = path.read_bytes()
    if not data:
        return [], 0
    segments = data.split(b"\n")
    keep = segments[-(n + 1):] if len(segments) > n else segments
    text = b"\n".join(keep).decode("utf-8", errors="replace")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    return lines, len(data)


def follow_files(paths: list[Path], stop_check, printer, backfill: int = 20):
    offsets = {}
    for p in paths:
        lines, offsets[p] = _backfill(p, backfill)
        for ln in lines:
            printer(colorize(ln))
    printer("── live · Ctrl+C to stop ──")
    while not stop_check():
        for p in paths:
            try:
                size = p.stat().st_size
            except OSError:
                size = 0
            off = offsets.get(p, 0)
            if size < off:
                printer(f"── {p.name} rotated · restarting ──")
                offsets[p] = 0
                continue
            if size == off:
                continue
            with open(p, "rb") as f:
                f.seek(off)
                chunk = f.read(size - off)
            cut = chunk.rfind(b"\n")
            if cut == -1:
                continue
            text = chunk[: cut + 1].decode("utf-8", errors="replace")
            offsets[p] += cut + 1
            for ln in text.split("\n"):
                if ln.strip():
                    printer(colorize(ln))
        time.sleep(POLL_INTERVAL)


def main():
    enable_ansi()
    parser = argparse.ArgumentParser(description="Live viewer for Jarvis logs")
    parser.add_argument("--lines", type=int, default=20, help="initial lines per file")
    args = parser.parse_args()

    paths = [LOGS_DIR / "activity.log", LOGS_DIR / "jarvis.log"]
    print(f"{_COLORS['grey']}watching {', '.join(p.name for p in paths)}{_COLORS['reset']}")
    try:
        follow_files(paths, lambda: False, print, backfill=max(0, args.lines))
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
