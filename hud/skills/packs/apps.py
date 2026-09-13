"""Open applications by name — with conversational confirmation.

First call resolves the target and asks ("Open 'Google Chrome'? Reply
'yes' within a minute."). The agent re-calls with confirmed=true after
the user agrees — or a bare "yes" confirms the outstanding request via
the deterministic direct route. Arbitrary paths and URLs are always
refused — app names only. Launches are verified: the tool reports what
is actually running, never a blind claim. System tier: armed via the
HUD checkbox.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import time
from pathlib import Path

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

CONFIRM_WINDOW = 60.0
# Single outstanding confirmation (single-user HUD, serialized turns).
# (target, expiry_monotonic). Lenient by design: weak local models often
# confirm with just "yes" and forget the exact app name.
_PENDING: tuple[dict, float] | None = None

AFFIRM_RE = re.compile(
    r"^(yes|yeah|yep|yup|ok|okay|sure|do it|open it|go ahead|please do|yes please)[.!]*$",
    re.IGNORECASE,
)
OPEN_RE = re.compile(r"\bopen\b\s+(.+?)(?:\s+(?:for me|please|now))?[.!]*$", re.IGNORECASE)
_FILLER_TAIL_RE = re.compile(r"\s+(app|application|program)\s*$", re.IGNORECASE)


def _looks_like_path_or_url(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    low = t.lower()
    if "://" in t or low.startswith(("http", "www.", "ftp.")):
        return True
    if "/" in t or "\\" in t or low.endswith((".exe", ".bat", ".cmd", ".msi", ".ps1")):
        return True
    if len(t) >= 2 and t[1] == ":":
        return True
    return False


def _start_menu_dirs() -> list[Path]:
    dirs = []
    for var in ("ProgramData", "APPDATA"):
        base = os.environ.get(var, "")
        if not base:
            continue
        probe = Path(base) / ("Microsoft/Windows/Start Menu/Programs" if var == "ProgramData" else "Microsoft/Windows/Start Menu/Programs")
        if probe.is_dir():
            dirs.append(probe)
    return dirs


def resolve_app(name: str) -> dict | None:
    """Resolve an app name to a launch target. None when not found."""
    query = (name or "").strip().lower()
    if not query:
        return None
    # Common install locations for popular apps (PATH often lacks them).
    well_known = {
        "chrome": ("Google Chrome", ["Google/Chrome/Application/chrome.exe"]),
        "spotify": ("Spotify", ["Spotify/Spotify.exe"]),
        "vscode": ("VS Code", ["Microsoft VS Code/Code.exe", "Programs/Microsoft VS Code/Code.exe"]),
        "code": ("VS Code", ["Microsoft VS Code/Code.exe", "Programs/Microsoft VS Code/Code.exe"]),
        "telegram": ("Telegram", ["Telegram Desktop/Telegram.exe"]),
        "whatsapp": ("WhatsApp", ["WhatsApp/WhatsApp.exe"]),
    }
    program_files = [os.environ.get("ProgramFiles", ""), os.environ.get("ProgramFiles(x86)", "")]
    appdata = [os.environ.get("LOCALAPPDATA", ""), os.environ.get("APPDATA", "")]
    for alias, (pretty, rels) in well_known.items():
        if alias in query:
            for root in program_files + appdata:
                for rel in rels:
                    candidate = Path(root) / rel if root else None
                    if candidate and candidate.is_file():
                        return {"key": f"exe:{candidate}", "display": pretty, "path": str(candidate)}
    # Anything on PATH (notepad, calc, mspaint, cmd, ...).
    exe = shutil.which(query.split()[0])
    if exe:
        return {"key": f"exe:{exe}", "display": Path(exe).stem, "path": exe}
    # Start-menu shortcuts (*.lnk launch fine via os.startfile).
    for menu in _start_menu_dirs():
        try:
            matches = [p for p in menu.rglob("*.lnk") if query in p.stem.lower()]
        except OSError:
            continue
        if matches:
            best = sorted(matches, key=lambda p: (len(p.stem), str(p)))[0]
            return {"key": f"lnk:{best}", "display": best.stem, "path": str(best)}
    return None


def _pending_target() -> dict | None:
    """Outstanding confirmation target, or None when absent/expired."""
    global _PENDING
    if _PENDING is None:
        return None
    target, expiry = _PENDING
    if time.monotonic() > expiry:
        _PENDING = None
        return None
    return target


def _take_pending() -> dict | None:
    global _PENDING
    target = _pending_target()
    _PENDING = None
    return target


def _proc_names() -> set[str]:
    try:
        import psutil

        names = set()
        for proc in psutil.process_iter(["name"]):
            pname = (proc.info.get("name") or "").lower()
            if pname:
                names.add(pname[:-4] if pname.endswith(".exe") else pname)
        return names
    except Exception:
        return set()


def _matches_running(target: dict, processes: set[str]) -> bool:
    """Conservative already-running check (exe basename or stem words).

    Store apps rename binaries (calc.exe runs as CalculatorApp), so short
    names also match by substring — worst case is a truthful-looking
    "already running" or one redundant launch, never a wrong app.
    """
    path = target.get("path", "")
    stem = Path(path).stem.lower()
    if path.lower().endswith(".exe") or target["key"].startswith("exe:"):
        base = Path(path).stem.lower()
        return any(
            base == p or (len(base) >= 3 and len(p) >= 3 and (base in p or p in base))
            for p in processes
        )
    words = [w for w in re.split(r"\W+", stem) if len(w) >= 3]
    return any(w in p or p in w for w in words for p in processes)


def _launch(path: str) -> None:
    if os.name != "nt":
        raise OSError("app launching is supported on Windows only")
    os.startfile(path)  # noqa: S606 (target resolved internally, never raw user input)


def _launch_verified(target: dict, timeout: float = 5.0, poll: float = 0.25) -> str:
    """Launch and verify the process actually appeared. Truthful either way."""
    if _matches_running(target, _proc_names()):
        return f"{target['display']} is already running."
    try:
        _launch(target["path"])
    except Exception as e:
        logger.warning("open_app launch failed for %s: %s", target["path"], e)
        return f"ERROR: couldn't launch {target['display']} ({e})."
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _matches_running(target, _proc_names()):
            logger.info("open_app launched %s (verified running)", target["display"])
            return f"Opened {target['display']} (verified running)."
        time.sleep(poll)
    logger.warning("open_app launched %s but no process appeared", target["display"])
    return (
        f"ERROR: I asked Windows to open {target['display']} but it isn't "
        f"running — try opening it by hand and tell me what happened."
    )


@tool
def open_app(app_name: str, confirmed: bool = False) -> str:
    """Launch an application by name (e.g. 'Chrome', 'Notepad', 'Spotify').

    Always asks for confirmation first: call once with just app_name and
    relay the NEEDS-CONFIRM question to the user. When the user agrees,
    call again with confirmed=true (app_name may be repeated or omitted —
    a bare 'yes' confirms the outstanding request). Give app names, never
    file paths or URLs. Results are verified against running processes,
    so relay them verbatim instead of inventing success.
    """
    global _PENDING
    name = (app_name or "").strip()
    if _looks_like_path_or_url(name) and name:
        return (
            f"ERROR: I only open applications by name, not paths or links "
            f"('{name}'). Tell me the app name."
        )
    target = resolve_app(name) if name and not AFFIRM_RE.fullmatch(name) else None
    pending = _pending_target()
    if confirmed or (not name or AFFIRM_RE.fullmatch(name)):
        # Confirmation path: launch ONLY with a valid outstanding ask.
        # No pending (fresh/expired/consumed) -> ask anew, never launch.
        # A differently resolving name replaces the ask instead of
        # launching the wrong app.
        if pending is None:
            if target is None:
                return "ERROR: tell me which app to open."
            _PENDING = (target, time.monotonic() + CONFIRM_WINDOW)
            return (
                f"NEEDS-CONFIRM: Open '{target['display']}'? "
                f"Reply 'yes' within a minute to proceed."
            )
        if target is not None and target["key"] != pending["key"]:
            _PENDING = (target, time.monotonic() + CONFIRM_WINDOW)
            return (
                f"NEEDS-CONFIRM: Open '{target['display']}'? "
                f"Reply 'yes' within a minute to proceed."
            )
        _PENDING = None
        return _launch_verified(pending)
    if target is None:
        return (
            f"ERROR: couldn't find an app matching '{name}' in PATH or the "
            f"Start menu. Try the full name as shown in Start."
        )
    _PENDING = (target, time.monotonic() + CONFIRM_WINDOW)
    return (
        f"NEEDS-CONFIRM: Open '{target['display']}'? "
        f"Reply 'yes' within a minute to proceed — then call open_app again "
        f"with confirmed=true."
    )


def direct_match(text: str, context: dict | None = None) -> tuple[str, dict] | None:
    """Deterministic route for open-app intents (weak-model-proof).

    "open X" resolves without the model; a bare "yes" confirms the
    outstanding request. Unknown targets return None (agent handles).
    """
    raw = (text or "").strip()
    if not raw:
        return None
    pending = _pending_target()
    if AFFIRM_RE.fullmatch(raw):
        if pending is None:
            return None
        return ("open_app", {"app_name": pending["display"], "confirmed": True})
    match = OPEN_RE.search(raw)
    if not match:
        return None
    candidate = _FILLER_TAIL_RE.sub("", match.group(1)).strip(" ,.")
    if not candidate or _looks_like_path_or_url(candidate):
        return None
    target = resolve_app(candidate)
    if target is None:
        return None
    return ("open_app", {"app_name": candidate, "confirmed": False})


SKILL = {
    "name": "open_app",
    "version": "1.0",
    "description": "Launch applications by name (asks for confirmation first).",
    "risk": "system",
    "tools": [open_app],
}
