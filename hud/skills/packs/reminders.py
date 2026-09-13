"""One-shot reminders via Windows Task Scheduler (system tier).

Actions: add ("remind me in 20 minutes to stretch"), list, cancel.
Creation uses schtasks with a `msg *` popup payload — built into Windows,
no daemons, no new dependencies. Past/invalid times are refused with a
spoken correction, never silently scheduled.
"""

from __future__ import annotations

import csv
import io
import logging
import os
import re
import subprocess
import time
from datetime import datetime, timedelta

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

TASK_PREFIX = "JarvisReminder_"
_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


# ------------------------------------------------------------- parsing


def parse_reminder_when(text: str, now: datetime | None = None) -> datetime | None:
    """Parse natural datetime input. None when unparseable or in the past."""
    now = now or datetime.now()
    raw = (text or "").strip().lower()
    if not raw:
        return None

    m = re.fullmatch(r"in\s+(\d+)\s*(minute|minutes|min|mins)\b.*", raw)
    if m:
        return now + timedelta(minutes=int(m.group(1)))
    m = re.fullmatch(r"in\s+(\d+)\s*(hour|hours|hr|hrs)\b.*", raw)
    if m:
        return now + timedelta(hours=int(m.group(1)))
    m = re.fullmatch(r"in\s+(\d+)\s*(day|days)\b.*", raw)
    if m:
        return now + timedelta(days=int(m.group(1)))

    m = re.fullmatch(r"(?:at\s+)?(?:(today|tomorrow)\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", raw)
    if m:
        day_word, hour_s, minute_s, meridiem = m.groups()
        hour, minute = int(hour_s), int(minute_s or 0)
        if hour > 23 or minute > 59:
            return None
        if meridiem == "pm" and hour < 12:
            hour += 12
        if meridiem == "am" and hour == 12:
            hour = 0
        day = now.date() + timedelta(days=1 if day_word == "tomorrow" else 0)
        candidate = datetime.combine(day, datetime.min.time()).replace(hour=hour, minute=minute)
        if candidate <= now and not day_word:
            candidate += timedelta(days=1)  # "at 9" after 9am means tomorrow
        return candidate if candidate > now else None

    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M"):
        try:
            candidate = datetime.strptime(raw, fmt)
            return candidate if candidate > now else None
        except ValueError:
            continue
    return None


def _slug(message: str, when: datetime) -> str:
    safe = re.sub(r"[^A-Za-z0-9]+", "", message)[:24]
    return f"{TASK_PREFIX}{when.strftime('%Y%m%d_%H%M%S')}_{safe or 'note'}"


# ------------------------------------------------------------- schtasks


def _run_schtasks(args: list[str]) -> tuple[bool, str]:
    if os.name != "nt":
        return False, "reminders are supported on Windows only"
    try:
        proc = subprocess.run(
            ["schtasks"] + args,
            capture_output=True,
            text=True,
            timeout=20,
            creationflags=_CREATE_NO_WINDOW,
        )
    except Exception as e:
        return False, str(e)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, out.strip()[:500]


def _list_tasks() -> list[dict]:
    ok, out = _run_schtasks(["/query", "/fo", "csv", "/v"])
    if not ok:
        return []
    tasks = []
    try:
        for row in csv.DictReader(io.StringIO(out)):
            name = (row.get("TaskName") or "").strip("\\")
            if name.startswith(TASK_PREFIX):
                tasks.append({"name": name})
    except Exception:
        pass
    return sorted(tasks, key=lambda t: t["name"])


def _decode_task(name: str) -> str:
    # JarvisReminder_YYYYMMDD_HHMMSS_slug -> "slug @ YYYY-MM-DD HH:MM"
    m = re.fullmatch(rf"{re.escape(TASK_PREFIX)}(\d{{8}})_(\d{{6}})_(.*)", name)
    if not m:
        return name
    return f"{m.group(3) or 'note'} @ {m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]} {m.group(2)[:2]}:{m.group(2)[2:4]}"


# ------------------------------------------------------------- direct routing

# Trigger + time-span patterns for deterministic matching (small models
# thrash with 16+ bound tools, so clear reminder intents skip routing).
_TRIGGER_RE = re.compile(r"\bremind\s+me\b|\bset\s+(?:a\s+)?reminders?\b|\bremind\b", re.IGNORECASE)
_TIME_RES = [
    r"in\s+\d+\s*(?:minutes?|mins?|hours?|hrs?|days?)\b",
    r"(?:tomorrow|today)\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?",
    r"(?:at\s+)?\d{1,2}:\d{2}\s*(?:am|pm)?",
    r"\d{4}-\d{2}-\d{2}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?",
]
_LIST_RE = re.compile(r"\b(list|show|what are)\b.{0,25}\breminders?\b|^reminders?\??$", re.IGNORECASE)
_CANCEL_RE = re.compile(r"\b(?:cancel|delete|remove|stop|clear)\b(.{0,30}?)\bremind\w*\b", re.IGNORECASE)
_LEAD_FILLER_RE = re.compile(r"^(?:me|to|for|about|that|of|on|please|kindly)\b[\s,]*", re.IGNORECASE)


def split_reminder(text: str) -> tuple[str, str, str, str] | None:
    """Split a reminder request into (action, message, when, target).

    Returns None when the text isn't a clear reminder intent (the agent
    then handles it, e.g. by asking for the missing time).
    """
    raw = (text or "").strip()
    if not raw:
        return None
    if _LIST_RE.search(raw):
        return ("list", "", "", "")
    cancel_match = _CANCEL_RE.search(raw)
    if cancel_match:
        target = _LEAD_FILLER_RE.sub("", " ".join(cancel_match.group(1).split())).strip(" ,.")
        return ("cancel", "", "", target)
    trigger = _TRIGGER_RE.search(raw)
    if not trigger:
        return None
    time_match = None
    for pattern in _TIME_RES:
        time_match = re.search(pattern, raw, re.IGNORECASE)
        if time_match:
            break
    if not time_match:
        return None
    when = time_match.group(0)
    if parse_reminder_when(when) is None:
        return None
    message = (raw[: trigger.start()] + " " + raw[trigger.end():]).strip()
    message = message.replace(when, " ", 1)
    message = _LEAD_FILLER_RE.sub("", " ".join(message.split())).strip(" ,.")
    if not message:
        return None
    return ("add", message, when, "")


def direct_match(text: str, context: dict | None = None) -> tuple[str, dict] | None:
    """Deterministic route for clear reminder intents.

    Returns (tool_name, args) or None. Used by the HUD so weak local
    models don't have to discover the tool through a 16-tool belt.
    """
    split = split_reminder(text)
    if split is None:
        return None
    action, message, when, target = split
    return ("remind_me", {"action": action, "message": message, "when": when, "target": target})


# ------------------------------------------------------------- tool


@tool
def remind_me(action: str = "add", message: str = "", when: str = "", target: str = "") -> str:
    """Manages one-shot reminders (Windows Task Scheduler popup).

    Actions: add | list | cancel. To add: give the message and when, e.g.
    when="in 20 minutes", "tomorrow 9am", "at 14:30", "2026-09-14 09:00".
    Past times are refused. To cancel: target=number from list, or text
    matching the reminder.
    """
    act = (action or "add").strip().lower()
    if act == "list":
        tasks = _list_tasks()
        if not tasks:
            return "No reminders scheduled."
        return "\n".join(f"{i + 1}. {_decode_task(t['name'])}" for i, t in enumerate(tasks))
    if act == "cancel":
        tasks = _list_tasks()
        if not tasks:
            return "No reminders scheduled."
        want = (target or "").strip().lower()
        pick = None
        if want.isdigit():
            idx = int(want) - 1
            if 0 <= idx < len(tasks):
                pick = tasks[idx]["name"]
        else:
            hits = [t["name"] for t in tasks if want and want in _decode_task(t["name"]).lower()]
            if len(hits) == 1:
                pick = hits[0]
            elif len(hits) > 1:
                return f"ERROR: '{target}' matches several — be specific:\n" + "\n".join(_decode_task(h) for h in hits)
        if pick is None:
            listing = "\n".join(f"{i + 1}. {_decode_task(t['name'])}" for i, t in enumerate(tasks))
            return f"ERROR: couldn't match '{target}'. Scheduled:\n{listing}"
        ok, out = _run_schtasks(["/delete", "/tn", pick, "/f"])
        if ok:
            logger.info("reminder cancelled %s", pick)
            return f"Cancelled: {_decode_task(pick)}."
        return f"ERROR: couldn't cancel ({out})."

    # add (default)
    msg = (message or "").strip()
    if not msg:
        return "ERROR: tell me what to remind you about."
    moment = parse_reminder_when(when or "")
    if moment is None:
        return (
            f"ERROR: couldn't understand '{when}' as a future time. "
            f"Try 'in 20 minutes', 'tomorrow 9am' or '2026-09-14 14:30'."
        )
    name = _slug(msg, moment)
    popup = 'msg * "' + msg.replace('"', "'")[:200] + '"'
    ok, out = _run_schtasks([
        "/create", "/tn", name,
        "/tr", popup,
        "/sc", "once",
        "/st", moment.strftime("%H:%M"),
        "/sd", moment.strftime("%m/%d/%Y"),
        "/f",
    ])
    if ok:
        logger.info("reminder scheduled %s", name)
        return f"Reminder set: {msg} @ {moment.strftime('%A %d %B, %I:%M %p')}."
    return f"ERROR: couldn't schedule ({out})."


SKILL = {
    "name": "reminders",
    "version": "1.0",
    "description": "One-shot reminders (add/list/cancel) via Windows Task Scheduler.",
    "risk": "system",
    "tools": [remind_me],
}
