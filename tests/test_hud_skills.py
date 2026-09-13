"""HUD skill packs: loader isolation/gating, confirm flow, reminders.

All OS touchpoints (os.startfile, subprocess, Qt, vision) are mocked —
no windows opened, no tasks scheduled, no Ollama needed.
"""

import asyncio
import time
from datetime import datetime
from pathlib import Path

from hud.skills import loader as loader_mod
from hud.skills.loader import (
    describe,
    load_skills,
    set_system_armed,
    set_turn_context,
    turn_tools,
)


def _write_pack(directory: Path, name: str, body: str) -> None:
    (directory / name).write_text(body, encoding="utf-8")


_GOOD = '''
from langchain_core.tools import tool

@tool
def hello_skill() -> str:
    """Says hello."""
    return "hi"

SKILL = {"name": "hello", "version": "1.0",
         "description": "Says hello.", "risk": "readonly",
         "tools": [hello_skill]}
'''

_SYSTEM = '''
from langchain_core.tools import tool

@tool
def scary_skill() -> str:
    """Does scary things."""
    return "boo"

SKILL = {"name": "scary", "version": "1.0",
         "description": "Does scary things.", "risk": "system",
         "tools": [scary_skill]}
'''

_BROKEN = "this is not python (((("

_NOMANIFEST = '''
from langchain_core.tools import tool

@tool
def orphan() -> str:
    """No manifest here."""
    return "?"
'''


def _reset():
    loader_mod._registry = {}
    loader_mod._providers.clear()
    set_system_armed(False)
    set_turn_context()
    import hud.skills.packs.apps as _apps

    _apps._PENDING = None


def test_loader_isolation_and_gating(tmp_path, monkeypatch):
    monkeypatch.setenv("HUD_SKILLS_ALLOW", "")
    _reset()
    _write_pack(tmp_path, "a_good.py", _GOOD)
    _write_pack(tmp_path, "b_system.py", _SYSTEM)
    _write_pack(tmp_path, "c_broken.py", _BROKEN)
    _write_pack(tmp_path, "d_nomanifest.py", _NOMANIFEST)
    _write_pack(tmp_path, "_private.py", _GOOD)
    reg = load_skills(tmp_path)
    assert set(reg) == {"hello", "scary"}
    assert "hello" in describe() and "scary" in describe()
    # System gated off by default.
    assert [t.name for t in turn_tools()] == ["hello_skill"]
    set_system_armed(True)
    assert sorted(t.name for t in turn_tools()) == ["hello_skill", "scary_skill"]
    set_system_armed(False)
    # Env override arms without checkbox.
    monkeypatch.setenv("HUD_SKILLS_ALLOW", "1")
    set_system_armed(False)
    assert sorted(t.name for t in turn_tools()) == ["hello_skill", "scary_skill"]


def test_loader_duplicate_names(tmp_path):
    _reset()
    _write_pack(tmp_path, "a_one.py", _GOOD)
    _write_pack(tmp_path, "b_one.py", _GOOD.replace('"hello"', '"hello"'))
    reg = load_skills(tmp_path)
    assert set(reg) == {"hello"}
    assert reg["hello"].source == "a_one.py"


def test_open_app_confirm_flow(monkeypatch):
    _reset()
    import hud.skills.packs.apps as apps_mod

    monkeypatch.setattr(apps_mod, "resolve_app", lambda name: {"key": "k", "display": "FakeApp", "path": "C:/x.exe"} if name == "fake" else None)
    launched = []
    monkeypatch.setattr(apps_mod, "_launch", lambda path: launched.append(path))
    # NOTE: do not freeze time.monotonic here — the verify poll loop needs
    # a live clock or it spins forever.
    monkeypatch.setattr(apps_mod, "_proc_names", lambda: set())

    first = asyncio.run(apps_mod.open_app.ainvoke({"app_name": "fake"}))
    assert first.startswith("NEEDS-CONFIRM")
    assert launched == []
    # Wrong target / unknown app cannot ride the pending confirmation.
    assert "couldn't find" in asyncio.run(apps_mod.open_app.ainvoke({"app_name": "nope"}))
    # FakeApp never appears in the (mocked-empty) process list: the launch
    # is attempted (proving confirm consumed the pending ask) but reported
    # honestly instead of claimed. assert fast by shrinking the poll window.
    import hud.skills.packs.apps as _apps

    _orig_verify = _apps._launch_verified
    monkeypatch.setattr(
        _apps, "_launch_verified",
        lambda target, timeout=5.0, poll=0.25: _orig_verify(target, timeout=0.05, poll=0.01),
    )
    done = asyncio.run(apps_mod.open_app.ainvoke({"app_name": "fake", "confirmed": True}))
    assert done.startswith("ERROR") and "isn't running" in done
    assert launched == ["C:/x.exe"]
    # Confirmation is single-use.
    again = asyncio.run(apps_mod.open_app.ainvoke({"app_name": "fake", "confirmed": True}))
    assert again.startswith("NEEDS-CONFIRM")


def test_open_app_confirm_expiry(monkeypatch):
    _reset()
    import hud.skills.packs.apps as apps_mod

    now = [1000.0]
    monkeypatch.setattr(apps_mod, "resolve_app", lambda name: {"key": "k", "display": "FakeApp", "path": "C:/x.exe"})
    monkeypatch.setattr(apps_mod.time, "monotonic", lambda: now[0])
    asyncio.run(apps_mod.open_app.ainvoke({"app_name": "fake"}))
    now[0] += 61.5  # past the 60s window
    assert asyncio.run(apps_mod.open_app.ainvoke({"app_name": "fake", "confirmed": True})).startswith("NEEDS-CONFIRM")


def test_open_app_refuses_paths_and_urls():
    _reset()
    import hud.skills.packs.apps as apps_mod

    for bad in ["C:/evil.exe", "..\\x", "https://x.example", "www.x.example", "run.bat"]:
        out = asyncio.run(apps_mod.open_app.ainvoke({"app_name": bad}))
        assert out.startswith("ERROR"), bad


def test_reminder_parse():
    from hud.skills.packs.reminders import parse_reminder_when

    base = datetime(2026, 9, 13, 10, 0, 0)
    assert parse_reminder_when("in 20 minutes", base) == datetime(2026, 9, 13, 10, 20)
    assert parse_reminder_when("in 2 hours", base) == datetime(2026, 9, 13, 12, 0)
    assert parse_reminder_when("tomorrow 9am", base) == datetime(2026, 9, 14, 9, 0)
    assert parse_reminder_when("at 14:30", base) == datetime(2026, 9, 13, 14, 30)
    assert parse_reminder_when("at 9", base) == datetime(2026, 9, 14, 9, 0)  # past -> tomorrow
    assert parse_reminder_when("2026-09-14 09:00", base) == datetime(2026, 9, 14, 9, 0)
    assert parse_reminder_when("yesterday", base) is None
    assert parse_reminder_when("2020-01-01 00:00", base) is None
    assert parse_reminder_when("", base) is None
    assert parse_reminder_when("at 25:00", base) is None


def test_reminder_add_builds_schtasks(monkeypatch):
    import hud.skills.packs.reminders as rem_mod

    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        assert args[0] == "schtasks" and args[1] == "/create"
        assert "/tn" in args and "/tr" in args and "/st" in args and "/sd" in args

        class R:
            returncode = 0
            stdout = "SUCCESS"
            stderr = ""

        return R()

    monkeypatch.setattr(rem_mod.subprocess, "run", fake_run)
    out = asyncio.run(rem_mod.remind_me.ainvoke({"action": "add", "message": "stretch", "when": "in 20 minutes"}))
    assert out.startswith("Reminder set:")
    assert calls and "stretch" in " ".join(calls[0])


def test_reminder_add_rejects_past():
    import hud.skills.packs.reminders as rem_mod

    out = asyncio.run(rem_mod.remind_me.ainvoke({"action": "add", "message": "x", "when": "2000-01-01 00:00"}))
    assert out.startswith("ERROR")


def test_look_needs_grabber_and_context(monkeypatch):
    _reset()
    import hud.skills.packs.look as look_mod

    out = asyncio.run(look_mod.look_at_screen.ainvoke({"question": "what?"}))
    assert "unavailable" in out

    from hud.skills import loader as loader

    loader.set_provider("screen_grabber", lambda timeout=10.0: b"PNGDATA")
    set_turn_context(chat_key="hud:test", owner="hud")

    async def fake_vision(session_key, owner, raw, fmt, question):
        assert session_key == "hud:test" and fmt == "png" and raw == b"PNGDATA"
        return "A blue window.", False

    import core

    monkeypatch.setattr(core, "vision_respond", fake_vision)
    out = asyncio.run(look_mod.look_at_screen.ainvoke({"question": "what?"}))
    assert "blue window" in out
    loader.set_provider("screen_grabber", None)


def test_goodbye_calls_provider():
    _reset()
    import hud.skills.packs.session as session_mod
    from hud.skills import loader as loader

    assert "close the window" in asyncio.run(session_mod.say_goodbye.ainvoke({}))
    fired = []
    loader.set_provider("on_goodbye", lambda: fired.append(True))
    out = asyncio.run(session_mod.say_goodbye.ainvoke({}))
    assert fired == [True] and "Goodbye" in out
    loader.set_provider("on_goodbye", None)


def test_wallpaper_validates(monkeypatch, tmp_path):
    import hud.skills.packs.wallpaper as wp_mod

    out = asyncio.run(wp_mod.set_wallpaper.ainvoke({"image_path": "C:/nope.png"}))
    assert "doesn't exist" in out
    pic = tmp_path / "bg.txt"
    pic.write_text("x", encoding="utf-8")
    out = asyncio.run(wp_mod.set_wallpaper.ainvoke({"image_path": str(pic)}))
    assert "isn't a supported image" in out
    monkeypatch.setattr(wp_mod.os, "name", "posix")
    pic2 = tmp_path / "bg.png"
    pic2.write_bytes(b"\x89PNG")
    assert "Windows only" in asyncio.run(wp_mod.set_wallpaper.ainvoke({"image_path": str(pic2)}))


def test_open_app_already_running(monkeypatch):
    import hud.skills.packs.apps as apps_mod

    monkeypatch.setattr(apps_mod, "_proc_names", lambda: {"spotify", "chrome"})
    target = {"key": "exe:x", "display": "Spotify", "path": "C:/x/Spotify.exe"}
    assert apps_mod._launch_verified(target, timeout=0.1) == "Spotify is already running."


def test_open_app_verify_success_and_failure(monkeypatch):
    import hud.skills.packs.apps as apps_mod

    monkeypatch.setattr(apps_mod, "_launch", lambda path: None)
    seen = {"n": 0}

    def procs():
        seen["n"] += 1
        return {"chrome"} if seen["n"] >= 2 else set()

    monkeypatch.setattr(apps_mod, "_proc_names", procs)
    target = {"key": "exe:x", "display": "chrome", "path": "C:/x/chrome.exe"}
    assert apps_mod._launch_verified(target, timeout=1.0, poll=0.01) == "Opened chrome (verified running)."

    monkeypatch.setattr(apps_mod, "_proc_names", lambda: set())
    out = apps_mod._launch_verified(target, timeout=0.05, poll=0.01)
    assert out.startswith("ERROR") and "isn't running" in out


def test_open_app_affirmative_confirms_pending(monkeypatch):
    _reset()
    import hud.skills.packs.apps as apps_mod

    monkeypatch.setattr(apps_mod, "resolve_app", lambda name: {"key": "k", "display": "FakeApp", "path": "C:/x/FakeApp.exe"})
    monkeypatch.setattr(apps_mod, "_launch", lambda path: None)
    polls = {"n": 0}

    def procs():
        polls["n"] += 1
        return {"fakeapp"} if polls["n"] >= 2 else set()

    monkeypatch.setattr(apps_mod, "_proc_names", procs)
    first = asyncio.run(apps_mod.open_app.ainvoke({"app_name": "fake"}))
    assert first.startswith("NEEDS-CONFIRM")
    # Bare "yes" (model forgot the name) still confirms the outstanding ask.
    second = asyncio.run(apps_mod.open_app.ainvoke({"app_name": "yes", "confirmed": True}))
    assert "verified running" in second


def test_open_app_mismatched_name_replaces(monkeypatch):
    _reset()
    import hud.skills.packs.apps as apps_mod

    def resolve(name):
        return {"key": f"k:{name}", "display": name.title(), "path": f"C:/{name}.exe"}

    monkeypatch.setattr(apps_mod, "resolve_app", resolve)
    asyncio.run(apps_mod.open_app.ainvoke({"app_name": "spotify"}))
    out = asyncio.run(apps_mod.open_app.ainvoke({"app_name": "chrome", "confirmed": True}))
    assert out.startswith("NEEDS-CONFIRM") and "Chrome" in out


def test_apps_direct_match(monkeypatch):
    _reset()
    import hud.skills.packs.apps as apps_mod

    monkeypatch.setattr(apps_mod, "resolve_app", lambda name: {"key": "k", "display": "Chrome", "path": "C:/c.exe"} if "chrome" in name.lower() else None)
    assert apps_mod.direct_match("Open Chrome", {}) == ("open_app", {"app_name": "Chrome", "confirmed": False})
    assert apps_mod.direct_match("please open spotify app", {}) is None  # unknown here
    assert apps_mod.direct_match("C:/x.exe", {}) is None
    assert apps_mod.direct_match("tell me about chrome", {}) is None
    # Bare yes with nothing pending -> None (agent handles).
    assert apps_mod.direct_match("yes", {}) is None


def test_apps_direct_match_yes_confirms(monkeypatch):
    _reset()
    import hud.skills.packs.apps as apps_mod

    apps_mod._PENDING = ({"key": "k", "display": "Chrome", "path": "C:/c.exe"}, apps_mod.time.monotonic() + 60)
    try:
        assert apps_mod.direct_match("yes", {}) == ("open_app", {"app_name": "Chrome", "confirmed": True})
        assert apps_mod.direct_match("YES PLEASE!", {}) == ("open_app", {"app_name": "Chrome", "confirmed": True})
    finally:
        apps_mod._PENDING = None


def test_core_direct_route_hit_and_miss(monkeypatch):
    import core
    from langchain_core.tools import tool

    @tool
    def ping_tool() -> str:
        """Pings."""
        return "pong-ci"

    calls = []

    async def fake_run_agent(messages, **kw):
        calls.append(True)
        return "agent answer", []

    monkeypatch.setattr(core, "run_agent", fake_run_agent)

    def matcher(text, context):
        if "ping now" in text.lower():
            return ("ping_tool", {})
        return None

    key = "test-direct-hit"
    try:
        body, sources, failed = asyncio.run(
            core.respond(key, "ping now please", extra_tools=[ping_tool], direct_routes=[matcher])
        )
        assert body == "pong-ci" and failed is False
        assert calls == []  # agent bypassed
    finally:
        core.chat_histories.pop(key, None)

    key2 = "test-direct-miss"
    try:
        body, _, _ = asyncio.run(
            core.respond(key2, "unrelated chat", extra_tools=[ping_tool], direct_routes=[matcher])
        )
        assert body == "agent answer"
        assert len(calls) == 1
    finally:
        core.chat_histories.pop(key2, None)


def test_core_direct_route_unknown_tool_skipped(monkeypatch):
    import core

    async def fake_run_agent(messages, **kw):
        return "agent answer", []

    monkeypatch.setattr(core, "run_agent", fake_run_agent)

    def matcher(text, context):
        return ("no_such_tool", {})

    key = "test-direct-unknown"
    try:
        body, _, _ = asyncio.run(core.respond(key, "hi", direct_routes=[matcher]))
        assert body == "agent answer"
    finally:
        core.chat_histories.pop(key, None)


def test_reminder_direct_match_shapes():
    from hud.skills.packs.reminders import direct_match

    assert direct_match("Remind me in 2 mins for dinner", {}) == (
        "remind_me", {"action": "add", "message": "dinner", "when": "in 2 mins", "target": ""})
    assert direct_match("show my reminders", {})[0] == "remind_me"
    assert direct_match("cancel my dinner reminder", {})[0] == "remind_me"
    assert direct_match("what is the weather", {}) is None
    assert direct_match("remind me to call mom", {}) is None  # no time -> agent clarifies


def test_system_status_runs():
    import hud.skills.packs.system_info as sys_mod

    out = asyncio.run(sys_mod.system_status.ainvoke({}))
    assert "CPU" in out and "MEM" in out
    out2 = asyncio.run(sys_mod.desk_stats.ainvoke({}))
    assert "free of" in out2 or "ERROR" in out2


_APP = None


def _qapp():
    global _APP
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def test_skills_checkbox_arms_tier():
    import os

    os.environ.pop("HUD_SKILLS_ALLOW", None)
    from hud.skills import loader as loader_mod
    from hud.window import MainWindow

    _qapp()
    win = MainWindow()
    try:
        assert win.skills_check.isChecked() is False
        win._on_skills_toggled(True)
        assert loader_mod._system_armed is True
        assert "on" in win.skills_check.text()
        win._on_skills_toggled(False)
        assert loader_mod._system_armed is False
        assert "off" in win.skills_check.text()
    finally:
        loader_mod._system_armed = False
        loader_mod._providers.clear()
        loader_mod._registry = {}
        win.close()
        win.deleteLater()


def test_extra_tools_default_unchanged(monkeypatch):
    """Telegram path: no extra_tools -> global-agent belt, as before."""
    import core

    captured = {}

    async def fake_run_agent(messages, **kw):
        captured.update(kw)
        return "ok", []

    monkeypatch.setattr(core, "run_agent", fake_run_agent)
    key = "test-skills-default"
    try:
        body, _, _ = asyncio.run(core.respond(key, "hi"))
        assert body == "ok"
        assert captured.get("tools") is None
    finally:
        core.chat_histories.pop(key, None)


def test_extra_tools_merged(monkeypatch):
    from langchain_core.tools import tool

    import core

    @tool
    def hud_ping() -> str:
        """Pings the HUD."""
        return "pong"

    captured = {}

    async def fake_run_agent(messages, **kw):
        captured["tools"] = kw.get("tools")
        return "ok", []

    monkeypatch.setattr(core, "run_agent", fake_run_agent)
    key = "test-skills-extra"
    try:
        asyncio.run(core.respond(key, "hi", extra_tools=[hud_ping]))
        names = {t.name for t in captured["tools"]}
        assert "hud_ping" in names and "web_search" in names  # appended, not replaced
    finally:
        core.chat_histories.pop(key, None)
