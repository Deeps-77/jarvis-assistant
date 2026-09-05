"""SessionStore: round-trip, isolation, cap, rename/delete, corrupt tolerance."""

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from code_assistant.sessions import SessionStore, auto_title


def _msgs():
    return [
        HumanMessage(content="list files"),
        AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "list_files", "args": {"path": ""}}],
        ),
        ToolMessage(content="a.py", tool_call_id="c1", name="list_files"),
        AIMessage(content="done"),
    ]


def test_round_trip_with_tool_calls(tmp_path):
    store = SessionStore(tmp_path / "s.json")
    s = store.create("W1", title="t1", mode="plan")
    store.save("W1", s.id, _msgs(), "build", title="renamed")
    got = store.get("W1", s.id[:6])  # id-prefix match
    assert got is not None
    assert (got.title, got.mode) == ("renamed", "build")
    assert len(got.messages) == 4
    assert type(got.messages[1]).__name__ == "AIMessage"
    assert got.messages[1].tool_calls[0]["name"] == "list_files"
    assert type(got.messages[2]).__name__ == "ToolMessage"
    assert got.messages[2].tool_call_id == "c1"


def test_disk_reload(tmp_path):
    p = tmp_path / "s.json"
    s = SessionStore(p).create("W1", title="t", mode="plan")
    SessionStore(p).save("W1", s.id, _msgs(), "plan")
    got = SessionStore(p).get("W1", s.id)
    assert got is not None and len(got.messages) == 4 and got.title == "t"


def test_workspace_isolation(tmp_path):
    store = SessionStore(tmp_path / "s.json")
    store.create("W1", title="a")
    store.create("W2", title="b")
    assert len(store.list("W1")) == 1
    assert len(store.list("W2")) == 1


def test_cap_prunes_oldest(tmp_path):
    store = SessionStore(tmp_path / "s.json", max_per_workspace=20)
    for i in range(25):
        store.create("W3", title=f"s{i}")
    lst = store.list("W3")
    assert len(lst) == 20
    assert lst[0].title == "s24"  # most recent first


def test_rename_and_delete(tmp_path):
    store = SessionStore(tmp_path / "s.json")
    s = store.create("W1", title="old")
    assert store.rename("W1", s.id, "  new name  ").title == "new name"
    assert store.rename("W1", s.id, "   ") is None
    assert store.delete("W1", s.id) is not None
    assert store.get("W1", s.id) is None
    assert store.list("W1") == []


def test_corrupt_file_starts_empty(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{broken", encoding="utf-8")
    assert SessionStore(p).list("W") == []


def test_auto_title():
    assert auto_title("  hello world  ") == "hello world"
    assert auto_title("") == "Untitled session"
    assert len(auto_title("x" * 100)) <= 41


def test_malformed_rows_skipped(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(
        json.dumps({"W": [{"id": "", "workspace_root": ""}, {"id": "a", "workspace_root": "W"}]}),
        encoding="utf-8",
    )
    lst = SessionStore(p).list("W")
    assert len(lst) == 1 and lst[0].id == "a"
