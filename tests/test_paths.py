"""paths.py: legacy fallback + migrate-once semantics (sandboxed via monkeypatch)."""

import paths


def test_data_path_prefers_new_location(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    target = paths.data_path("x.db")
    assert target == tmp_path / "data" / "x.db"


def test_data_path_migrates_legacy_once(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    legacy = tmp_path / "x.db"
    legacy.write_text("v", encoding="utf-8")
    assert paths.data_path("x.db") == tmp_path / "data" / "x.db"
    assert not legacy.exists()
    assert (tmp_path / "data" / "x.db").read_text(encoding="utf-8") == "v"
    # second call is stable (no duplicate, no error)
    assert paths.data_path("x.db") == tmp_path / "data" / "x.db"


def test_data_path_never_overwrites_target(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "x.db").write_text("new", encoding="utf-8")
    (tmp_path / "x.db").write_text("legacy", encoding="utf-8")
    assert paths.data_path("x.db") == tmp_path / "data" / "x.db"
    assert (tmp_path / "data" / "x.db").read_text(encoding="utf-8") == "new"


def test_helpers_use_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(paths, "PROJECT_ROOT", tmp_path)
    assert paths.memory_db() == tmp_path / "data" / "memory.db"
    assert paths.chat_threads_db() == tmp_path / "data" / "chat_threads.db"
    assert paths.code_sessions_file() == tmp_path / "data" / "code_sessions.json"
