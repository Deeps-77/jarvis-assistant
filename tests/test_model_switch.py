"""Model listing / switching helpers. All Ollama I/O is mocked."""

import asyncio
import json

import core


class _FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if not (200 <= self.status_code < 300):
            raise RuntimeError(f"http {self.status_code}")

    def json(self):
        return self._payload


class _FakeClient:
    last_post = None

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url):
        assert url.endswith("/api/tags")
        return _FakeResp(
            {
                "models": [
                    {"name": "model-a:latest", "size": 2_000_000_000, "modified_at": "t"},
                    {"name": "model-b:7b", "size": 100, "modified_at": "t"},
                ]
            }
        )

    async def post(self, url, json=None):
        _FakeClient.last_post = (url, json)
        return _FakeResp({})


def test_list_ollama_models(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    models = asyncio.run(core.list_ollama_models())
    assert [m["name"] for m in models] == ["model-a:latest", "model-b:7b"]


def test_persist_ollama_model_env_roundtrip(tmp_path, monkeypatch):
    import core as core_mod

    tmp_mod = tmp_path / "core.py"
    tmp_mod.write_text("x=1", encoding="utf-8")
    (tmp_path / ".env").write_text(
        "# comment\nTELEGRAM_BOT_TOKEN=x\nOLLAMA_MODEL=old:1\nOTHER=1\n",
        encoding="utf-8",
    )
    orig_file = core_mod.__file__
    try:
        core_mod.__file__ = str(tmp_mod)
        out = core_mod.persist_ollama_model_env("new:2")
        assert out == tmp_path / ".env"
        text = out.read_text(encoding="utf-8")
        assert "OLLAMA_MODEL=new:2" in text
        assert "TELEGRAM_BOT_TOKEN=x" in text
        assert text.count("OLLAMA_MODEL=") == 1
    finally:
        core_mod.__file__ = orig_file


def test_set_chat_model_swaps_and_unloads(monkeypatch):
    created = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            created.setdefault("llms", []).append(kwargs)

    def fake_agent(llm, tools=None):
        created["agent_llm"] = llm
        return object()

    monkeypatch.setattr(core, "ChatOllama", FakeLLM)
    monkeypatch.setattr(core, "create_react_agent", fake_agent)
    monkeypatch.setattr(core, "persist_ollama_model_env", lambda name: None)

    unloaded = {}

    async def fake_unload(name):
        unloaded["name"] = name
        return True

    monkeypatch.setattr(core, "_unload_ollama_model", fake_unload)
    # Neutralize memory extractor rebuild.
    if core.memory_store is not None:
        monkeypatch.setattr(core.memory_store, "extractor", None)

    old = core.MODEL_NAME
    new = "model-b:7b" if old != "model-b:7b" else "model-a:latest"
    got_old = asyncio.run(core.set_chat_model(new))
    assert got_old == old
    assert core.MODEL_NAME == new
    assert unloaded["name"] == old
    # Same-name switch is a no-op returning current.
    assert asyncio.run(core.set_chat_model(new)) == new


def test_setup_logging_idempotent():
    import botlog

    botlog.setup_logging()
    first = len(botlog._activity.handlers)
    botlog.setup_logging()
    assert len(botlog._activity.handlers) == first == 1
