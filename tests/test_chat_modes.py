"""Chat modes (normal/quick/research/docs) for the web UI gear panel.

All LLM/tool I/O is mocked — no Ollama needed.
"""

import asyncio

import core


def test_normalize_chat_mode():
    assert core.normalize_chat_mode(None) == "normal"
    assert core.normalize_chat_mode("") == "normal"
    assert core.normalize_chat_mode("nonsense") == "normal"
    assert core.normalize_chat_mode("Quick") == "quick"
    assert core.normalize_chat_mode(" RESEARCH ") == "research"
    assert core.normalize_chat_mode("docs") == "docs"
    assert set(core.CHAT_MODES) == {"normal", "quick", "research", "docs"}


def test_tools_for_mode():
    assert core._tools_for_mode("normal") is None
    assert core._tools_for_mode("research") is None
    docs = core._tools_for_mode("docs")
    assert {t.name for t in docs} == {
        "search_documents",
        "summarize_document",
        "list_documents",
    }


class _FakeLLM:
    def __init__(self):
        self.calls = []

    async def ainvoke(self, messages):
        self.calls.append(messages)
        from langchain_core.messages import AIMessage

        return AIMessage(content="direct answer")

    async def astream(self, messages):
        from langchain_core.messages import AIMessageChunk

        self.calls.append(messages)
        yield AIMessageChunk(content="direct ")
        yield AIMessageChunk(content="answer")


def _clean(key):
    core.chat_histories.pop(key, None)


def test_quick_mode_skips_agent(monkeypatch):
    fake = _FakeLLM()
    monkeypatch.setattr(core, "llm", fake)

    async def boom(*a, **k):
        raise AssertionError("agent must not run in quick mode")

    monkeypatch.setattr(core, "run_agent", boom)
    key = "test-mode-quick"
    try:
        body, sources, failed = asyncio.run(core.respond(key, "hi there", mode="quick"))
        assert body == "direct answer"
        assert sources == [] and failed is False
        assert fake.calls, "llm was never called"
    finally:
        _clean(key)


def test_quick_mode_streams_tokens(monkeypatch):
    fake = _FakeLLM()
    monkeypatch.setattr(core, "llm", fake)
    seen = []

    async def collect(token):
        seen.append(token)

    async def go():
        return await core._run_direct([], on_token=collect)

    assert asyncio.run(go()) == "direct answer"
    assert "".join(seen) == "direct answer"


def test_invalid_mode_falls_back_to_normal(monkeypatch):
    captured = {}

    async def fake_run_agent(messages, **kw):
        captured.update(kw)
        return "ok", []

    monkeypatch.setattr(core, "run_agent", fake_run_agent)
    key = "test-mode-invalid"
    try:
        body, _, _ = asyncio.run(core.respond(key, "hi", mode="bogus"))
        assert body == "ok"
        assert captured.get("tools") is None  # full-belt global agent path
    finally:
        _clean(key)


def test_docs_mode_restricts_tools(monkeypatch):
    captured = {}

    async def fake_run_agent(messages, **kw):
        captured["tools"] = kw.get("tools")
        return "from docs", []

    monkeypatch.setattr(core, "run_agent", fake_run_agent)
    key = "test-mode-docs"
    try:
        body, _, _ = asyncio.run(core.respond(key, "what is in my files?", mode="docs"))
        assert body == "from docs"
        assert {t.name for t in captured["tools"]} == {
            "search_documents",
            "summarize_document",
            "list_documents",
        }
    finally:
        _clean(key)


def test_research_mode_mandatory_search(monkeypatch):
    from langchain_core.messages import SystemMessage

    presearch_calls = []

    async def fake_presearch(query):
        presearch_calls.append(query)
        return "RESEARCH-CONTEXT", ["http://fresh.example/x"]

    captured = {}

    async def fake_run_agent(messages, **kw):
        captured["messages"] = messages
        assert any(
            isinstance(m, SystemMessage) and "RESEARCH-CONTEXT" in str(m.content)
            for m in messages
        ), "pre-search context missing from agent input"
        return "researched answer [1]", ["http://agent.example/y"]

    monkeypatch.setattr(core, "_mandatory_presearch", fake_presearch)
    monkeypatch.setattr(core, "run_agent", fake_run_agent)
    key = "test-mode-research"
    try:
        body, sources, _ = asyncio.run(
            core.respond(key, "iphone price?", mode="research")
        )
        assert presearch_calls == ["iphone price?"]
        assert body == "researched answer [1]"
        # Pre-search URLs come first (freshest), then agent sources.
        assert sources == ["http://fresh.example/x", "http://agent.example/y"]
    finally:
        _clean(key)


def test_research_proceeds_when_presearch_fails(monkeypatch):
    async def fake_presearch(query):
        return "", []

    async def fake_run_agent(messages, **kw):
        return "best effort", []

    monkeypatch.setattr(core, "_mandatory_presearch", fake_presearch)
    monkeypatch.setattr(core, "run_agent", fake_run_agent)
    key = "test-mode-research-fail"
    try:
        body, _, _ = asyncio.run(core.respond(key, "iphone price?", mode="research"))
        assert body == "best effort"
    finally:
        _clean(key)


def test_mandatory_presearch_merges_and_dedupes(monkeypatch):
    seen_queries = []

    class FakeSearch:
        async def ainvoke(self, args):
            seen_queries.append(args["query"])
            if "2026" in args["query"] or "20" in args["query"]:
                return "fresh result\nURL: http://x.example/a"
            return "base result\nURL: http://x.example/a\nURL: http://x.example/b"

    monkeypatch.setattr(core, "web_search", FakeSearch())
    text, urls = asyncio.run(core._mandatory_presearch("iphone price"))
    assert len(seen_queries) == 2  # verbatim + freshness variant
    assert seen_queries[0] == "iphone price"
    assert "URL: http://x.example/a" in text
    assert urls == ["http://x.example/a", "http://x.example/b"]  # deduped


def test_mandatory_presearch_never_raises(monkeypatch):
    class Boom:
        async def ainvoke(self, args):
            raise RuntimeError("ddgs down")

    monkeypatch.setattr(core, "web_search", Boom())
    assert asyncio.run(core._mandatory_presearch("q")) == ("", [])
