"""CodeBrain streaming: thinking events (+cap) and empty-round recovery.

All LLM I/O is mocked — no Ollama needed.
"""

import asyncio

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from code_assistant.brain import CodeBrain
from code_assistant.modes import Mode
from code_assistant.workspace import Workspace
from llm_provider import LLMConfig


def _brain(tmp_path):
    ws = Workspace(tmp_path)
    cfg = LLMConfig(model="test-model")
    return CodeBrain(workspace=ws, llm_config=cfg, tracker=None, mode=Mode.PLAN)


def test_thinking_events_streamed(tmp_path):
    b = _brain(tmp_path)

    async def fake(messages):
        yield AIMessageChunk(content="", additional_kwargs={"reasoning_content": "let me think"})
        yield AIMessageChunk(content="hello")

    class FakeLLM:
        def astream(self, messages):
            return fake(messages)

    b._llm_with_tools = FakeLLM()

    async def go():
        return await b._stream_model_round([HumanMessage(content="hi")])

    evs, ai = asyncio.run(go())
    kinds = [e.type for e in evs]
    assert "thinking" in kinds and "token" in kinds
    assert ai.content == "hello"


def test_thinking_capped_at_4k(tmp_path):
    from code_assistant import brain as brain_mod

    b = _brain(tmp_path)

    async def big(messages):
        yield AIMessageChunk(content="", additional_kwargs={"reasoning_content": "z" * 9000})
        yield AIMessageChunk(content="done")

    class FakeLLM:
        def astream(self, messages):
            return big(messages)

    b._llm_with_tools = FakeLLM()

    async def go():
        return await b._stream_model_round([HumanMessage(content="hi")])

    evs, _ = asyncio.run(go())
    total = sum(len(e.data.get("data", "")) for e in evs if e.type == "thinking")
    assert total <= brain_mod.MAX_REASONING_CHARS


def test_empty_round_nudges_instead_of_giving_up(tmp_path):
    b = _brain(tmp_path)
    calls = {"n": 0}

    async def fake_round(messages):
        from code_assistant.brain import BrainEvent

        calls["n"] += 1
        if calls["n"] == 1:
            return [], AIMessage(content="")  # flaked round
        return [BrainEvent(type="token", data={"data": "hi"})], AIMessage(content="hi")

    b._stream_model_round = fake_round
    replies = []

    async def go():
        async for ev in b.run("hello"):
            if ev.type == "done":
                replies.append(ev.data["reply"])

    asyncio.run(go())
    assert replies == ["hi"], replies
    assert calls["n"] == 2


def test_structured_calls_from_any_chunk(tmp_path):
    b = _brain(tmp_path)

    async def fake(messages):
        yield AIMessageChunk(
            content="",
            tool_calls=[{"id": "a1", "name": "list_files", "args": {"path": ""}}],
        )
        yield AIMessageChunk(content="")

    class FakeLLM:
        def astream(self, messages):
            return fake(messages)

    b._llm_with_tools = FakeLLM()

    async def go():
        return await b._stream_model_round([HumanMessage(content="hi")])

    _, ai = asyncio.run(go())
    assert ai.tool_calls and ai.tool_calls[0]["name"] == "list_files"
