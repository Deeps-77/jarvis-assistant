import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

import botlog
from botlog import setup_logging
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    messages_from_dict,
    messages_to_dict,
)
from langchain_ollama import ChatOllama
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent
from memory import MemoryStore
from tools import (
    date_calculator,
    get_crypto_price,
    get_current_time,
    get_exchange_rate,
    get_weather,
    web_search,
)

MAX_HISTORY_MESSAGES = 16
MAX_TOOL_ROUNDS = 4


def load_env_file(env_path: Path | None = None):
    env_path = env_path or Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


load_env_file()

MODEL_NAME = os.environ.get("OLLAMA_MODEL", "hf.co/LiquidAI/LFM2.5-2.6B-GGUF:latest")
print(MODEL_NAME)
HISTORY_FILE = Path(__file__).parent / "chat_history.json"

SYSTEM_PROMPT = """You are Jarvis, a personal AI assistant on Telegram.
Identity rules:
- Your name is Jarvis.Never mention your underlying model or its maker.
- If asked who you are, what your name is, or what model you are, you simply say you are Jarvis.
- The User Name is Deepak M. He is your master and you are his companion.
The current date and time is provided in a system message at the start of every turn - trust it above all other sources.
NEVER call web_search for the current date, day of the week, or clock time; answer those directly from the provided date.
For math, logic puzzles, coding, translation, definitions of common concepts, and creative writing: answer directly from your own knowledge and NEVER call web_search.
Use web_search ONLY when the question needs real-time information: news, prices, weather, sports scores, product rankings, or recent events.
Dedicated tools give exact live facts and are preferred over web_search when they match: get_current_time for time or date anywhere in the world, date_calculator for calendar math, get_weather for current weather, get_exchange_rate for currency rates, get_crypto_price for cryptocurrency prices.
When you do search: call web_search at most twice per question, never repeat a query you already tried, then answer using ONLY the results and cite them like [1][2].
When judging search results, compare their dates against the current date and say so if they look outdated.
If search results don't contain the answer, say you don't know instead of guessing.
Never invent facts, dates, or numbers.
Remember: you are Jarvis."""

SPECIAL_TOKEN_RE = re.compile(r"<\|.*?\|>")

setup_logging()
logger = logging.getLogger(__name__)

chat_histories: dict[str, list] = {}
memory_store: MemoryStore | None = None
_background_tasks: set = set()


def save_histories():
    try:
        data = {str(cid): messages_to_dict(msgs) for cid, msgs in chat_histories.items()}
        tmp = HISTORY_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(HISTORY_FILE)
    except Exception:
        logger.exception("Failed to save chat history")


def load_histories():
    if not HISTORY_FILE.exists():
        return
    try:
        raw = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
        for cid, msgs in raw.items():
            history = messages_from_dict(msgs)
            trim_history(history)
            chat_histories[cid] = history
    except Exception:
        logger.exception("Failed to load chat history; starting fresh")


def init_memory(db_path: Path):
    global memory_store
    try:
        memory_store = MemoryStore(db_path)
    except Exception as e:
        logger.warning("Memory store init failed (%s); continuing without it", e)


def clear_session(session_key: str):
    chat_histories.pop(session_key, None)
    save_histories()


llm = ChatOllama(model=MODEL_NAME, temperature=0.2, num_ctx=8192, timeout=600, keep_alive=-1)

TOOLBELT = [
    web_search,
    get_current_time,
    date_calculator,
    get_weather,
    get_exchange_rate,
    get_crypto_price,
]

agent = create_react_agent(llm, tools=TOOLBELT)

FORCE_FINAL_PROMPT = (
    "Stop searching. Write your final answer NOW using ONLY the information you "
    "already gathered above. Cite sources like [1][2]. If something could not be "
    "verified, say so explicitly instead of guessing."
)


def _tool_names(messages: list) -> list[str]:
    return [
        tc["name"]
        for m in messages
        if getattr(m, "tool_calls", None)
        for tc in m.tool_calls
    ]


async def run_agent(messages: list) -> tuple[str, list[str]]:
    generated: list = []
    config = {"recursion_limit": MAX_TOOL_ROUNDS * 2 + 4}
    try:
        async for update in agent.astream({"messages": messages}, config=config):
            for node_output in update.values():
                if isinstance(node_output, list):
                    generated.extend(node_output)
                elif isinstance(node_output, dict) and "messages" in node_output:
                    generated.extend(node_output["messages"])
        botlog.log_tools(_tool_names(generated))
        return content_to_str(generated[-1].content), extract_sources(generated)
    except GraphRecursionError:
        logger.warning("Tool-round cap hit; forcing final answer from gathered context")
        botlog.log_tools(_tool_names(generated))
        forced = await llm.ainvoke(
            messages + generated + [HumanMessage(content=FORCE_FINAL_PROMPT)]
        )
        return content_to_str(forced.content), extract_sources(generated)


def sanitize(text: str) -> str:
    return SPECIAL_TOKEN_RE.sub("", text).strip()


def enforce_identity(text: str) -> str:
    t = text.replace("LFM (Liquid Foundation Model)", "Jarvis")
    t = re.sub(r"(?:the\s+)?liquid\s+foundation\s+model", "Jarvis", t, flags=re.IGNORECASE)
    t = re.sub(r"\blfm\b", "Jarvis", t, flags=re.IGNORECASE)
    t = re.sub(
        r",?\s*\b(?:built|made|created|developed|designed|trained)\s+by\s+liquid\s*ai\b",
        "",
        t,
        flags=re.IGNORECASE,
    )
    t = re.sub(r",?\s*\bby\s+liquid\s*ai\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r",?\s*\bfrom\s+liquid\s*ai\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bliquid\s*ai\b", "my developers", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+([.,!?])", r"\1", t)
    t = re.sub(r" {2,}", " ", t)
    return t


def content_to_str(content) -> str:
    if isinstance(content, str):
        return content
    return " ".join(str(p) for p in content)


def trim_history(history: list):
    while len(history) > MAX_HISTORY_MESSAGES:
        history.pop(0)
    while history and not isinstance(history[0], HumanMessage):
        history.pop(0)


def extract_sources(messages: list) -> list[str]:
    urls, seen = [], set()
    for m in messages:
        if isinstance(m, ToolMessage):
            for url in re.findall(r"URL: (\S+)", content_to_str(m.content)):
                if url not in seen:
                    seen.add(url)
                    urls.append(url)
    return urls


FRAMEWORK_FAILURE_PHRASES = (
    "recursion limit",
    "iteration limit",
    "agent stopped",
)


def looks_like_failure(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 2:
        return True
    low = stripped.lower()
    if "sorry, need more steps" in low:
        return True
    return any(
        low.startswith(phrase) or (len(low) < 120 and phrase in low)
        for phrase in FRAMEWORK_FAILURE_PHRASES
    )


async def respond(session_key: str, text: str) -> tuple[str, list[str], bool]:
    history = chat_histories.setdefault(session_key, [])
    history.append(HumanMessage(content=text))
    trim_history(history)

    now = datetime.now().astimezone()
    preamble = (
        f"Current date and time: {now:%A}, {now:%d %B %Y}, {now:%I:%M %p} "
        f"({now:%Z}, UTC{now:%z}). Trust this above any other date information."
    )
    if memory_store and memory_store.enabled:
        t0 = time.perf_counter()
        recall_texts = await memory_store.search(session_key, text)
        if recall_texts:
            preamble += (
                "\nEarlier conversation excerpts that may be relevant:\n"
                + "\n---\n".join(t.replace("\n", " ")[:400] for t in recall_texts)
                + "\nUse them only if relevant to the current question."
            )
        logger.debug("memory.recall took %.0fms", (time.perf_counter() - t0) * 1000)

    system_text = f"{SYSTEM_PROMPT}\n\n{preamble}"
    agent_messages = [SystemMessage(content=system_text)] + list(history)

    raw_reply, sources = await run_agent(agent_messages)
    reply_md = enforce_identity(sanitize(raw_reply))
    sources = sources[:5]

    failed = looks_like_failure(reply_md)
    if failed:
        logger.warning("Gated failure-style model output: %r", reply_md[:300])
        if sources:
            body = "I found sources but couldn't write a reliable summary. Here's what I found:"
            stored_reply = "I wasn't able to write a reliable summary from my last search."
        else:
            body = "I couldn't produce a reliable answer for that. Please try rephrasing."
            stored_reply = body
    else:
        body = reply_md
        stored_reply = reply_md

    history.append(AIMessage(content=stored_reply))

    footer_sources = [] if (not failed and "couldn't find" in body.lower()) else sources

    if memory_store and memory_store.enabled:
        exchange = f"User: {text}\nAssistant: {stored_reply}"
        task = asyncio.create_task(memory_store.add(session_key, exchange))
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)

    return body, footer_sources, failed
