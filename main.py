import asyncio
import html
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

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
from markdown_it import MarkdownIt
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, TimedOut
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

MAX_MESSAGE_LEN = 4096
TG_CHUNK_LIMIT = 4000
MAX_HISTORY_MESSAGES = 16
MAX_TOOL_ROUNDS = 4

_md_parser = MarkdownIt("commonmark").enable(["table", "strikethrough"])
_ALLOWED_TAG_RE = re.compile(
    r"</?(b|strong|i|em|u|ins|s|strike|del|code|pre|a|blockquote)(\s[^>]*)?/?>"
)


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
- Your name is Jarvis. NEVER refer to yourself as LFM, Liquid Foundation Model, or as a product made by Liquid AI. Never mention your underlying model or its maker.
- If asked who you are, what your name is, or what model you are, you simply say you are Jarvis.
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

chat_histories: dict[int, list] = {}
allowed_ids: set[int] | None = None
owner_id: int | None = None
memory_store: MemoryStore | None = None
_background_tasks: set = set()


def is_authorized(update: Update) -> bool:
    if allowed_ids is None:
        return True
    user = update.effective_user
    return user is not None and user.id in allowed_ids


def denial_text(update: Update) -> str:
    user = update.effective_user
    return (
        f"⛔ Not authorized. Your Telegram ID is {user.id if user else 'unknown'} — "
        "ask the bot owner to allowlist it."
    )


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
            chat_histories[int(cid)] = history
    except Exception:
        logger.exception("Failed to load chat history; starting fresh")


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


def _tool_names(messages: list) -> list[str]:
    return [
        tc["name"]
        for m in messages
        if getattr(m, "tool_calls", None)
        for tc in m.tool_calls
    ]


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


def split_message(text: str) -> list[str]:
    text = text.strip()
    if len(text) <= MAX_MESSAGE_LEN:
        return [text]
    return [text[i : i + MAX_MESSAGE_LEN] for i in range(0, len(text), MAX_MESSAGE_LEN)]


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


def format_sources(sources: list[str]) -> str:
    return "\n".join(f"[{i}] {url}" for i, url in enumerate(sources, start=1))


def _esc(text: str) -> str:
    return html.escape(str(text))


def _inline_text(children: list) -> str:
    parts = []
    for t in children:
        if t.type == "text":
            parts.append(t.content)
        elif t.type == "code_inline":
            parts.append(t.content)
        elif t.children:
            parts.append(_inline_text(t.children))
    return "".join(parts)


def _inline_html(children: list) -> str:
    out = []
    for t in children:
        tt = t.type
        if tt == "text":
            out.append(_esc(t.content))
        elif tt == "code_inline":
            out.append(f"<code>{_esc(t.content)}</code>")
        elif tt == "strong_open":
            out.append("<b>")
        elif tt == "strong_close":
            out.append("</b>")
        elif tt == "em_open":
            out.append("<i>")
        elif tt == "em_close":
            out.append("</i>")
        elif tt == "s_open":
            out.append("<s>")
        elif tt == "s_close":
            out.append("</s>")
        elif tt == "link_open":
            href = t.attrGet("href") or ""
            out.append(f'<a href="{_esc(href)}">')
        elif tt == "link_close":
            out.append("</a>")
        elif tt == "image":
            src = t.attrGet("src") or ""
            alt = _inline_text(t.children or []) or src
            out.append(f'<a href="{_esc(src)}">{_esc(alt)}</a>')
        elif tt == "html_inline":
            out.append(_esc(t.content))
        elif tt in ("softbreak", "hardbreak"):
            out.append("\n")
        elif t.children:
            out.append(_inline_html(t.children))
    return "".join(out)


def _render_list_items(tokens: list, ordered: bool, start: int = 1) -> list[str]:
    items, i, n = [], 0, len(tokens)
    counter = start
    while i < n:
        t = tokens[i]
        if t.type != "list_item_open":
            i += 1
            continue
        depth, j = 1, i + 1
        while j < n and depth:
            if tokens[j].type == "list_item_open":
                depth += 1
            elif tokens[j].type == "list_item_close":
                depth -= 1
            j += 1
        inner = _render_blocks(tokens[i + 1 : j - 1])
        if inner:
            prefix = f"{counter}. " if ordered else "• "
            inner[0] = prefix + inner[0]
            items.append("\n".join(inner))
        counter += 1
        i = j
    return items


def _render_table(tokens: list) -> str | None:
    rows = []
    i, n = 0, len(tokens)
    while i < n:
        if tokens[i].type == "tr_open":
            depth, j = 1, i + 1
            while j < n and depth:
                if tokens[j].type == "tr_open":
                    depth += 1
                elif tokens[j].type == "tr_close":
                    depth -= 1
                j += 1
            cells, k = [], i + 1
            while k < j - 1:
                tk = tokens[k]
                if tk.type in ("th_open", "td_open"):
                    d2, m = 1, k + 1
                    while m < j - 1 and d2:
                        if tokens[m].type in ("th_open", "td_open"):
                            d2 += 1
                        elif tokens[m].type in ("th_close", "td_close"):
                            d2 -= 1
                        m += 1
                    cell = "".join(
                        _inline_text(x.children or []) for x in tokens[k + 1 : m - 1] if x.type == "inline"
                    )
                    cells.append(cell.strip())
                    k = m
                else:
                    k += 1
            rows.append(cells)
            i = j
        else:
            i += 1
    if not rows:
        return None
    width = max(len(r) for r in rows)
    for r in rows:
        r.extend([""] * (width - len(r)))
    cols = [min(40, max(len(r[c]) for r in rows)) for c in range(width)]
    lines = [" | ".join(r[c].ljust(cols[c]) for c in range(width)) for r in rows]
    lines.insert(1, "-+-".join("-" * c for c in cols))
    return "\n".join(lines)


def _render_container(name: str, open_tok, inner: list) -> str | None:
    if name == "bullet_list":
        return "\n".join(_render_list_items(inner, ordered=False))
    if name == "ordered_list":
        try:
            start = int(open_tok.attrGet("start") or 1)
        except (TypeError, ValueError):
            start = 1
        return "\n".join(_render_list_items(inner, ordered=True, start=start))
    if name == "blockquote":
        parts = _render_blocks(inner)
        return "<blockquote>" + "\n".join(parts) + "</blockquote>" if parts else None
    if name == "table":
        return _render_table(inner)
    return None


def _render_blocks(tokens: list) -> list[str]:
    blocks, i, n = [], 0, len(tokens)
    while i < n:
        t = tokens[i]
        tt = t.type
        if tt == "inline":
            txt = _inline_html(t.children or [])
            if txt.strip():
                blocks.append(txt)
            i += 1
        elif tt in ("fence", "code_block"):
            lang = (t.info or "").strip().split(" ")[0]
            code = _esc(t.content.rstrip("\n"))
            if lang:
                blocks.append(f'<pre><code class="language-{_esc(lang)}">{code}</code></pre>')
            else:
                blocks.append(f"<pre>{code}</pre>")
            i += 1
        elif tt == "heading_open":
            inner = _inline_html(tokens[i + 1].children or []) if i + 2 < n else ""
            blocks.append(f"<b>{inner}</b>")
            i += 3
        elif tt == "hr":
            blocks.append("────────────────────")
            i += 1
        elif tt in ("html_block", "html_inline"):
            if t.content.strip():
                blocks.append(_esc(t.content.strip()))
            i += 1
        elif tt.endswith("_open"):
            name = tt[: -len("_open")]
            if name in ("paragraph", "list_item", "thead", "tbody", "heading"):
                i += 1
                continue
            depth, j = 1, i + 1
            while j < n and depth:
                if tokens[j].type == f"{name}_open":
                    depth += 1
                elif tokens[j].type == f"{name}_close":
                    depth -= 1
                j += 1
            rendered = _render_container(name, t, tokens[i + 1 : j - 1])
            if rendered and rendered.strip():
                blocks.append(rendered)
            i = j
        else:
            i += 1
    return blocks


def render_markdown_blocks(md_text: str) -> list[str]:
    return [b for b in _render_blocks(_md_parser.parse(md_text)) if b.strip()]


def pack_blocks(blocks: list[str], limit: int = TG_CHUNK_LIMIT) -> list[str]:
    chunks, cur = [], ""
    for b in blocks:
        candidate = f"{cur}\n\n{b}" if cur else b
        if len(candidate) <= limit:
            cur = candidate
            continue
        if cur:
            chunks.append(cur)
        if len(b) <= limit:
            cur = b
        else:
            pieces, part = [], ""
            for line in b.split("\n"):
                cand = f"{part}\n{line}" if part else line
                if len(cand) <= limit:
                    part = cand
                else:
                    if part:
                        pieces.append(part)
                    while len(line) > limit:
                        pieces.append(line[:limit])
                        line = line[limit:]
                    part = line
            if part:
                pieces.append(part)
            chunks.extend(pieces[:-1])
            cur = pieces[-1] if pieces else ""
    if cur:
        chunks.append(cur)
    return chunks


def strip_tags(html_text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", html_text))


def domain_of(url: str) -> str:
    netloc = urlparse(url).netloc
    netloc = netloc.removeprefix("www.")
    return netloc or url


def sources_footer(sources: list[str]) -> str:
    lines = ["<b>Sources</b>"]
    for i, url in enumerate(sources, start=1):
        lines.append(f'<a href="{_esc(url)}">[{i}] {_esc(domain_of(url))}</a>')
    return "\n".join(lines)


async def send_reply(update: Update, text: str):
    chat = update.effective_chat
    for part in split_message(text):
        if not part:
            continue
        for attempt in range(3):
            try:
                await chat.send_message(part)
                break
            except TimedOut:
                if attempt == 2:
                    raise
                await asyncio.sleep(2 * (attempt + 1))


async def send_html_reply(update: Update, md_body: str, sources: list[str] | None = None):
    blocks = render_markdown_blocks(md_body)
    if sources:
        blocks.append(sources_footer(sources))
    chat = update.effective_chat
    for chunk in pack_blocks(blocks):
        parse_html = True
        for attempt in range(3):
            try:
                if parse_html:
                    try:
                        await chat.send_message(chunk, parse_mode=ParseMode.HTML)
                        break
                    except BadRequest:
                        logger.warning("HTML chunk rejected by Telegram; resending as plain text")
                        parse_html = False
                        continue
                else:
                    await chat.send_message(strip_tags(chunk))
                    break
            except TimedOut:
                if attempt == 2:
                    raise
                await asyncio.sleep(2 * (attempt + 1))


async def typing_indicator(bot, chat_id):
    try:
        while True:
            await bot.send_chat_action(chat_id, ChatAction.TYPING)
            await asyncio.sleep(4)
    except asyncio.CancelledError:
        pass
    except Exception:
        pass


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        botlog.log_denied(_user_name(update), update.effective_user.id)
        await update.effective_message.reply_text(denial_text(update))
        return
    botlog.log_command("start", _user_name(update), update.effective_user.id)
    chat_histories.pop(update.effective_chat.id, None)
    save_histories()
    await update.effective_message.reply_text(
        "Hello! I am your local AI Assistant. Ask me anything, "
        "and I will search the web when needed."
    )


async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        botlog.log_denied(_user_name(update), update.effective_user.id)
        await update.effective_message.reply_text(denial_text(update))
        return
    botlog.log_command("reset", _user_name(update), update.effective_user.id)
    chat_histories.pop(update.effective_chat.id, None)
    save_histories()
    await update.effective_message.reply_text("Conversation cleared.")


async def forget_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        botlog.log_denied(_user_name(update), update.effective_user.id)
        await update.effective_message.reply_text(denial_text(update))
        return
    botlog.log_command("forget", _user_name(update), update.effective_user.id)
    chat_id = update.effective_chat.id
    chat_histories.pop(chat_id, None)
    save_histories()
    if memory_store:
        await memory_store.clear_chat(str(chat_id))
    await update.effective_message.reply_text("Conversation and long-term memory cleared.")


def _user_name(update: Update) -> str:
    user = update.effective_user
    return getattr(user, "full_name", None) or f"id:{getattr(user, 'id', '?')}"


def _chat_desc(update: Update) -> str:
    chat = update.effective_chat
    return getattr(chat, "title", None) or f"{chat.type} chat"


async def logs_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if owner_id is None or user_id != owner_id:
        botlog.log_denied(_user_name(update), user_id)
        await update.effective_message.reply_text(
            "⛔ /logs is restricted to the bot owner."
        )
        return
    botlog.log_command("logs", _user_name(update), user_id)

    arg = (context.args[0] if context.args else "").strip()
    limit = min(int(arg), 50) if arg.isdigit() and int(arg) > 0 else 15

    log_path = Path(__file__).parent / "logs" / "activity.log"
    if not log_path.exists():
        await update.effective_message.reply_text("No activity recorded yet.")
        return
    lines = [ln for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    selected = lines[-limit:]
    if not selected:
        await update.effective_message.reply_text("Activity log is empty.")
        return

    payload = f"<pre>{_esc(chr(10).join(selected))}</pre>"
    while len(payload) > TG_CHUNK_LIMIT and len(selected) > 1:
        selected = selected[1:]
        payload = f"<pre>{_esc(chr(10).join(selected))}</pre>"

    chat = update.effective_chat
    for attempt in range(3):
        try:
            try:
                await chat.send_message(payload, parse_mode=ParseMode.HTML)
            except BadRequest:
                await chat.send_message(strip_tags(payload))
            break
        except TimedOut:
            if attempt == 2:
                raise
            await asyncio.sleep(2 * (attempt + 1))


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        logger.warning("Unauthorized access attempt from user %s", update.effective_user.id)
        botlog.log_denied(_user_name(update), update.effective_user.id)
        await update.effective_message.reply_text(denial_text(update))
        return

    message = update.effective_message
    chat_id = update.effective_chat.id
    botlog.log_user_msg(
        _user_name(update), update.effective_user.id, _chat_desc(update), message.text
    )
    t_start = time.perf_counter()

    history = chat_histories.setdefault(chat_id, [])
    history.append(HumanMessage(content=message.text))
    trim_history(history)

    typing_task = asyncio.create_task(typing_indicator(context.bot, chat_id))
    try:
        now = datetime.now().astimezone()
        preamble = (
            f"Current date and time: {now:%A}, {now:%d %B %Y}, {now:%I:%M %p} "
            f"({now:%Z}, UTC{now:%z}). Trust this above any other date information."
        )
        agent_messages = list(history)
        if memory_store and memory_store.enabled:
            t0 = time.perf_counter()
            recall_texts = await memory_store.search(str(chat_id), message.text)
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
        await send_html_reply(update, body, footer_sources)
        botlog.log_reply(
            time.perf_counter() - t_start, len(footer_sources), "ok" if not failed else "gated-fallback"
        )

        if memory_store and memory_store.enabled:
            exchange = f"User: {message.text}\nAssistant: {stored_reply}"
            task = asyncio.create_task(memory_store.add(str(chat_id), exchange))
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
    finally:
        typing_task.cancel()
        save_histories()


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Exception while handling an update:", exc_info=context.error)
    botlog.log_error_note(f"{type(context.error).__name__}: {str(context.error)[:200]}")


def main():
    global allowed_ids, owner_id
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit(
            'Missing TELEGRAM_BOT_TOKEN. Put it in a .env file next to main.py:\nTELEGRAM_BOT_TOKEN="123456:ABC..."'
        )

    raw_ids = os.environ.get("ALLOWED_TELEGRAM_IDS", "").strip()
    if raw_ids:
        try:
            parsed_ids = [int(x.strip()) for x in raw_ids.split(",") if x.strip()]
        except ValueError:
            raise SystemExit("ALLOWED_TELEGRAM_IDS must be comma-separated integers")
        allowed_ids = set(parsed_ids)
        owner_id = parsed_ids[0]
        logger.info("Allowlist active: %s (owner: %s)", sorted(allowed_ids), owner_id)
    else:
        logger.warning("ALLOWED_TELEGRAM_IDS is empty - the bot is open to ANYONE")

    load_histories()

    global memory_store
    try:
        memory_store = MemoryStore(Path(__file__).parent / "memory.db")
    except Exception as e:
        logger.warning("Memory store init failed (%s); continuing without it", e)

    users_desc = "open to everyone" if allowed_ids is None else f"{len(allowed_ids)} allowlisted"
    botlog.log_startup(
        MODEL_NAME,
        users_desc,
        bool(memory_store and memory_store.enabled),
        len(TOOLBELT),
    )

    app = (
        Application.builder()
        .token(token)
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(60)
        .pool_timeout(30)
        .get_updates_connect_timeout(30)
        .get_updates_read_timeout(30)
        .get_updates_write_timeout(30)
        .get_updates_pool_timeout(30)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("reset", reset_command))
    app.add_handler(CommandHandler("forget", forget_command))
    app.add_handler(CommandHandler("logs", logs_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    logger.info("Bot is running locally... Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
