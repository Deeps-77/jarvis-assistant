import asyncio
import html
import logging
import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import botlog
import core
from core import MODEL_NAME, TOOLBELT, setup_logging
from langchain_core.messages import AIMessage, HumanMessage
from markdown_it import MarkdownIt
from telegram import Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest, TimedOut
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

MAX_MESSAGE_LEN = 4096
TG_CHUNK_LIMIT = 4000
DOCUMENTS_DIR = Path(__file__).parent / "documents"
SUPPORTED_UPLOADS = {".pdf", ".docx", ".txt", ".md"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

_md_parser = MarkdownIt("commonmark").enable(["table", "strikethrough"])

setup_logging()
logger = logging.getLogger(__name__)

allowed_ids: set[int] | None = None
owner_id: int | None = None


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


def _user_name(update: Update) -> str:
    user = update.effective_user
    return getattr(user, "full_name", None) or f"id:{getattr(user, 'id', '?')}"


def _chat_desc(update: Update) -> str:
    chat = update.effective_chat
    return getattr(chat, "title", None) or f"{chat.type} chat"


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


def split_message(text: str) -> list[str]:
    text = text.strip()
    if len(text) <= MAX_MESSAGE_LEN:
        return [text]
    return [text[i : i + MAX_MESSAGE_LEN] for i in range(0, len(text), MAX_MESSAGE_LEN)]


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
    core.clear_session(str(update.effective_chat.id))
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
    core.clear_session(str(update.effective_chat.id))
    await update.effective_message.reply_text("Conversation cleared.")


async def forget_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        botlog.log_denied(_user_name(update), update.effective_user.id)
        await update.effective_message.reply_text(denial_text(update))
        return
    botlog.log_command("forget", _user_name(update), update.effective_user.id)
    chat_id = update.effective_chat.id
    core.clear_session(str(chat_id))
    if core.memory_store:
        await core.memory_store.clear_chat(str(chat_id))
    await update.effective_message.reply_text("Conversation and long-term memory cleared.")


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

    typing_task = asyncio.create_task(typing_indicator(context.bot, chat_id))
    try:
        body, footer_sources, failed = await core.respond(
            str(chat_id), message.text, owner=str(update.effective_user.id)
        )
        await send_html_reply(update, body, footer_sources)
        botlog.log_reply(
            time.perf_counter() - t_start,
            len(footer_sources),
            "ok" if not failed else "gated-fallback",
        )
    finally:
        typing_task.cancel()
        await asyncio.to_thread(core.save_histories)


async def docs_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        botlog.log_denied(_user_name(update), update.effective_user.id)
        await update.effective_message.reply_text(denial_text(update))
        return
    botlog.log_command("docs", _user_name(update), update.effective_user.id)
    if not core.doc_store or not core.doc_store.enabled:
        await update.effective_message.reply_text("⚠️ Document storage is unavailable.")
        return
    docs = await core.doc_store.list_docs(str(update.effective_user.id))
    if not docs:
        await update.effective_message.reply_text(
            "No documents yet — send me a PDF, DOCX, TXT or MD file and I'll index it."
        )
        return
    listing = "\n".join(f"📄 {d['source']} — {d['chunks']} chunks ({d['date']})" for d in docs)
    await update.effective_message.reply_text(f"Your documents:\n{listing}")


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if owner_id is None or user_id != owner_id:
        botlog.log_denied(_user_name(update), user_id)
        await update.effective_message.reply_text("⛔ /stats is restricted to the bot owner.")
        return
    botlog.log_command("stats", _user_name(update), user_id)
    await update.effective_message.reply_text(botlog.get_stats())


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        botlog.log_denied(_user_name(update), update.effective_user.id)
        await update.effective_message.reply_text(denial_text(update))
        return
    if not core.doc_store or not core.doc_store.enabled:
        await update.effective_message.reply_text("⚠️ Document storage is unavailable right now.")
        return

    doc = update.effective_message.document
    fname = os.path.basename(doc.file_name or "upload.bin")
    suffix = Path(fname).suffix.lower()
    user_id = update.effective_user.id
    botlog.log_user_msg(
        _user_name(update), user_id, _chat_desc(update), f"📎 uploaded {fname}"
    )

    if suffix not in SUPPORTED_UPLOADS:
        await update.effective_message.reply_text(
            f"Unsupported format '{suffix}'. I can index PDF, DOCX, TXT and MD files."
        )
        return
    if (doc.file_size or 0) > MAX_UPLOAD_BYTES:
        await update.effective_message.reply_text("File too large (limit 20 MB).")
        return

    typing_task = asyncio.create_task(typing_indicator(context.bot, update.effective_chat.id))
    try:
        tg_file = await doc.get_file()
        raw = bytes(await tg_file.download_as_bytearray())

        DOCUMENTS_DIR.mkdir(exist_ok=True)
        safe_name = f"{user_id}_{fname}"
        (DOCUMENTS_DIR / safe_name).write_bytes(raw)

        result = await core.ingest_document(str(user_id), fname, raw)
        status = result.get("status")
        if status == "added":
            await update.effective_message.reply_text(
                f"✅ Indexed {fname} ({result['chunks']} chunks). Ask me anything about it!"
            )
        elif status == "unchanged":
            await update.effective_message.reply_text(
                f"✅ {fname} is already indexed — nothing new."
            )
        else:
            await update.effective_message.reply_text(f"⚠️ {result.get('message', 'Indexing failed.')}")
    except TimedOut:
        await update.effective_message.reply_text("Download timed out — please resend the file.")
    except Exception as e:
        logger.exception("Document handling failed")
        await update.effective_message.reply_text(f"⚠️ Could not process that file: {e}")
    finally:
        typing_task.cancel()


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        botlog.log_denied(_user_name(update), update.effective_user.id)
        await update.effective_message.reply_text(denial_text(update))
        return

    msg = update.effective_message
    user_id = update.effective_user.id
    caption = (msg.caption or "").strip()
    botlog.log_user_msg(
        _user_name(update), user_id, _chat_desc(update), f"🖼️ photo {caption}".strip() or "🖼️ photo"
    )

    typing_task = asyncio.create_task(typing_indicator(context.bot, chat_id := update.effective_chat.id))
    t_start = time.perf_counter()
    try:
        biggest = msg.photo[-1]
        tg_file = await biggest.get_file()
        raw = bytes(await tg_file.download_as_bytearray())
        question = caption or "Describe this image in detail."
        body, failed = await core.vision_respond(
            str(chat_id), str(user_id), raw, "jpeg", question
        )
        await send_html_reply(update, body)
        botlog.log_reply(
            time.perf_counter() - t_start,
            0,
            "ok" if not failed else "vision-error",
        )
    finally:
        typing_task.cancel()


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_authorized(update):
        botlog.log_denied(_user_name(update), update.effective_user.id)
        await update.effective_message.reply_text(denial_text(update))
        return
    if not core.speech_transcriber or not core.speech_transcriber.enabled:
        await update.effective_message.reply_text("⚠️ Speech transcription is unavailable right now.")
        return

    msg = update.effective_message
    audio = msg.voice or msg.audio
    fname = getattr(audio, "file_name", None) or f"voice_{msg.message_id}.oga"
    user_id = update.effective_user.id
    botlog.log_user_msg(_user_name(update), user_id, _chat_desc(update), "🎤 voice message")
    t_start = time.perf_counter()

    typing_task = asyncio.create_task(typing_indicator(context.bot, chat_id := update.effective_chat.id))
    try:
        tg_file = await audio.get_file()
        raw = bytes(await tg_file.download_as_bytearray())

        transcript = await core.transcribe_audio(raw, fname)
        if not transcript:
            await update.effective_message.reply_text(
                "🎤 I couldn't make out any speech in that message. Try again a bit closer to the mic?"
            )
            return

        echo = f"🎤 I heard: {_esc(transcript[:400])}"
        try:
            await update.effective_message.reply_text(echo, parse_mode=ParseMode.HTML)
        except BadRequest:
            await update.effective_message.reply_text(f"🎤 I heard: {transcript[:400]}")

        body, footer_sources, failed = await core.respond(str(chat_id), transcript, owner=str(user_id))
        await send_html_reply(update, body, footer_sources)
        botlog.log_reply(
            time.perf_counter() - t_start,
            len(footer_sources),
            "ok" if not failed else "gated-fallback",
        )
    finally:
        typing_task.cancel()
        await asyncio.to_thread(core.save_histories)


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

    core.load_histories()
    core.init_memory(Path(__file__).parent / "memory.db")
    core.init_docs(Path(__file__).parent / "memory.db")
    core.init_speech()
    DOCUMENTS_DIR.mkdir(exist_ok=True)

    users_desc = "open to everyone" if allowed_ids is None else f"{len(allowed_ids)} allowlisted"
    botlog.log_startup(
        MODEL_NAME,
        users_desc,
        bool(core.memory_store and core.memory_store.enabled),
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
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("docs", docs_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, handle_voice))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_error_handler(error_handler)

    logger.info("Bot is running locally... Press Ctrl+C to stop.")
    app.run_polling()


if __name__ == "__main__":
    main()
