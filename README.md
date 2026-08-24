# 🤖 Jarvis — Local-First AI Assistant for Telegram

A privacy-first Telegram bot that runs **entirely on your own machine**: a local LLM served by [Ollama](https://ollama.com), web search when needed, exact live-fact tools, long-term semantic memory, and Telegram-native markdown rendering.

No cloud AI APIs. No per-token costs. Your conversations never leave your hardware.

## ✨ Features

- 🔍 **Web search fallback** — DuckDuckGo search for open-ended questions, answers cited with clickable sources
- 📄 **Document Q&A (RAG)** — upload PDF/DOCX/TXT/MD, ask questions with per-user private indexing
- 🧰 **Exact live-fact tools** — current time anywhere on Earth, calendar math, live weather (wttr.in), exchange rates (ECB), crypto prices (CoinGecko) — all keyless public APIs
- 🧠 **Long-term semantic memory** — past exchanges are embedded and recalled when relevant, isolated per chat (sqlite-vec)
- 📝 **Telegram-native formatting** — the model's GitHub-flavored Markdown is converted to proper Telegram HTML: headings, lists, code blocks, tables, clickable source links
- 🛡️ **Anti-hallucination guardrails** — per-turn date grounding, output sanity gate, tool-round caps with forced grounding, "say you don't know" instructions
- 👥 **Allowlist access control** — only people you approve can talk to it; first allowlisted ID is the owner
- 📊 **Human-readable logs** — emoji-rich daily log files, a colored terminal tail, and a `/logs` command for the owner

## 🏗️ How it works

```
Telegram ⇄ python-telegram-bot
                │
                ▼
      LangGraph ReAct agent  ◄── single system prompt (persona + live clock)
        │             │
        │             ├── web_search ────────► DuckDuckGo
        │             ├── get_current_time ──► stdlib zoneinfo (offline)
        │             ├── date_calculator ───► stdlib datetime (offline)
        │             ├── get_weather ────────► wttr.in
        │             ├── get_exchange_rate ─► frankfurter.dev (ECB)
        │             └── get_crypto_price ──► CoinGecko
        ▼
   Ollama LLM (local)

   sqlite-vec memory ◄── embedded exchanges, per-chat isolation
```

Every turn is grounded with the real current date/time, the agent chooses between dedicated tools and web search, answers pass a sanity gate, get converted to Telegram HTML, and flow into per-chat vector memory.

## 📋 Requirements

| Requirement | Notes |
|---|---|
| Python **3.14+** | easiest via [uv](https://docs.astral.sh/uv/) |
| [uv](https://docs.astral.sh/uv/) | package manager |
| [Ollama](https://ollama.com/download) | runs the models locally |
| Telegram bot token | from [@BotFather](https://t.me/BotFather) |

## 🚀 Quick start

```bash
# 1. Clone and install dependencies
git clone https://github.com/Deeps-77/jarvis-telegram-bot.git
cd jarvis-telegram-bot
uv sync

# 2. Pull the models into Ollama
ollama pull hf.co/LiquidAI/LFM2.5-2.6B-GGUF:latest    # chat model (tool calling)
ollama pull nomic-embed-text                           # embeddings for memory

# 3. Configure
cp .env.example .env
# edit .env → set TELEGRAM_BOT_TOKEN and ALLOWED_TELEGRAM_IDS (your Telegram ID)

# 4. Run
python main.py        # or: uv run main.py
```

Don't know your Telegram ID? Message the bot once — unauthorized users get a reply **containing their ID**, or ask [@userinfobot](https://t.me/userinfobot).

## ⚙️ Configuration (`.env`)

| Variable | Default | Purpose |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | *(required)* | token from @BotFather |
| `ALLOWED_TELEGRAM_IDS` | *(empty = open to anyone)* | comma-separated allowlist; **first ID = owner** |
| `OLLAMA_MODEL` | LFM2.5-2.6B GGUF | any tool-calling chat model in `ollama list` |
| `EMBED_MODEL` | `nomic-embed-text` | embedding model behind long-term memory |
| `MEMORY_TOP_K` | `4` | max recalled old exchanges per query |
| `MEMORY_MIN_SIMILARITY` | `0.55` | cosine floor for recall (empirically tuned) |
| `MEMORY_MAX_PER_CHAT` | `500` | FIFO eviction cap per chat |

## 💬 Commands

| Command | Who | Effect |
|---|---|---|
| `/start` | allowed users | start a fresh conversation |
| `/reset` | allowed users | clear the current chat window |
| `/forget` | allowed users | clear the window **and** long-term memory |
| `/docs` | allowed users | list your uploaded documents |
| `/logs [n]` | owner only | last n activity lines sent as a message |

### 📄 Documents

Send any **PDF / DOCX / TXT / MD** file to the bot — it's parsed, chunked, embedded and indexed under your account only. Then just ask questions ("what does my lease say about deposits?") or request summaries. Documents are private per user; `search_documents`, `list_documents` and `summarize_document` tools ground the answers in your files with `filename · chunk` references.

## 🖥️ Chainlit web UI (optional)

A local ChatGPT-style web interface sharing the same brain, memory and documents as the bot:

```bash
# in .env → set CHAINLIT_PASSWORD (web is fail-closed without it)
uv run chainlit run app.py --host 0.0.0.0 --port 8000 --headless
```

Then open **http://localhost:8000** on this machine, or `http://<your-LAN-IP>:8000` from other devices (`ipconfig` → your Wi-Fi IPv4). Never browse to `0.0.0.0` itself — Windows rejects it; it only selects which interfaces listen. Log in with `CHAINLIT_USERNAME` / `CHAINLIT_PASSWORD`. Attach PDF/DOCX/TXT/MD files directly in the chat to index them. Full markdown rendering — tables and headings look better here than in Telegram.

## 📡 Watching logs live

```bash
python watch_logs.py            # color-coded tail of activity + diagnostics
python watch_logs.py --lines 50
```

Log files land in `logs/`, rotate at midnight, and are kept for 30 days. Written in real time — follow them from a second terminal while the bot runs.

## 🔄 Swapping models

```bash
# PowerShell
$env:OLLAMA_MODEL="phi4-mini:latest"; python main.py

# bash
OLLAMA_MODEL=phi4-mini:latest python main.py
```

Or set it permanently in `.env`. Any Ollama model with tool-calling support works; smaller models (≤3B) vary in how reliably they follow instructions.

## 🔒 Security & privacy notes

- Message content stays on your machine except when the model explicitly queries public APIs (search/weather/FX/crypto).
- `.env` and all runtime data (`logs/`, `memory.db`, `chat_history.json`) are gitignored and never committed.
- If a token ever leaks, revoke it immediately via @BotFather.
- `logs/activity.log` contains message previews — it stays local; treat it as sensitive.

## 📁 Project layout

```
main.py          Telegram adapter: handlers, auth, markdown→HTML pipeline
app.py           Chainlit web UI adapter (password auth, file uploads)
core.py          backend-agnostic agent brain (shared by all frontends)
tools.py         nine agent tools (search + live facts + document RAG)
docs.py          per-user document store (parse, chunk, embed, retrieve)
memory.py        sqlite-vec long-term conversational memory
botlog.py        logging setup + human-readable event helpers
watch_logs.py    colored terminal log follower
```

## 🧪 Known limits

- Small (≤3B) models vary in instruction-following: persona discipline, tool choice, and hallucination resistance improve with larger models.
- Weather/FX/crypto rely on free keyless services — occasional throttling degrades gracefully instead of inventing numbers.

## 📄 License

[MIT](LICENSE) © 2026 Deepak M
