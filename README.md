# 🤖 Jarvis — Local-First AI Assistant (Telegram + Web)

A privacy-first AI assistant that runs **entirely on your own machine** and is reachable from **both Telegram and a local Chainlit web UI**: a local LLM served by [Ollama](https://ollama.com), web search when needed, exact live-fact tools, long-term semantic memory, and native markdown rendering on each frontend.

No cloud AI APIs. No per-token costs. Your conversations never leave your hardware.

## ✨ Features

- 🔍 **Web search fallback** — DuckDuckGo search for open-ended questions, answers cited with clickable sources
- 📄 **Document Q&A (RAG)** — upload PDF/DOCX/TXT/MD, ask questions with per-user private indexing
- 🖼️ **Image understanding** — attach photos/screenshots/charts from Telegram or the web UI; a local vision model (granite3.2-vision) analyzes what's actually visible
- 🎤 **Voice input** — local faster-whisper transcription (Telegram voice notes, Chainlit mic + audio files)
- 🧰 **Exact live-fact tools** — current time anywhere on Earth, calendar math, live weather (wttr.in), exchange rates (ECB), crypto prices (CoinGecko) — all keyless public APIs
- 🧠 **Long-term fact memory** — an LLM extractor distills durable facts from each exchange (deduped, capped, embedded via sqlite-vec, recalled per chat); scanner-flagged facts are quarantined and never recalled
- 🛡️ **Prompt-injection guard** — web results and document chunks are scanned for instruction-like text, malicious spans are redacted, and all external text enters the model wrapped in untrusted-content boundaries
- ⚡ **Live token streaming** in the web UI — answers appear as they're generated; tool-chatter is retracted so only the final answer stays
- 📝 **Native formatting** — the model's GitHub-flavored Markdown is converted to proper Telegram HTML (headings, lists, code blocks, tables, clickable source links); the web UI renders full Markdown directly
- 🛡️ **Anti-hallucination guardrails** — per-turn date grounding, output sanity gate, tool-round caps with forced grounding, "say you don't know" instructions
- 👥 **Allowlist access control** — only people you approve can talk to it; first allowlisted ID is the owner
- 📊 **Human-readable logs** — emoji-rich daily log files, a colored terminal tail, and a `/logs` command for the owner

## 🏗️ How it works

```
            ┌─────────────────────────┐      ┌─────────────────────────┐
   You ───► │ Telegram (main.py)      │      │ Chainlit web UI         │◄─── You
            │ python-telegram-bot,    │      │ (app.py: password auth, │
            │ handlers + markdown→HTML│      │  file/audio upload, mic)│
            └────────────┬────────────┘      └────────────┬────────────┘
                         │                                │
                         └───────────────┬────────────────┘
                                         ▼
                          core.py — LangGraph ReAct agent  ◄── single system prompt (persona + live clock)
                            │             │
                            │             ├── web_search ────────► DuckDuckGo
                            │             ├── get_current_time ──► stdlib zoneinfo (offline)
                            │             ├── date_calculator ───► stdlib datetime (offline)
                            │             ├── get_weather ────────► wttr.in
                            │             ├── get_exchange_rate ─► frankfurter.dev (ECB)
                            │             └── get_crypto_price ──► CoinGecko
                            ▼
                       Ollama LLM (local)

            sqlite-vec memory ◄── extracted facts, per-chat isolation (shared by both frontends)
```

Both frontends talk to the same `core.py` brain, so they share memory, documents and tools. Every turn is grounded with the real current date/time, the agent chooses between dedicated tools and web search, answers pass a sanity gate, get rendered natively for each frontend (Telegram HTML or web Markdown), and flow into per-chat vector memory.

## 📋 Requirements

| Requirement | Notes |
|---|---|
| Python **3.14+** | easiest via [uv](https://docs.astral.sh/uv/) |
| [uv](https://docs.astral.sh/uv/) | package manager |
| [Ollama](https://ollama.com/download) | runs the models locally |
| Telegram bot token | from [@BotFather](https://t.me/BotFather) — **only needed for the Telegram frontend** |
| `CHAINLIT_PASSWORD` | **only needed for the web UI** — set in `.env`; web is fail-closed without it |

## 🚀 Quick start

```bash
# 1. Clone and install dependencies
git clone https://github.com/Deeps-77/jarvis-assistant.git
cd jarvis-assistant
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
| `MEMORY_TOP_K` | `4` | max recalled facts per query |
| `MEMORY_MIN_SIMILARITY` | `0.55` | cosine floor for recall (empirically tuned) |
| `MEMORY_MAX_PER_CHAT` | `500` | FIFO eviction cap per chat |
| `MEMORY_EXTRACT_MODEL` | *(empty = main model)* | model that distills durable facts from each exchange |
| `MEMORY_MAX_FACTS_PER_TURN` | `10` | cap on facts captured per exchange |
| `MEMORY_MAX_FACT_CHARS` | `200` | per-fact length cap |
| `INJECTION_GUARD` | `true` | scan/redact/tag external text (web + docs) and quarantine suspicious auto-memories |

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

## 🖥️ Chainlit web UI

A local ChatGPT-style web interface that shares the **same brain, memory and documents** as the Telegram bot — the two frontends are fully interchangeable:

```bash
# in .env → set CHAINLIT_PASSWORD (web is fail-closed without it)
python app.py            # serves http://localhost:8000
```

Open **http://localhost:8000** on this machine, or `http://<your-LAN-IP>:8000` from other devices (`ipconfig` → your Wi-Fi IPv4). Override host/port via `CHAINLIT_HOST` / `CHAINLIT_PORT`.

### 🎙️ Using the microphone from other devices (HTTPS)

Browsers only allow microphone capture on **secure contexts**: `https://` pages, or plain `http://localhost`. The mic works out of the box on this machine, but is blocked when the UI is opened from other devices over plain HTTP.

**Option 1 — built-in self-signed HTTPS**

```bash
# in .env → CHAINLIT_TLS=true
python app.py
```

A certificate covering localhost + this machine's IPs is auto-generated into `.chainlit/certs/`. Then open **`https://<your-LAN-IP>:8000`** — each device shows a one-time *"Your connection is not private"* warning: click **Advanced → Proceed**. After that, the mic works permanently.

**Option 2 — per-device browser flag (no HTTPS)**

On each device, open `chrome://flags/#unsafely-treat-insecure-origin-as-secure`, enter `http://<your-LAN-IP>:8000`, enable, and relaunch the browser. Mic then works over plain HTTP.

> ℹ️ Launched through uvicorn directly rather than `chainlit run`: the CLI's `nest_asyncio` patching breaks async detection on Python 3.14 (`AsyncLibraryNotFoundError`). Log in with `CHAINLIT_USERNAME` / `CHAINLIT_PASSWORD`. Attach PDF/DOCX/TXT/MD files directly in the chat to index them. Full markdown rendering — tables and headings look better here than in Telegram.

### 🎤 Voice notes

Telegram voice notes and Chainlit audio attachments (or the browser mic button) are transcribed locally by faster-whisper (`WHISPER_MODEL`, default `small`, int8 CPU with automatic CUDA when available — GPU detected on this machine). The transcript is echoed back ("🎤 I heard: …") and then answered through the full pipeline — search, live tools, documents, all of it.

## 📡 Watching logs live

```bash
python watch_logs.py            # color-coded tail of activity + diagnostics
python watch_logs.py --lines 50
```

Log files land in `logs/`, rotate at midnight, and are kept for 30 days. Written in real time — follow them from a second terminal while the bot runs.

| File | Content |
|---|---|
| `logs/activity.log` | human-readable diary: 👤 messages, 🎤 transcriptions, 💬 replies w/ timing, 🔧 tool usage, 📄 document indexing, 🛡️ denials, ⌨️ commands |
| `logs/jarvis.log` | full diagnostics with timestamps |
| `logs/events.jsonl` | machine-readable event stream (one JSON per event) |

**Usage analytics**: the owner can send `/stats` anytime for a same-day summary (messages per user, replies, voice notes, top tools, documents, gated failures). Stats are rebuilt from `events.jsonl`, so they survive restarts.

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
app.py           Chainlit web UI adapter (password auth, file/audio uploads, mic STT route)
core.py          backend-agnostic agent brain (shared by all frontends)
tools.py         nine agent tools (search + live facts + document RAG)
docs.py          per-user document store (parse, chunk, embed, retrieve)
memory.py        sqlite-vec long-term conversational memory
speech.py        whisper-based speech transcription
botlog.py        logging setup + human-readable event helpers
watch_logs.py    colored terminal log follower
```

## 🧪 Known limits

- Small (≤3B) models vary in instruction-following: persona discipline, tool choice, and hallucination resistance improve with larger models.
- Weather/FX/crypto rely on free keyless services — occasional throttling degrades gracefully instead of inventing numbers.

## 📄 License

[MIT](LICENSE) © 2026 Deepak M
