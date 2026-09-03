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
main.py                Telegram adapter: handlers, auth, markdown→HTML pipeline
app.py                 Chainlit web UI adapter (password auth, file/audio uploads, mic STT route)
code_ui.py             Chainlit code-assistant adapter (Phase 1+; workspace, mode toggle, streaming, approval gate)
core.py                backend-agnostic chat agent brain (shared by all chat frontends)
code_assistant/        code-assistant brain + tools (Phase 2: read + write + approval)
  workspace.py         Workspace model + path validation + saved-workspaces registry
  tools.py             LangChain @tool functions (read + write + exec; mode-gated)
  modes.py             Plan/Build mode enum + tool gating + system prompts
  brain.py             CodeBrain manual ReAct loop with approval gate, streams BrainEvents
  sandbox.py           subprocess sandbox: env strip, timeout, output cap, copy_out
tools.py               nine chat tools (search + live facts + document RAG)
llm_provider.py        Ollama + OpenAI provider abstraction (Phase 0)
token_usage.py         TokenTracker LangChain callback + JSONL persistence (Phase 0)
docs.py                per-user document store (parse, chunk, embed, retrieve)
memory.py              sqlite-vec long-term conversational memory
speech.py              whisper-based speech transcription
botlog.py              logging setup + human-readable event helpers
watch_logs.py          colored terminal log follower
```

## 💻 Code Assistant (Phase 2 — read + write with approval gate)

A second web UI dedicated to **workspace-oriented code work**. Runs alongside
the chat UI on a different port (default `:8500` vs. chat's `:8000`), so the
two never collide and can run at the same time.

```bash
# .env → set CHAINLIT_CODE_PASSWORD (the UI is fail-closed without it)
python code_ui.py           # serves http://localhost:8500
```

### What's in Phase 2

**Read-only tools (always available, no approval needed):**

- `list_files` — recursive listing, optional glob
- `read_file` — paginated read with line numbers, byte-capped
- `grep_files` — regex search (uses `rg` when available)
- `get_file_info` — size, mtime, language guess

**Write/exec tools (BUILD mode only, every call approval-gated):**

- `write_file` — atomic file write with diff preview
- `edit_file` — targeted find/replace with diff (refuses ambiguous matches)
- `apply_patch` — apply a unified diff
- `mkdir` — recursive directory creation
- `delete_path` — soft-delete to `.jarvis-sandbox/trash/`
- `run_command` — single shell command in `.jarvis-sandbox/tmp/` (env
  stripped, hard timeout, output capped)
- `copy_out` — move a sandbox-produced file into the workspace

### Safety rails (three layers)

1. **Tool gating** — `modes.filter_tools` ensures Plan mode never sees
   write tools, period. Build mode exposes them but with a separate
   `REQUIRES_APPROVAL` registry.
2. **Mode double-check** — every write tool calls `_require_build_mode()`
   on entry and returns an error string if the brain isn't in BUILD
   mode. The brain calls `set_current_mode` before each run so a stale
   UI mode never lets write tools slip through.
3. **Per-call approval gate** — before any approval-required tool runs,
   the brain yields an `approval_required` BrainEvent and pauses. The
   UI shows a confirmation card; the user replies with:
   - `approve` — run with the proposed args
   - `edit` — run with edited args (paste a JSON object)
   - `reject` — decline; the model sees a `USER_REJECTED` tool result
     and adapts
   - `reject <reason>` — decline with an explanation for the model

### Sandboxed command execution

`run_command` runs in `<workspace>/.jarvis-sandbox/tmp/` with the
environment stripped to a small allow-list (`PATH`, `LANG`, `HOME`,
`TZ`, …). Hard timeout (default 30 s) and a 200 KB per-stream output
cap are enforced. Files produced by the sandbox are NOT visible in the
workspace until `copy_out` is called explicitly. The sandbox is a
discipline tool, **not** a security boundary — local single-user only.

### Slash commands in the code UI

| Command | Effect |
|---|---|
| `/workspace <path>` | Switch to a different folder |
| `/mode plan` / `/mode build` | Toggle mode (Build = read-only until approval-gated writes flow) |
| `/usage` | Token usage summary + last 10 turns |
| `/reset` | Clear this chat's history |
| `/help` | Show the welcome card again |

### Workspace picker

- Type an absolute path (e.g. `D:\Projects\myrepo`) to open a project
  root. The UI remembers the last few workspaces in
  `code_workspaces.json`.
- Every call is validated: paths must be relative, traversal (`..`) is
  refused, and a deny-glob list hides `.git/`, `.venv/`,
  `node_modules/`, `.jarvis-sandbox/`, build artefacts, etc.

### Live token observation

Every LLM call goes through `TokenTracker`; the sidebar shows
per-session input/output totals and an estimated USD cost (OpenAI only;
Ollama is local → $0). `/usage` prints the last 10 turns.

### Streaming events

The agent streams `BrainEvent`s (`token` / `tool_start` / `tool_end` /
`approval_required` / `usage` / `done` / `error`) instead of returning
one big string, so the UI can show tool chips, retract intermediate
chatter, and update the usage card live.

### Cross-provider

Same `LLMConfig` as the chat brain. Ollama by default, OpenAI when
`CODE_LLM_PROVIDER=openai` and `CODE_LLM_API_KEY=...`.

### Architecture

```
            ┌─────────────────────────┐      ┌─────────────────────────┐
   You ───► │ Chainlit chat (app.py)  │      │ Chainlit code (code_ui) │◄─── You
            │ password auth, files,   │      │ password auth, workspace│
            │ voice, history sidebar  │      │ picker, mode toggle,    │
            └────────────┬────────────┘      │ approval cards          │
                         │                   └────────────┬────────────┘
                         ▼                                │
                  core.py + tools.py              code_assistant/brain.py
                  (chat brain, ReAct)             (CodeBrain, manual ReAct loop
                         │                          with approval gate)
                         │                                │
                         └────────────┬───────────────────┘
                                      ▼
                          llm_provider.py  ◄── Ollama OR OpenAI
                                      │
                          token_usage.TokenTracker  ◄── callbacks into both brains
                                      │
                          logs/code_tokens.jsonl  (per-turn usage log)
```

The code assistant brain is fully **additive** — `core.py` / `tools.py` /
`app.py` / `main.py` are untouched. Phases 0/1/2 (`llm_provider.py`,
`token_usage.py`, `code_assistant/`, `code_ui.py`) are all new modules.

### Recommended models

| Provider | Recommended | Notes |
|---|---|---|
| Ollama | `qwen2.5-coder:7b-instruct` (or larger) | Strong tool-calling discipline. The 3B variant in `.env.example` works but sometimes drops tool calls on long refactors. |
| OpenAI | `gpt-4o-mini` | Cheap, reliable tool-calling. Set `CODE_LLM_PROVIDER=openai` + `CODE_LLM_API_KEY`. |

### Known limits (Phase 2)

- The brain streams tool-call chatter as raw JSON with small local models
  (the model's structured `tool_calls` channel is unreliable below ~7B).
  The UI auto-detects and retracts that text so the user only sees the
  structured tool chip.
- The sandbox kills the subprocess on timeout, but on Windows the child
  process tree can linger in the background briefly. Use `delete_path`
  to clean up sandbox artefacts if needed.
- The code UI does **not** run alongside Telegram's `/mode` command — the
  mode toggle is web-UI-only, per the original design.

## 💻 Code Assistant (Phase 3 — observability + customisable harness)

Phase 3 adds **token observability**, a **YAML harness config**, and a
**subclassing API** so power users can tailor the agent without forking.

### Token observability

- **Daily roll-up** — `TokenTracker.daily_summary(days=N)` reads
  `logs/code_tokens.jsonl` and aggregates usage per calendar day,
  broken down by model. Use `/usage` in the UI to see a markdown card
  with session totals, a 7-day roll-up table, and the last 100 turns.
- **`render_usage_card(limit=100)`** — one-call method that produces
  a ready-to-render markdown card (session totals, daily roll-up,
  per-turn table). Used by the `/usage` slash command.
- **Per-model cost tracking** — OpenAI models get USD estimates from a
  static price table; Ollama is always free. Costs are persisted in
  `logs/code_tokens.jsonl` so roll-ups survive restarts.

### YAML harness config (`code_assistant.yaml`)

Create `code_assistant.yaml` next to `code_ui.py` to override defaults
without touching Python. Env vars (`CODE_LLM_*`, `CODE_SANDBOX_TIMEOUT`,
etc.) **always win** over YAML.

```yaml
code_assistant:
  llm:
    provider: ollama           # or "openai"
    model: qwen2.5-coder:3b
    base_url: http://localhost:11434
    temperature: 0.1
    num_ctx: 16384
    timeout: 600
    keep_alive: -1
    max_tokens: 0              # 0 = provider default
    api_key: ""                # openai only; env CODE_LLM_API_KEY wins

  sandbox:
    timeout_seconds: 30
    max_output_bytes: 200000
    env_keep: [PATH, LANG, HOME]   # additive with DEFAULT_ENV_KEEP

  modes:
    plan:
      tools: [list_files, read_file, grep_files, get_file_info]   # or "all"
      extra_system_prompt: ""   # appended to the built-in prompt
    build:
      tools: all
      extra_system_prompt: ""

  harness:
    max_tool_rounds: 6
    max_history_messages: 24
    require_approval_for: [write_file, edit_file, ...]   # additive to REQUIRES_APPROVAL
```

**Precedence:** `os.environ` → `code_assistant.yaml` → hard-coded defaults.

### Harness subclassing API

Subclass `CodeBrain` or configure at runtime:

```python
brain = CodeBrain(workspace=ws, llm_config=cfg, tracker=tracker, mode=Mode.BUILD)

# Add a custom tool (available in Build mode by default)
brain.register_tool(my_custom_tool, modes=[Mode.BUILD])

# Append custom prompt suffix
brain.override_prompt(Mode.PLAN, "\nAlways reply in Pirate English.")

# Dynamically add a tool to the approval gate
brain.register_approval_required("custom_risky_tool")

# Tune limits
brain.max_tool_rounds = 10
brain.max_history_messages = 50

# Apply YAML config at runtime
from code_assistant.config import load_config
brain._apply_config_overrides(load_config().raw)
```

**Harness API surface:**

| Method | Purpose |
|---|---|
| `register_tool(tool, modes)` | Add a custom `@tool` to `modes` (default `BUILD`) |
| `override_prompt(mode, suffix)` | Append `suffix` to the system prompt for `mode` |
| `register_approval_required(name)` | Add `name` to the approval gate |
| `unregister_approval_required(name)` | Remove from approval gate |
| `_apply_config_overrides(dict)` | Bulk-apply limits, prompts, approval list |

### Model capability flags

`LLMConfig` now carries auto-detected flags so the harness can branch:

| Flag | Meaning | Auto-detected for |
|---|---|---|
| `supports_vision` | Model can process images | `llava`, `gpt-4o*`, `granite3.2-vision` |
| `supports_structured_output` | Native structured output (JSON schema) | `gpt-4o*`, `o1/o3/o4` series |
| `supports_function_calling` | Native function/tool calling | All modern models |

Auto-detected at provider construction; override in `code_assistant.yaml`
via `llm:` block if needed.

### Updated limits (Phase 3)

- Token dashboard now shows daily roll-up and per-model breakdown.
- Config file is optional; env vars always take precedence.
- Harness API is fully typed and documented in `code_assistant/brain.py`.

### Known limits (Phase 3)

- Token dashboard now shows daily roll-up and per-model breakdown.
- Config file is optional; env vars always take precedence.
- Harness API is fully typed and documented in `code_assistant/brain.py`.

### Known limits (Phase 2)

- The brain streams tool-call chatter as raw JSON with small local models
  (the model's structured `tool_calls` channel is unreliable below ~7B).
  The UI auto-detects and retracts that text so the user only sees the
  structured tool chip.
- The sandbox kills the subprocess on timeout, but on Windows the child
  process tree can linger in the background briefly. Use `delete_path`
  to clean up sandbox artefacts if needed.
- The code UI does **not** run alongside Telegram's `/mode` command — the
  mode toggle is web-UI-only, per the original design.

## 🧪 Known limits

- Small (≤3B) models vary in instruction-following: persona discipline, tool choice, and hallucination resistance improve with larger models.
- Weather/FX/crypto rely on free keyless services — occasional throttling degrades gracefully instead of inventing numbers.

## 📄 License

[MIT](LICENSE) © 2026 Deepak M
