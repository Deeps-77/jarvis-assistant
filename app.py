import json
import logging
import os
import secrets
import sys
import tomllib
from pathlib import Path

os.environ["LITERAL_TELEMETRY_OFF"] = "true"
sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)


def _ensure_chainlit_config(base: Path):
    cfg = base / ".chainlit" / "config.toml"
    cfg.parent.mkdir(exist_ok=True)
    data = {}
    if cfg.exists():
        try:
            with open(cfg, "rb") as f:
                data = tomllib.load(f)
        except Exception:
            data = {}

    port = os.environ.get("CHAINLIT_PORT", "8000")

    def set_path(d, path, value):
        cur = d
        for key in path[:-1]:
            cur = cur.setdefault(key, {})
        cur[path[-1]] = value

    set_path(data, ["features", "audio", "enabled"], True)
    set_path(data, ["STT", "enabled"], True)
    set_path(data, ["STT", "engine"], "openai")
    set_path(data, ["STT", "model"], "local-whisper")
    set_path(data, ["OpenAI", "api_key"], "local-stt")
    set_path(data, ["OpenAI", "base_url"], f"http://127.0.0.1:{port}/v1")

    lines = []

    def emit(table: dict, prefix: str):
        for k, v in table.items():
            if not isinstance(v, dict):
                if isinstance(v, bool):
                    lines.append(f"{k} = {'true' if v else 'false'}")
                elif isinstance(v, (int, float)):
                    lines.append(f"{k} = {v}")
                else:
                    lines.append(f"{k} = {json.dumps(str(v))}")
        for k, v in table.items():
            if isinstance(v, dict):
                full = f"{prefix}{k}."
                lines.append("")
                lines.append(f"[{full.rstrip('.')}]")
                emit(v, full)

    emit(data, "")
    cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")


_ensure_chainlit_config(Path(__file__).parent / ".chainlit")

_secret_file = Path(__file__).parent / ".chainlit.secret"
if not _secret_file.exists():
    _secret_file.write_text(secrets.token_hex(32), encoding="utf-8")
os.environ.setdefault(
    "CHAINLIT_AUTH_SECRET", _secret_file.read_text(encoding="utf-8").strip()
)

import chainlit as cl

import core

AUDIO_UPLOADS = {".oga", ".ogg", ".wav", ".mp3", ".m4a", ".flac", ".webm"}

try:
    from chainlit.server import app as _server_app
    from fastapi import File, UploadFile

    @_server_app.post("/v1/audio/transcriptions")
    async def _voice_stt(file: UploadFile = File(...)):
        data = await file.read()
        text = await core.transcribe_audio(data, file.filename or "audio.ogg")
        return {"text": text}

except Exception as e:
    logger.debug("STT route not mounted: %s", e)

setup_done = False


def ensure_setup():
    global setup_done
    if not setup_done:
        core.load_histories()
        base = Path(__file__).parent
        core.init_memory(base / "memory.db")
        core.init_docs(base / "memory.db")
        core.init_speech()
        (base / "documents").mkdir(exist_ok=True)
        setup_done = True


def _password() -> str:
    return os.environ.get("CHAINLIT_PASSWORD", "")


def _username() -> str:
    return os.environ.get("CHAINLIT_USERNAME", "webuser")


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    expected = _password()
    if not expected:
        print("CHAINLIT_PASSWORD is not set - web access denied (fail-closed).")
        return None
    if username == _username() and password == expected:
        return cl.User(identifier=username)
    return None


SUPPORTED_UPLOADS = {".pdf", ".docx", ".txt", ".md"}


def session_key() -> str:
    return f"web:{_username()}"


async def handle_attachments(elements) -> tuple[list[str], list[str]]:
    confirmations: list[str] = []
    transcripts: list[str] = []
    for el in elements or []:
        path = getattr(el, "path", None)
        name = getattr(el, "name", None) or "upload.bin"
        suffix = Path(name).suffix.lower()

        if suffix in AUDIO_UPLOADS:
            if not path or not Path(path).exists():
                confirmations.append(f"⚠️ {name}: file missing on server")
                continue
            transcript = await core.transcribe_audio(Path(path).read_bytes(), name)
            if not transcript:
                confirmations.append(f"⚠️ {name}: couldn't hear any speech")
                continue
            confirmations.append(f"🎤 Transcribed {name}")
            transcripts.append(transcript)
            continue

        if suffix not in SUPPORTED_UPLOADS:
            confirmations.append(f"⚠️ {name}: unsupported format (use PDF/DOCX/TXT/MD)")
            continue
        if not path or not Path(path).exists():
            confirmations.append(f"⚠️ {name}: file missing on server")
            continue
        raw = Path(path).read_bytes()
        result = await core.ingest_document(_username(), name, raw)
        status = result.get("status")
        if status == "added":
            confirmations.append(f"✅ Indexed {name} ({result['chunks']} chunks)")
        elif status == "unchanged":
            confirmations.append(f"✅ {name} already indexed")
        else:
            confirmations.append(f"⚠️ {name}: {result.get('message', 'failed')}")
    return confirmations, transcripts


@cl.on_chat_start
async def on_chat_start():
    ensure_setup()
    await cl.Message(
        content=(
            f"Hello! I'm Jarvis, running locally on this machine.\n\n"
            f"- Ask me anything — I'll search the web when needed.\n"
            f"- Attach a **PDF / DOCX / TXT / MD** file and I'll index it for Q&A.\n"
            f"- Type `list my documents` to see what's indexed."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message):
    ensure_setup()
    owner = _username()

    notes: list[str] = []
    transcripts: list[str] = []
    if message.elements:
        notes, transcripts = await handle_attachments(message.elements)

    if notes:
        await cl.Message(content="\n".join(notes)).send()
    for t in transcripts:
        await cl.Message(content=f"🎤 I heard: {t[:400]}").send()

    text = message.content.strip() or " ".join(transcripts).strip()
    if not text:
        return

    answer_msg = cl.Message(content="")
    await answer_msg.send()

    body, sources, failed = await core.respond(session_key(), text, owner=owner)

    if sources:
        links = "\n".join(f"[{i}] {url}" for i, url in enumerate(sources, start=1))
        body = f"{body}\n\n**Sources:**\n{links}"

    answer_msg.content = body or "(no content)"
    await answer_msg.update()


if __name__ == "__main__":
    import uvicorn
    from chainlit.server import app as server_app

    host = os.environ.get("CHAINLIT_HOST", "0.0.0.0")
    port = int(os.environ.get("CHAINLIT_PORT", "8000"))
    uvicorn.run(server_app, host=host, port=port)
