import json
import logging
import os
import re
import secrets
import sys
import time
from pathlib import Path

os.environ["LITERAL_TELEMETRY_OFF"] = "true"
sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)


def _ensure_chainlit_config(base: Path):
    cfg = base / ".chainlit" / "config.toml"
    cfg.parent.mkdir(exist_ok=True)
    if not cfg.exists():
        cfg.write_text(
            "[project]\n\n[features.audio]\nenabled = true\n"
            "\n[features.spontaneous_file_upload]\nenabled = true\naccept = [\"*/*\"]\n",
            encoding="utf-8",
        )
        return
    text = cfg.read_text(encoding="utf-8")
    if "[features.audio]" in text:
        pat = re.compile(r"(\[features\.audio\][^\[]*?)enabled\s*=\s*\S+", re.DOTALL)
        text = pat.sub(r"\g<1>enabled = true", text, count=1)
    else:
        text += "\n[features.audio]\nenabled = true\n"
    cfg.write_text(text, encoding="utf-8")


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
SAMPLE_RATE = 24000

import io
import wave


def _pcm_to_wav_bytes(pcm: bytes, rate: int = SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    return buffer.getvalue()


_audio_buffers: dict[str, list[bytes]] = {}


@cl.on_audio_start
async def on_audio_start():
    _audio_buffers[session_key()] = []
    return True


@cl.on_audio_chunk
async def on_audio_chunk(chunk) -> None:
    data = getattr(chunk, "data", None)
    if data:
        _audio_buffers.setdefault(session_key(), []).append(data)


@cl.on_audio_end
async def on_audio_end():
    ensure_setup()
    chunks = _audio_buffers.pop(session_key(), [])
    pcm = b"".join(chunks)
    if not pcm:
        await cl.Message(content="🎤 Empty recording.").send()
        return
    transcript = await core.transcribe_audio(_pcm_to_wav_bytes(pcm), "mic.wav")
    if not transcript:
        await cl.Message(content="🎤 I couldn't hear anything. Try again closer to the mic.").send()
        return
    await cl.Message(content=f"🎤 I heard: {transcript[:400]}").send()
    await _respond_and_send(transcript)

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
    await _respond_and_send(text)


async def _respond_and_send(text: str):
    answer_msg = cl.Message(content="")
    await answer_msg.send()
    body, sources, failed = await core.respond(session_key(), text, owner=_username())
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
