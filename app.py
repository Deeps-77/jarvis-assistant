import json
import logging
import os
import re
import secrets
import socket
import sys
import time
from pathlib import Path

os.environ["LITERAL_TELEMETRY_OFF"] = "true"
sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)


def _ensure_translations(chainlit_dir: Path):
    tdir = chainlit_dir / "translations"
    source = tdir / "en-US.json"
    if not source.exists():
        return
    for variant in ("en-IN", "en-GB"):
        target = tdir / f"{variant}.json"
        if not target.exists():
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def _local_ipv4s() -> list[str]:
    ips = set()
    try:
        host = socket.gethostname()
        for info in socket.getaddrinfo(host, None, family=socket.AF_INET):
            ips.add(info[4][0])
    except OSError:
        pass
    return sorted(ips)


def _ensure_tls(base: Path) -> tuple[Path, Path] | None:
    """Generate a self-signed certificate covering localhost + this machine's IPs."""
    if os.environ.get("CHAINLIT_TLS", "").strip().lower() not in ("true", "1", "yes"):
        return None
    certs = base / "certs"
    certs.mkdir(exist_ok=True)
    cert_file = certs / "jarvis.crt"
    key_file = certs / "jarvis.key"
    if cert_file.exists() and key_file.exists():
        return cert_file, key_file

    import datetime
    import ipaddress

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Jarvis Local")])

    san = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
    ]
    try:
        san.append(x509.DNSName(socket.gethostname()))
    except (OSError, UnicodeError):
        pass
    for ip in _local_ipv4s():
        try:
            san.append(x509.IPAddress(ipaddress.ip_address(ip)))
        except ValueError:
            pass

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.SubjectAlternativeName(san), critical=False)
        .sign(key, hashes.SHA256())
    )
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )
    logger.info(
        "Self-signed TLS certificate generated (%s) covering: %s",
        cert_file,
        ", ".join(str(n.value) for n in san),
    )
    return cert_file, key_file


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
    _ensure_translations(base)


_ensure_chainlit_config(Path(__file__).parent / ".chainlit")

_secret_file = Path(__file__).parent / ".chainlit.secret"
if not _secret_file.exists():
    _secret_file.write_text(secrets.token_hex(32), encoding="utf-8")
os.environ.setdefault(
    "CHAINLIT_AUTH_SECRET", _secret_file.read_text(encoding="utf-8").strip()
)

import chainlit as cl
from chainlit.types import ThreadDict
from langchain_core.messages import AIMessage, HumanMessage

import botlog
import core
from datalayer import SQLiteDataLayer
from paths import chat_threads_db, documents_dir, memory_db


@cl.data_layer
def get_data_layer():
    return SQLiteDataLayer(chat_threads_db())

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
        core.init_memory(memory_db())
        core.init_docs(memory_db())
        core.init_speech()
        documents_dir().mkdir(exist_ok=True)
        setup_done = True


def current_session_key() -> str:
    key = cl.user_session.get("session_key")
    if not key:
        import uuid

        key = f"web:{uuid.uuid4().hex[:12]}"
        cl.user_session.set("session_key", key)
    return key


def _rebuild_history_from_steps(steps) -> list:
    msgs: list = []
    for s in steps or []:
        stype = s.get("type")
        content = (s.get("output") or s.get("input") or "").strip()
        if not content:
            continue
        if stype == "user_message":
            msgs.append(HumanMessage(content=content))
        elif stype == "assistant_message":
            msgs.append(AIMessage(content=content))
    return msgs[-core.MAX_HISTORY_MESSAGES:]


def _password() -> str:
    return os.environ.get("CHAINLIT_PASSWORD", "")


def _username() -> str:
    return os.environ.get("CHAINLIT_USERNAME", "webuser")


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    expected = _password()
    if not expected:
        logger.warning("CHAINLIT_PASSWORD is not set - web access denied (fail-closed).")
        return None
    if username == _username() and password == expected:
        botlog.log_web_login(username)
        return cl.User(identifier=username)
    botlog.log_denied(username, "web")
    return None


SUPPORTED_UPLOADS = {".pdf", ".docx", ".txt", ".md"}


def session_key() -> str:
    key = cl.user_session.get("session_key")
    if not key:
        key = current_session_key()
    return key


IMAGE_UPLOADS = {".png", ".jpg", ".jpeg", ".webp"}


async def handle_attachments(elements) -> tuple[list[str], list[str], list[tuple[str, bytes, str]]]:
    confirmations: list[str] = []
    transcripts: list[str] = []
    images: list[tuple[str, bytes, str]] = []
    for el in elements or []:
        path = getattr(el, "path", None)
        name = getattr(el, "name", None) or "upload.bin"
        suffix = Path(name).suffix.lower()

        if suffix in IMAGE_UPLOADS:
            if not path or not Path(path).exists():
                confirmations.append(f"⚠️ {name}: file missing on server")
                continue
            images.append((name, Path(path).read_bytes(), suffix.lstrip(".")))
            continue

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
    return confirmations, transcripts, images


@cl.on_chat_start
async def on_chat_start():
    ensure_setup()
    await cl.Message(
        content=(
            f"Hello! I'm Jarvis, running locally on this machine.\n\n"
            f"- Ask me anything — I'll search the web when needed.\n"
            f"- Attach a **PDF / DOCX / TXT / MD** file and I'll index it for Q&A.\n"
            f"- Attach an **image** and I'll analyze it.\n"
            f"- Type `list my documents` to see what's indexed.\n"
            f"- Your past conversations appear in the sidebar — click to resume."
        )
    ).send()


@cl.on_chat_resume
async def on_chat_resume(thread: "ThreadDict"):
    ensure_setup()
    thread_id = thread["id"]
    cl.user_session.set("session_key", f"thread:{thread_id}")

    history = _rebuild_history_from_steps(thread.get("steps"))
    core.chat_histories[f"thread:{thread_id}"] = history


@cl.on_message
async def on_message(message: cl.Message):
    ensure_setup()
    owner = _username()

    notes: list[str] = []
    transcripts: list[str] = []
    images: list[tuple[str, bytes, str]] = []
    if message.elements:
        notes, transcripts, images = await handle_attachments(message.elements)

    if notes:
        await cl.Message(content="\n".join(notes)).send()
    for t in transcripts:
        await cl.Message(content=f"🎤 I heard: {t[:400]}").send()

    text = message.content.strip() or " ".join(transcripts).strip()

    if images:
        name, raw, fmt = images[0]
        question = text or "Describe this image in detail."
        botlog.log_user_msg(owner, "-", "web UI", f"🖼️ {name}: {question}", kind="image")
        body, failed = await core.vision_respond(session_key(), owner, raw, fmt, question)
        await cl.Message(content=body).send()
        botlog.log_reply(0, 0, "ok" if not failed else "vision-error")
        return

    if not text:
        return

    kind = "voice" if (transcripts and not message.content.strip()) else "text"
    preview = text if kind == "text" else " ".join(transcripts)
    botlog.log_user_msg(owner, "-", "web UI", preview, kind=kind)

    await _respond_and_send(text)


async def _respond_and_send(text: str):
    answer_msg = cl.Message(content="")
    await answer_msg.send()
    streamed: list[str] = []

    async def on_token(token: str):
        streamed.append(token)
        await answer_msg.stream_token(token)

    async def on_retry():
        # The streamed text belonged to a tool-calling round, not the final
        # answer — retract it so the real answer starts from a clean slate.
        streamed.clear()
        answer_msg.content = ""
        await answer_msg.update()

    t0 = time.perf_counter()
    try:
        body, sources, failed = await core.respond(
            session_key(), text, owner=_username(), on_token=on_token, on_retry=on_retry
        )
    except Exception as e:
        botlog.log_error_note(f"{type(e).__name__}: {str(e)[:200]}")
        await cl.Message(content="⚠️ Something went wrong processing that.").send()
        return
    if sources:
        links = "\n".join(f"[{i}] {url}" for i, url in enumerate(sources, start=1))
        body = f"{body}\n\n**Sources:**\n{links}"
    # Reconcile once: streamed text was raw model output; the final content is
    # post-processed (identity/sanitize) + sources footer + failure gating.
    answer_msg.content = body or "(no content)"
    await answer_msg.update()
    botlog.log_reply(
        time.perf_counter() - t0,
        len(sources),
        "ok" if not failed else "gated-fallback",
    )


if __name__ == "__main__":
    import threading

    import uvicorn
    from chainlit.server import app as server_app
    from starlette.applications import Starlette
    from starlette.responses import RedirectResponse

    host = os.environ.get("CHAINLIT_HOST", "0.0.0.0")
    base_port = int(os.environ.get("CHAINLIT_PORT", "8000"))

    ensure_setup()

    tls = _ensure_tls(Path(__file__).parent)
    if not tls:
        logger.info("HTTP mode - open http://localhost:%d", base_port)
        uvicorn.run(
            server_app, host=host, port=base_port, timeout_graceful_shutdown=5
        )
    else:
        cert_file, key_file = tls
        tls_port = int(os.environ.get("CHAINLIT_TLS_PORT", "8443"))

        from fastapi import FastAPI, Request

        redirect_app = FastAPI()

        @redirect_app.api_route(
            "/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"],
        )
        async def _redirect_to_https(request: Request, path: str):
            hostname = (request.headers.get("host") or "localhost").split(":")[0]
            query = f"?{request.url.query}" if request.url.query else ""
            return RedirectResponse(
                url=f"https://{hostname}:{tls_port}/{path}{query}", status_code=307
            )

        threading.Thread(
            target=uvicorn.run,
            args=(redirect_app,),
            kwargs={"host": host, "port": base_port, "log_level": "warning"},
            daemon=True,
        ).start()

        logger.info(
            "HTTPS enabled - open https://localhost:%d "
            "(accept the self-signed certificate warning once per device); "
            "http://localhost:%d now redirects there.",
            tls_port, base_port,
        )
        uvicorn.run(
            server_app,
            host=host,
            port=tls_port,
            ssl_certfile=str(cert_file),
            ssl_keyfile=str(key_file),
            timeout_graceful_shutdown=5,
        )
