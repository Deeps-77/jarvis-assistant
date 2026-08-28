import asyncio
import hashlib
import logging
import os
import re
import sqlite3
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path

import sqlite_vec
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
MAX_CHUNKS_PER_DOC = 200


class UnsupportedFormatError(ValueError):
    pass


def extract_text(filename: str, raw: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(raw))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    elif suffix == ".docx":
        from docx import Document

        document = Document(BytesIO(raw))
        text = "\n".join(p.text for p in document.paragraphs)
    elif suffix in (".txt", ".md"):
        text = raw.decode("utf-8", errors="replace")
    else:
        raise UnsupportedFormatError(
            f"'{suffix}' is not supported. Use PDF, DOCX, TXT, or MD."
        )
    cleaned = re.sub(r"[ \t]+", " ", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned) < 20:
        raise ValueError(
            "No extractable text found - the file may be scanned images or empty."
        )
    return cleaned


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= size:
        return [text] if text else []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            cut = text.rfind(". ", start + size // 2, end)
            if cut != -1:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


class DocStore:
    def __init__(self, db_path: Path):
        self.embed_model = os.environ.get("EMBED_MODEL", "nomic-embed-text")
        self.search_k = max(1, int(os.environ.get("DOCS_TOP_K", "4")))
        self.min_similarity = float(os.environ.get("DOCS_MIN_SIMILARITY", "0.35"))
        self.enabled = True
        self._conn = None
        try:
            self.embedder = OllamaEmbeddings(model=self.embed_model, keep_alive=-1)
            dim = len(self.embedder.embed_query("dimension probe"))
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks USING vec0("
                f"embedding float[{dim}] distance_metric=cosine, "
                f"owner TEXT partition key, source TEXT, "
                f"text TEXT, chunk_index INTEGER, ts INTEGER)"
            )
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS doc_files(
                    owner TEXT NOT NULL,
                    source TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    chunks INTEGER NOT NULL,
                    ts INTEGER NOT NULL,
                    PRIMARY KEY(owner, source))"""
            )
            self._conn.commit()
            logger.info("Document store ready at %s (dim=%d)", db_path.name, dim)
        except Exception as e:
            self.enabled = False
            logger.error("Document store unavailable (%s); document features disabled", e)

    def _disable(self, reason: str):
        if not self.enabled:
            return
        self.enabled = False
        logger.warning("Document store disabled after error: %s", reason)

    async def ingest(self, owner: str, filename: str, raw: bytes) -> dict:
        owner = str(owner)
        if not self.enabled:
            return {"status": "error", "message": "document storage is unavailable"}
        try:
            digest = hashlib.sha256(raw).hexdigest()
            existing = await asyncio.to_thread(self._existing_hash, owner, filename)
            if existing == digest:
                return {"status": "unchanged", "chunks": 0}
            text = await asyncio.to_thread(extract_text, filename, raw)
            pieces = (await asyncio.to_thread(chunk_text, text))[:MAX_CHUNKS_PER_DOC]
            if not pieces:
                return {"status": "error", "message": "no indexable content found"}
            vectors = await asyncio.to_thread(self.embedder.embed_documents, pieces)
            await asyncio.to_thread(
                self._replace_sync, owner, filename, digest, pieces, vectors
            )
            return {"status": "added", "chunks": len(pieces)}
        except (UnsupportedFormatError, ValueError) as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            self._disable(f"ingest failed: {e}")
            return {"status": "error", "message": f"indexing failed: {e}"}

    def _existing_hash(self, owner: str, filename: str) -> str | None:
        owner = str(owner)
        row = self._conn.execute(
            "SELECT sha256 FROM doc_files WHERE owner = ? AND source = ?",
            (owner, filename),
        ).fetchone()
        return row[0] if row else None

    def _replace_sync(self, owner: str, filename: str, digest: str, pieces: list[str], vectors: list):
        owner = str(owner)
        ts = int(time.time() * 1000)
        # Serialize the delete+insert so concurrent processes can't interleave
        # and create duplicate chunks (BEGIN IMMEDIATE acquires the write lock
        # immediately; busy_timeout makes it wait instead of raising).
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._delete_sync(owner, filename)
            self._conn.execute(
                "INSERT INTO doc_files(owner, source, sha256, chunks, ts) VALUES (?, ?, ?, ?, ?)",
                (owner, filename, digest, len(pieces), ts),
            )
            rows = [
                (
                    sqlite_vec.serialize_float32(vec),
                    owner,
                    filename,
                    piece,
                    idx,
                    ts,
                )
                for idx, (piece, vec) in enumerate(zip(pieces, vectors))
            ]
            self._conn.executemany(
                "INSERT INTO doc_chunks(embedding, owner, source, text, chunk_index, ts) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    async def search(self, owner: str, query: str, k: int | None = None) -> list[tuple[str, int, str]]:
        owner = str(owner)
        if not self.enabled:
            return []
        k = k or self.search_k
        try:
            vec = await asyncio.to_thread(self.embedder.embed_query, query[:2000])
            return await asyncio.to_thread(self._search_sync, owner, vec, k)
        except Exception as e:
            self._disable(f"search failed: {e}")
            return []

    def _search_sync(self, owner: str, vec: list[float], k: int) -> list[tuple[str, int, str]]:
        owner = str(owner)
        cutoff = 1.0 - self.min_similarity
        rows = self._conn.execute(
            "SELECT source, chunk_index, text, distance FROM doc_chunks "
            "WHERE owner = ? AND embedding MATCH ? AND k = ? ORDER BY distance",
            (owner, sqlite_vec.serialize_float32(vec), k),
        ).fetchall()
        return [(r[0], r[1], r[2]) for r in rows if r[3] <= cutoff]

    async def doc_text(self, owner: str, source: str, cap: int = 6000) -> tuple[str, bool]:
        owner = str(owner)
        if not self.enabled:
            return "", True
        try:
            return await asyncio.to_thread(self._doc_text_sync, owner, source, cap)
        except Exception as e:
            self._disable(f"doc_text failed: {e}")
            return "", True

    def _doc_text_sync(self, owner: str, source: str, cap: int) -> tuple[str, bool]:
        owner = str(owner)
        rows = self._conn.execute(
            "SELECT text FROM doc_chunks WHERE owner = ? AND source = ? ORDER BY chunk_index",
            (owner, source),
        ).fetchall()
        parts, total, truncated = [], 0, False
        for (piece,) in rows:
            if total + len(piece) > cap:
                truncated = True
                break
            parts.append(piece)
            total += len(piece)
        return "\n\n".join(parts), truncated

    async def list_docs(self, owner: str) -> list[dict]:
        owner = str(owner)
        if not self.enabled:
            return []
        try:
            return await asyncio.to_thread(self._list_sync, owner)
        except Exception as e:
            self._disable(f"list failed: {e}")
            return []

    def _list_sync(self, owner: str) -> list[dict]:
        owner = str(owner)
        rows = self._conn.execute(
            "SELECT source, chunks, ts FROM doc_files WHERE owner = ? ORDER BY ts DESC",
            (owner,),
        ).fetchall()
        return [
            {
                "source": r[0],
                "chunks": r[1],
                "date": datetime.fromtimestamp(r[2] / 1000).strftime("%Y-%m-%d"),
            }
            for r in rows
        ]

    async def count(self, owner: str) -> int:
        if not self.enabled:
            return 0
        try:
            return await asyncio.to_thread(
                lambda: self._conn.execute(
                    "SELECT COUNT(*) FROM doc_files WHERE owner = ?", (str(owner),)
                ).fetchone()[0]
            )
        except Exception:
            return 0

    async def delete(self, owner: str, source: str) -> bool:
        owner = str(owner)
        if not self.enabled:
            return False
        try:
            return await asyncio.to_thread(self._delete_and_report_sync, owner, source)
        except Exception as e:
            self._disable(f"delete failed: {e}")
            return False

    def _delete_sync(self, owner: str, source: str):
        owner = str(owner)
        rowids = [
            r[0]
            for r in self._conn.execute(
                "SELECT rowid FROM doc_chunks WHERE owner = ? AND source = ?", (owner, source)
            )
        ]
        self._conn.executemany("DELETE FROM doc_chunks WHERE rowid = ?", [(i,) for i in rowids])
        self._conn.execute(
            "DELETE FROM doc_files WHERE owner = ? AND source = ?", (owner, source)
        )

    def _delete_and_report_sync(self, owner: str, source: str) -> bool:
        existed = self._existing_hash(owner, source) is not None
        self._delete_sync(owner, source)
        self._conn.commit()
        return existed
