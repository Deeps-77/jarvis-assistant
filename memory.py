import asyncio
import logging
import os
import sqlite3
import time
from pathlib import Path

import sqlite_vec
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)


class MemoryStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.embed_model = os.environ.get("EMBED_MODEL", "nomic-embed-text")
        self.top_k = max(1, int(os.environ.get("MEMORY_TOP_K", "4")))
        self.min_similarity = float(os.environ.get("MEMORY_MIN_SIMILARITY", "0.55"))
        self.max_per_chat = max(10, int(os.environ.get("MEMORY_MAX_PER_CHAT", "500")))
        self.enabled = True
        self._conn = None
        try:
            self.embedder = OllamaEmbeddings(model=self.embed_model, keep_alive=-1)
            dim = len(self.embedder.embed_query("dimension probe"))
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            self._conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS memory USING vec0("
                f"embedding float[{dim}] distance_metric=cosine, "
                f"chat_id TEXT partition key, "
                f"text TEXT, ts INTEGER)"
            )
            self._conn.commit()
            logger.info("Memory store ready at %s (dim=%d)", self.db_path.name, dim)
        except Exception as e:
            self.enabled = False
            logger.error("Memory store unavailable (%s); running without long-term memory", e)

    def _disable(self, reason: str):
        if not self.enabled:
            return
        self.enabled = False
        logger.warning("Long-term memory disabled after error: %s", reason)

    async def add(self, chat_id: str, text: str):
        if not self.enabled:
            return
        text = text.strip()
        if len(text) < 20:
            return
        try:
            t0 = time.perf_counter()
            vec = await asyncio.to_thread(self.embedder.embed_query, text[:4000])
            await asyncio.to_thread(self._add_sync, str(chat_id), vec, text)
            logger.debug("memory.add took %.0fms", (time.perf_counter() - t0) * 1000)
        except Exception as e:
            self._disable(f"add failed: {e}")

    def _add_sync(self, chat_id: str, vec: list[float], text: str):
        self._conn.execute(
            "INSERT INTO memory(embedding, chat_id, ts, text) VALUES (?, ?, ?, ?)",
            (sqlite_vec.serialize_float32(vec), chat_id, int(time.time() * 1000), text),
        )
        self._evict(chat_id)
        self._conn.commit()

    def _evict(self, chat_id: str):
        count = self._conn.execute(
            "SELECT COUNT(*) FROM memory WHERE chat_id = ?", (chat_id,)
        ).fetchone()[0]
        excess = count - self.max_per_chat
        if excess <= 0:
            return
        ids = [
            r[0]
            for r in self._conn.execute(
                "SELECT rowid FROM memory WHERE chat_id = ? ORDER BY ts ASC LIMIT ?",
                (chat_id, excess),
            )
        ]
        self._conn.executemany("DELETE FROM memory WHERE rowid = ?", [(i,) for i in ids])

    async def search(self, chat_id: str, query: str) -> list[str]:
        if not self.enabled:
            return []
        try:
            vec = await asyncio.to_thread(self.embedder.embed_query, query[:2000])
            return await asyncio.to_thread(self._search_sync, str(chat_id), vec)
        except Exception as e:
            self._disable(f"search failed: {e}")
            return []

    def _search_sync(self, chat_id: str, vec: list[float]) -> list[str]:
        cutoff = 1.0 - self.min_similarity
        rows = self._conn.execute(
            "SELECT text, distance FROM memory "
            "WHERE chat_id = ? AND embedding MATCH ? AND k = ? ORDER BY distance",
            (chat_id, sqlite_vec.serialize_float32(vec), self.top_k),
        ).fetchall()
        return [text for text, dist in rows if dist <= cutoff]

    async def clear_chat(self, chat_id: str):
        if not self.enabled:
            return
        try:
            await asyncio.to_thread(self._clear_sync, str(chat_id))
        except Exception as e:
            self._disable(f"clear failed: {e}")

    def _clear_sync(self, chat_id: str):
        self._conn.execute("DELETE FROM memory WHERE chat_id = ?", (chat_id,))
        self._conn.commit()

    async def count(self, chat_id: str) -> int:
        if not self.enabled:
            return 0
        try:
            return await asyncio.to_thread(
                lambda: self._conn.execute(
                    "SELECT COUNT(*) FROM memory WHERE chat_id = ?", (str(chat_id),)
                ).fetchone()[0]
            )
        except Exception:
            return 0
