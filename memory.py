import asyncio
import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import sqlite_vec
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import OllamaEmbeddings

from security import get_scanner, is_blocking

logger = logging.getLogger(__name__)

# Provenance tiers, recorded on every stored fact (mirrors OpenJarvis
# memory/store.py). "" is the defensive legacy value: recallable.
TRUST_AUTO = "auto"  # LLM-extracted and scanner-clean → recallable
TRUST_TRUSTED = "trusted"  # explicitly vouched for by the user
TRUST_UNTRUSTED = "untrusted"  # scanner flagged the fact → quarantined

RECALLABLE_TRUST_TIERS = frozenset({"", TRUST_AUTO, TRUST_TRUSTED})


@dataclass(slots=True)
class Fact:
    """A single durable memory entry."""

    text: str
    ts: int = 0
    trust: str = ""

    @property
    def trusted_for_recall(self) -> bool:
        """Whether this fact may be placed in model-facing context.

        Only ``untrusted`` is withheld. Unknown future tiers fail closed
        rather than silently becoming prompt input.
        """
        return (self.trust or "").strip().lower() in RECALLABLE_TRUST_TIERS


_EXTRACT_SYSTEM_PROMPT = (
    "You extract durable, long-term facts about the user from a single "
    "conversation exchange. A good fact is stable over time and useful in "
    "future conversations: preferences, identity, goals, ongoing projects, "
    "constraints, or relationships. Ignore one-off task details, small talk, "
    "and anything the assistant said about itself.\n\n"
    "Respond with ONLY a JSON array of short fact strings (each under 200 "
    "characters). If there is nothing worth remembering, respond with []."
)


class FactExtractor:
    """Distill memory-worthy facts from one exchange via the chat model.

    Extraction is best-effort and runs off the request path: any failure —
    dropped Ollama connection, timeout, unparseable output — degrades to
    "no facts" instead of propagating.
    """

    def __init__(self, llm, max_facts_per_turn: int = 10, max_fact_chars: int = 200):
        self._llm = llm
        self._max_facts_per_turn = max(1, max_facts_per_turn)
        self._max_fact_chars = max(40, max_fact_chars)

    async def extract(self, user_text: str, assistant_text: str = "") -> list[str]:
        """Return durable facts from the exchange. Never raises."""
        user_text = (user_text or "").strip()
        if not user_text:
            return []

        exchange = f"User: {user_text}"
        if assistant_text and assistant_text.strip():
            exchange += f"\nAssistant: {assistant_text.strip()}"

        try:
            resp = await self._llm.ainvoke(
                [
                    SystemMessage(content=_EXTRACT_SYSTEM_PROMPT),
                    HumanMessage(content=exchange),
                ]
            )
        except Exception:
            logger.debug("Memory extraction failed", exc_info=True)
            return []

        content = getattr(resp, "content", "") or ""
        if not isinstance(content, str):
            content = " ".join(str(p) for p in content)
        return self._parse(content)

    # -- parsing ------------------------------------------------------------

    def _parse(self, content: str) -> list[str]:
        """Parse model output into a clean, deduped, capped list of facts."""
        if not content or not content.strip():
            return []

        raw = self._coerce_to_list(content)

        facts: list[str] = []
        seen: set[str] = set()
        for item in raw:
            fact = self._clean_fact(item)
            if not fact:
                continue
            key = _normalize(fact)
            if key in seen:
                continue
            seen.add(key)
            facts.append(fact)
            if len(facts) >= self._max_facts_per_turn:
                break
        return facts

    def _coerce_to_list(self, content: str) -> list[str]:
        """Best-effort conversion of model output to a list of strings."""
        # 1. Try to locate and parse a JSON array anywhere in the output
        #    (models often wrap it in prose or code fences).
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except (json.JSONDecodeError, ValueError):
                pass

        # 2. Fall back to line-based parsing (markdown bullets / numbered).
        #    The per-fact injection scan downstream is the real gate, so
        #    permissive parsing here loses no safety.
        items: list[str] = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line)
            items.append(line)
        return items

    def _clean_fact(self, item: str) -> str:
        fact = str(item).strip().strip("\"'").strip()
        if not fact or fact.lower() in ("[]", "none", "n/a", "null"):
            return ""
        if len(fact) > self._max_fact_chars:
            fact = fact[: self._max_fact_chars].rstrip()
        return fact


def _normalize(text: str) -> str:
    return " ".join((text or "").casefold().split())


class MemoryStore:
    def __init__(self, db_path: Path, extractor: FactExtractor | None = None):
        self.db_path = Path(db_path)
        self.extractor = extractor
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
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            # vec0 tables cannot be ALTERed, so a stale `memory_facts` schema
            # (e.g. an extra metadata column from an earlier iteration, or an
            # embedding dimension change) would brick every INSERT. Reconcile
            # at startup: drop and recreate on any mismatch.
            self._ensure_facts_schema(dim)
            self._conn.commit()
            logger.info("Memory store ready at %s (dim=%d)", self.db_path.name, dim)
        except Exception as e:
            self.enabled = False
            logger.error("Memory store unavailable (%s); running without long-term memory", e)

    # The canonical vec0 shape. (The legacy `memory` table holding whole
    # exchanges is left on disk untouched but never read.)
    _FACTS_DDL = (
        "CREATE VIRTUAL TABLE memory_facts USING vec0("
        "embedding float[{dim}] distance_metric=cosine, "
        "chat_id TEXT partition key, "
        "text TEXT, trust TEXT, ts INTEGER)"
    )
    _FACTS_COLUMNS = {"embedding", "chat_id", "text", "trust", "ts"}

    def _ensure_facts_schema(self, dim: int):
        row = self._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'memory_facts'"
        ).fetchone()
        if row and row[0]:
            body = re.search(r"vec0\((.*)\)", row[0], re.DOTALL)
            cols = (
                {part.strip().split()[0].strip('"`[]') for part in body.group(1).split(",")}
                if body
                else set()
            )
            dim_m = re.search(r"float\[(\d+)\]", body.group(1)) if body else None
            declared_dim = int(dim_m.group(1)) if dim_m else -1
            if cols == self._FACTS_COLUMNS and declared_dim == dim:
                return
            count = self._conn.execute("SELECT COUNT(*) FROM memory_facts").fetchone()[0]
            logger.warning(
                "memory_facts schema mismatch (cols=%s, dim=%s); dropping and "
                "recreating, %d fact(s) discarded",
                sorted(cols),
                declared_dim,
                count,
            )
            self._conn.execute("DROP TABLE memory_facts")
        self._conn.execute(self._FACTS_DDL.format(dim=dim))

    def _disable(self, reason: str):
        if not self.enabled:
            return
        self.enabled = False
        logger.warning("Long-term memory disabled after error: %s", reason)

    # -- writing ------------------------------------------------------------

    async def learn_from_exchange(self, chat_id: str, user_text: str, assistant_text: str):
        """Extract facts from one exchange and store the safe ones.

        Pipeline (mirrors OpenJarvis memory/service.py): scan the exchange —
        HIGH/CRITICAL hits drop it entirely; extract facts; scan each fact —
        flagged ones are quarantined (stored as untrusted for audit, never
        recalled); dedupe against existing facts; insert; evict oldest.
        """
        if not self.enabled or self.extractor is None:
            return
        if not (user_text or "").strip():
            return
        try:
            scanner = get_scanner()
            if scanner is not None and is_blocking(
                scanner.scan(f"{user_text}\n{assistant_text}")
            ):
                logger.info("Memory: injection detected in exchange; skipping extraction")
                return

            facts = await self.extractor.extract(user_text, assistant_text)
            if not facts:
                return

            rows: list[tuple[str, str]] = []  # (fact, trust)
            for fact in facts:
                trust = TRUST_AUTO
                if scanner is not None and scanner.scan(fact).findings:
                    trust = TRUST_UNTRUSTED
                rows.append((fact, trust))

            quarantined = sum(1 for _, t in rows if t == TRUST_UNTRUSTED)
            if quarantined:
                logger.info("Memory: quarantined %d extracted fact(s) as untrusted", quarantined)

            await asyncio.to_thread(self._store_facts, str(chat_id), rows)
        except Exception as e:
            # Write-side failure only: skip this exchange but keep recall
            # alive (a broken embedder still disables the store via search).
            logger.warning("Memory learn failed; skipping this exchange: %s", e)

    def _store_facts(self, chat_id: str, rows: list[tuple[str, str]]):
        seen = {
            _normalize(r[0])
            for r in self._conn.execute(
                "SELECT text FROM memory_facts WHERE chat_id = ?", (chat_id,)
            )
        }
        fresh: list[tuple[str, str]] = []
        for text, trust in rows:
            key = _normalize(text)
            if key in seen:
                continue
            seen.add(key)
            fresh.append((text, trust))
        if not fresh:
            return

        vectors = self.embedder.embed_documents([text for text, _ in fresh])
        ts = int(time.time() * 1000)
        self._conn.executemany(
            "INSERT INTO memory_facts(embedding, chat_id, text, trust, ts) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (sqlite_vec.serialize_float32(vec), chat_id, text, trust, ts)
                for (text, trust), vec in zip(fresh, vectors)
            ],
        )
        self._evict(chat_id)
        self._conn.commit()
        logger.debug("Memory: stored %d new fact(s)", len(fresh))

    def _evict(self, chat_id: str):
        count = self._conn.execute(
            "SELECT COUNT(*) FROM memory_facts WHERE chat_id = ?", (chat_id,)
        ).fetchone()[0]
        excess = count - self.max_per_chat
        if excess <= 0:
            return
        ids = [
            r[0]
            for r in self._conn.execute(
                "SELECT rowid FROM memory_facts WHERE chat_id = ? ORDER BY ts ASC LIMIT ?",
                (chat_id, excess),
            )
        ]
        self._conn.executemany("DELETE FROM memory_facts WHERE rowid = ?", [(i,) for i in ids])

    # -- recall -------------------------------------------------------------

    async def search(self, chat_id: str, query: str) -> list[str]:
        """Return recallable facts similar to the query, nearest first."""
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
            "SELECT text, trust, distance FROM memory_facts "
            "WHERE chat_id = ? AND trust IN ('', 'auto', 'trusted') "
            "AND embedding MATCH ? AND k = ? ORDER BY distance",
            (chat_id, sqlite_vec.serialize_float32(vec), self.top_k),
        ).fetchall()
        # Belt and braces: re-check provenance in Python, exactly like
        # Fact.trusted_for_recall, so a future trust tier can never leak
        # into model-facing context because of a stale SQL clause.
        return [
            text
            for text, trust, dist in rows
            if dist <= cutoff and Fact(text=text, trust=trust).trusted_for_recall
        ]

    # -- maintenance ----------------------------------------------------------

    async def clear_chat(self, chat_id: str):
        if not self.enabled:
            return
        try:
            await asyncio.to_thread(self._clear_sync, str(chat_id))
        except Exception as e:
            self._disable(f"clear failed: {e}")

    def _clear_sync(self, chat_id: str):
        self._conn.execute("DELETE FROM memory_facts WHERE chat_id = ?", (chat_id,))
        self._conn.commit()

    async def count(self, chat_id: str) -> int:
        if not self.enabled:
            return 0
        try:
            return await asyncio.to_thread(
                lambda: self._conn.execute(
                    "SELECT COUNT(*) FROM memory_facts WHERE chat_id = ?", (str(chat_id),)
                ).fetchone()[0]
            )
        except Exception:
            return 0
