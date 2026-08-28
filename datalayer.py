import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from chainlit.data.base import BaseDataLayer
from chainlit.types import (
    Pagination,
    PageInfo,
    PaginatedResponse,
    ThreadDict,
    ThreadFilter,
)

logger = logging.getLogger(__name__)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _iso(ts_ms: int) -> str:
    return (
        datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).astimezone().isoformat(timespec="seconds")
    )


class SQLiteDataLayer(BaseDataLayer):
    """Chainlit data layer backed by a local SQLite database.

    Enables the sidebar chat-management UI: past conversations are listed,
    resumable and deletable, and survive restarts.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users(
                identifier TEXT PRIMARY KEY,
                created_at INTEGER NOT NULL,
                meta TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS threads(
                id TEXT PRIMARY KEY,
                owner TEXT NOT NULL,
                name TEXT,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                tags TEXT NOT NULL DEFAULT '[]'
            );
            CREATE INDEX IF NOT EXISTS idx_threads_owner
                ON threads(owner, updated_at DESC);
            CREATE TABLE IF NOT EXISTS steps(
                id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                step_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_steps_thread
                ON steps(thread_id, created_at);
            CREATE TABLE IF NOT EXISTS elements(
                id TEXT PRIMARY KEY,
                thread_id TEXT,
                meta TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS feedbacks(
                id TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            );
            """
        )
        self._conn.commit()
        logger.info("Chat data layer ready at %s", self.db_path.name)

    async def close(self) -> None:
        try:
            with self._lock:
                self._conn.close()
        except Exception:
            pass

    async def build_debug_url(self) -> str:
        return ""

    # ── users ────────────────────────────────────────────────────────

    async def get_user(self, identifier: str):
        row = await asyncio.to_thread(self._get_user_sync, identifier)
        if not row:
            return None
        from chainlit.user import PersistedUser

        return PersistedUser(
            id=row[0],
            identifier=row[0],
            createdAt=_iso(row[1]),
            metadata=json.loads(row[2]),
        )

    def _get_user_sync(self, identifier: str):
        with self._lock:
            return self._conn.execute(
                "SELECT identifier, created_at, meta FROM users WHERE identifier = ?",
                (identifier,),
            ).fetchone()

    async def create_user(self, user):
        created = _now_ms()
        meta = json.dumps(getattr(user, "metadata", {}) or {})
        await asyncio.to_thread(self._create_user_sync, user.identifier, created, meta)
        from chainlit.user import PersistedUser

        return PersistedUser(
            id=user.identifier,
            identifier=user.identifier,
            createdAt=_iso(created),
            metadata=getattr(user, "metadata", {}) or {},
        )

    def _create_user_sync(self, identifier, created, meta):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO users(identifier, created_at, meta) VALUES (?, ?, ?)",
                (identifier, created, meta),
            )
            self._conn.commit()

    # ── steps ────────────────────────────────────────────────────────

    async def create_step(self, step_dict):
        await asyncio.to_thread(self._upsert_step_sync, step_dict)

    async def update_step(self, step_dict):
        await asyncio.to_thread(self._upsert_step_sync, step_dict)

    def _upsert_step_sync(self, step_dict):
        d = dict(step_dict)
        step_id = str(d.get("id"))
        thread_id = str(d["threadId"]) if d.get("threadId") is not None else None
        created = d.get("createdAt")
        if isinstance(created, (int, float)):
            created_at = int(created)
        elif isinstance(created, str) and created:
            try:
                created_at = int(
                    round(datetime.fromisoformat(created).timestamp() * 1000)
                )
            except (ValueError, TypeError):
                created_at = _now_ms()
        else:
            created_at = _now_ms()
        try:
            step_json = json.dumps(d, ensure_ascii=False, default=str)
        except TypeError:
            step_json = json.dumps(
                {k: str(v) for k, v in d.items()}, ensure_ascii=False, default=str
            )
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO steps(id, thread_id, created_at, step_json) VALUES (?, ?, ?, ?)",
                (step_id, thread_id, created_at, step_json),
            )
            self._conn.execute(
                "UPDATE threads SET updated_at = ? WHERE id = ?", (_now_ms(), thread_id)
            )
            self._conn.commit()

    async def delete_step(self, step_id: str):
        await asyncio.to_thread(self._delete_step_sync, step_id)

    def _delete_step_sync(self, step_id: str):
        with self._lock:
            self._conn.execute("DELETE FROM steps WHERE id = ?", (step_id,))
            self._conn.commit()

    # ── elements ─────────────────────────────────────────────────────

    async def create_element(self, element):
        el_id = getattr(element, "id", None) or str(uuid.uuid4())
        thread_id = getattr(element, "forId", None)
        meta = json.dumps(
            {"name": getattr(element, "name", None)}, ensure_ascii=False
        )
        await asyncio.to_thread(self._create_element_sync, el_id, thread_id, meta)

    def _create_element_sync(self, el_id, thread_id, meta):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO elements(id, thread_id, meta) VALUES (?, ?, ?)",
                (el_id, thread_id, meta),
            )
            self._conn.commit()

    async def get_element(self, element_id: str, thread_id: str | None = None):
        return None

    async def delete_element(self, element_id: str, thread_id: str | None = None):
        pass

    # ── feedback ─────────────────────────────────────────────────────

    async def delete_feedback(self, feedback_id: str):
        await asyncio.to_thread(self._delete_feedback_sync, feedback_id)

    def _delete_feedback_sync(self, feedback_id: str):
        with self._lock:
            self._conn.execute("DELETE FROM feedbacks WHERE id = ?", (feedback_id,))
            self._conn.commit()

    async def upsert_feedback(self, feedback):
        fid = getattr(feedback, "id", None) or str(uuid.uuid4())
        payload = json.dumps(feedback.__dict__, ensure_ascii=False, default=str)
        await asyncio.to_thread(self._upsert_feedback_sync, fid, payload)
        return fid

    def _upsert_feedback_sync(self, fid, payload):
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO feedbacks(id, payload) VALUES (?, ?)",
                (fid, payload),
            )
            self._conn.commit()

    async def get_favorite_steps(self, user_identifier: str):
        return []

    async def set_step_favorite(self, step_id: str, is_favorite: bool):
        return False

    # ── threads ──────────────────────────────────────────────────────

    def _thread_dict_sync(self, thread_id: str, include_steps: bool = True):
        with self._lock:
            row = self._conn.execute(
                "SELECT id, owner, name, created_at, metadata, tags FROM threads WHERE id = ?",
                (thread_id,),
            ).fetchone()
            if not row:
                return None
            steps = []
            if include_steps:
                srows = self._conn.execute(
                    "SELECT step_json FROM steps WHERE thread_id = ? ORDER BY created_at, rowid",
                    (thread_id,),
                ).fetchall()
                steps = [json.loads(s[0]) for s in srows]
        return {
            "id": row[0],
            "userId": row[1],
            "userIdentifier": row[1],
            "createdAt": _iso(row[3]),
            "name": row[2],
            "metadata": json.loads(row[4]),
            "tags": json.loads(row[5]),
            "steps": steps,
            "elements": [],
        }

    async def get_thread(self, thread_id: str) -> ThreadDict | None:
        return await asyncio.to_thread(self._thread_dict_sync, thread_id)

    async def get_thread_author(self, thread_id: str) -> str:
        return await asyncio.to_thread(self._get_thread_author_sync, thread_id)

    def _get_thread_author_sync(self, thread_id: str) -> str:
        with self._lock:
            row = self._conn.execute(
                "SELECT owner FROM threads WHERE id = ?", (thread_id,)
            ).fetchone()
        return row[0] if row else "unknown"

    async def update_thread(
        self,
        thread_id: str,
        name: str | None = None,
        user_id: str | None = None,
        metadata: dict | None = None,
        tags: list | None = None,
    ):
        metadata_json = (
            json.dumps(metadata, ensure_ascii=False) if metadata is not None else None
        )
        tags_json = (
            json.dumps(tags, ensure_ascii=False) if tags is not None else None
        )
        await asyncio.to_thread(
            self._update_thread_sync,
            thread_id,
            name,
            user_id or "",
            metadata_json,
            tags_json,
        )

    def _update_thread_sync(self, thread_id, name, user_id, metadata_json, tags_json):
        with self._lock:
            exists = self._conn.execute(
                "SELECT 1 FROM threads WHERE id = ?", (thread_id,)
            ).fetchone()
            if exists:
                sets, params = [], []
                if name is not None:
                    sets.append("name = ?")
                    params.append(name)
                if metadata_json is not None:
                    sets.append("metadata = ?")
                    params.append(metadata_json)
                if tags_json is not None:
                    sets.append("tags = ?")
                    params.append(tags_json)
                sets.append("updated_at = ?")
                params.append(_now_ms())
                params.append(thread_id)
                self._conn.execute(
                    f"UPDATE threads SET {', '.join(sets)} WHERE id = ?",
                    params,
                )
            else:
                self._conn.execute(
                    "INSERT INTO threads(id, owner, name, created_at, updated_at, metadata, tags) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        thread_id,
                        user_id,
                        name or "",
                        _now_ms(),
                        _now_ms(),
                        metadata_json or "{}",
                        tags_json or "[]",
                    ),
                )
            self._conn.commit()

    async def delete_thread(self, thread_id: str):
        await asyncio.to_thread(self._delete_thread_sync, thread_id)

    def _delete_thread_sync(self, thread_id: str):
        with self._lock:
            rowids = [
                r[0]
                for r in self._conn.execute(
                    "SELECT rowid FROM steps WHERE thread_id = ?", (thread_id,)
                )
            ]
            self._conn.executemany("DELETE FROM steps WHERE rowid = ?", [(i,) for i in rowids])
            self._conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
            self._conn.commit()

    async def list_threads(self, pagination: Pagination, filters: ThreadFilter):
        owner = filters.userId
        search = filters.search

        def _list_sync():
            clauses, params = ["1 = 1"], []
            if owner:
                clauses.append("owner = ?")
                params.append(owner)
            if search:
                clauses.append("LOWER(name) LIKE ?")
                params.append(f"%{search.lower()}%")
            where = " AND ".join(clauses)
            offset = int(pagination.cursor) if pagination.cursor else 0
            with self._lock:
                rows = self._conn.execute(
                    f"SELECT id, name, created_at, updated_at FROM threads WHERE {where} "
                    "ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                    (*params, pagination.first + 1, offset),
                ).fetchall()
            has_next = len(rows) > pagination.first
            rows = rows[: pagination.first]
            items = []
            for tid, name, created, updated in rows:
                items.append(
                    {
                        "id": tid,
                        "name": name or "(untitled)",
                        "createdAt": _iso(created),
                        "userId": owner,
                        "userIdentifier": owner,
                        "metadata": {},
                        "tags": [],
                        "steps": [],
                        "elements": [],
                    }
                )
            end_cursor = str(offset + len(items)) if has_next else None
            page_info = PageInfo(
                hasNextPage=has_next,
                startCursor=None,
                endCursor=end_cursor,
            )
            return PaginatedResponse(pageInfo=page_info, data=items)

        return await asyncio.to_thread(_list_sync)
