import asyncio
import json
import logging
import os
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from chainlit.data.base import BaseDataLayer
from chainlit.types import Pagination, PageInfo, ThreadDict, ThreadFilter

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
        self._conn.execute("PRAGMA journal_mode=WAL")
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
            self._conn.close()
        except Exception:
            pass

    async def build_debug_url(self) -> str:
        return ""

    # ── users ────────────────────────────────────────────────────────

    async def get_user(self, identifier: str):
        row = await asyncio.to_thread(
            self._conn.execute,
            "SELECT identifier, created_at, meta FROM users WHERE identifier = ?",
            (identifier,),
        )
        row = row.fetchone()
        if not row:
            return None
        from chainlit.types import PersistedUser

        return PersistedUser(
            identifier=row[0], createdAt=datetime.fromtimestamp(row[1] / 1000), metadata=json.loads(row[2])
        )

    async def create_user(self, user):
        await asyncio.to_thread(
            lambda: self._conn.execute(
                "INSERT OR REPLACE INTO users(identifier, created_at, meta) VALUES (?, ?, ?)",
                (user.identifier, _now_ms(), json.dumps(getattr(user, "metadata", {}) or {})),
            )
        )
        self._conn.commit()
        return True

    # ── steps ────────────────────────────────────────────────────────

    async def create_step(self, step_dict):
        await asyncio.to_thread(self._upsert_step_sync, step_dict)

    async def update_step(self, step_dict):
        await asyncio.to_thread(self._upsert_step_sync, step_dict)

    def _upsert_step_sync(self, step_dict):
        d = dict(step_dict)
        self._conn.execute(
            "INSERT OR REPLACE INTO steps(id, thread_id, created_at, step_json) VALUES (?, ?, ?, ?)",
            (
                d["id"],
                d.get("threadId"),
                int(d.get("createdAt") and time.mktime(datetime.fromisoformat(d["createdAt"]).timetuple()) * 1000
                    or _now_ms()),
                json.dumps(d, ensure_ascii=False),
            ),
        )
        self._conn.execute(
            "UPDATE threads SET updated_at = ? WHERE id = ?", (_now_ms(), d.get("threadId"))
        )
        self._conn.commit()

    async def delete_step(self, step_id: str):
        await asyncio.to_thread(
            lambda: self._conn.execute("DELETE FROM steps WHERE id = ?", (step_id,))
        )
        self._conn.commit()

    # ── elements ─────────────────────────────────────────────────────

    async def create_element(self, element):
        await asyncio.to_thread(
            lambda: self._conn.execute(
                "INSERT OR REPLACE INTO elements(id, thread_id, meta) VALUES (?, ?, ?)",
                (
                    getattr(element, "id", None) or str(uuid.uuid4()),
                    getattr(element, "forId", None),
                    json.dumps({"name": getattr(element, "name", None)}, ensure_ascii=False),
                ),
            )
        )
        self._conn.commit()

    async def get_element(self, element_id: str, thread_id: str | None = None):
        return None

    async def delete_element(self, element_id: str, thread_id: str | None = None):
        pass

    # ── feedback ─────────────────────────────────────────────────────

    async def delete_feedback(self, feedback_id: str):
        await asyncio.to_thread(
            lambda: self._conn.execute("DELETE FROM feedbacks WHERE id = ?", (feedback_id,))
        )
        self._conn.commit()

    async def upsert_feedback(self, feedback):
        fid = getattr(feedback, "id", None) or str(uuid.uuid4())
        await asyncio.to_thread(
            lambda: self._conn.execute(
                "INSERT OR REPLACE INTO feedbacks(id, payload) VALUES (?, ?)",
                (fid, json.dumps(feedback.__dict__, ensure_ascii=False, default=str)),
            )
        )
        self._conn.commit()
        return fid

    async def get_favorite_steps(self, user_identifier: str):
        return []

    async def set_step_favorite(self, step_id: str, is_favorite: bool):
        return False

    # ── threads ──────────────────────────────────────────────────────

    def _thread_dict_sync(self, thread_id: str, include_steps: bool = True):
        row = self._conn.execute(
            "SELECT id, owner, name, created_at, metadata, tags FROM threads WHERE id = ?",
            (thread_id,),
        ).fetchone()
        if not row:
            return None
        steps = []
        if include_steps:
            srows = self._conn.execute(
                "SELECT step_json FROM steps WHERE thread_id = ? ORDER BY created_at, id",
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
        row = await asyncio.to_thread(
            self._conn.execute,
            "SELECT owner FROM threads WHERE id = ?",
            (thread_id,),
        )
        row = row.fetchone()
        return row[0] if row else "unknown"

    async def update_thread(self, thread_id: str, thread_dict: ThreadDict):
        name = thread_dict.get("name") or ""
        metadata = json.dumps(thread_dict.get("metadata") or {}, ensure_ascii=False)
        tags = json.dumps(thread_dict.get("tags") or [], ensure_ascii=False)
        exists = await asyncio.to_thread(
            lambda: self._conn.execute(
                "SELECT 1 FROM threads WHERE id = ?", (thread_id,)
            ).fetchone()
        )
        if exists:
            await asyncio.to_thread(
                lambda: self._conn.execute(
                    "UPDATE threads SET name = ?, metadata = ?, tags = ?, updated_at = ? WHERE id = ?",
                    (name, metadata, tags, _now_ms(), thread_id),
                )
            )
        else:
            await asyncio.to_thread(
                lambda: self._conn.execute(
                    "INSERT INTO threads(id, owner, name, created_at, updated_at, metadata, tags) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (thread_id, thread_dict.get("userId") or "", name, _now_ms(), _now_ms(), metadata, tags),
                )
            )
        self._conn.commit()

    async def delete_thread(self, thread_id: str):
        def _delete():
            rowids = [
                r[0]
                for r in self._conn.execute(
                    "SELECT rowid FROM steps WHERE thread_id = ?", (thread_id,)
                )
            ]
            self._conn.executemany("DELETE FROM steps WHERE rowid = ?", [(i,) for i in rowids])
            self._conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,))

        await asyncio.to_thread(_delete)
        self._conn.commit()

    async def list_threads(self, pagination: Pagination, filters: ThreadFilter) -> dict:
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
            total = self._conn.execute(
                f"SELECT COUNT(*) FROM threads WHERE {where}", params
            ).fetchone()[0]
            offset = int(pagination.cursor) if pagination.cursor else 0
            rows = self._conn.execute(
                f"SELECT id, name, updated_at FROM threads WHERE {where} "
                "ORDER BY updated_at DESC LIMIT ? OFFSET ?",
                (*params, pagination.first + 1, offset),
            ).fetchall()
            has_next = len(rows) > pagination.first
            rows = rows[: pagination.first]
            items = []
            for tid, name, updated in rows:
                items.append(
                    {
                        "id": tid,
                        "name": name or "(untitled)",
                        "createdAt": _iso(updated),
                        "userId": owner,
                        "userIdentifier": owner,
                        "metadata": {},
                        "tags": [],
                        "steps": [],
                        "elements": [],
                    }
                )
            end_cursor = str(offset + len(items)) if has_next else None
            page_info = {
                "hasNextPage": has_next,
                "startCursor": None,
                "endCursor": end_cursor,
            }
            return {"data": items, "pageInfo": page_info}

        return await asyncio.to_thread(_list_sync)
