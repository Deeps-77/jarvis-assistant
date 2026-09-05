"""Per-workspace chat sessions for the code assistant.

Each workspace can hold many saved chats. A session captures the brain's
conversation history plus its mode so the user can resume an earlier chat
or start a fresh one from the workspace picker.

Persisted to ``code_sessions.json`` next to ``code_ui.py`` (same atomic
tmp+replace pattern as :class:`WorkspaceRegistry`). Layout::

    {
      "<workspace_root>": [
        {"id": ..., "title": ..., "created": ..., "updated": ...,
         "mode": "plan", "messages": [...]},
        ...
      ]
    }

Message payloads use LangChain's ``messages_to_dict`` /
``messages_from_dict`` so Human / AI (incl. tool_calls) / Tool messages
round-trip exactly.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

logger = logging.getLogger(__name__)

DEFAULT_SESSIONS_PATH = Path(__file__).parent.parent / "data" / "code_sessions.json"

#: Max saved chats per workspace; oldest pruned beyond this.
MAX_SESSIONS_PER_WORKSPACE = 20

#: Auto-title length for titles derived from the first user message.
TITLE_MAX_CHARS = 40


def auto_title(first_text: str) -> str:
    """Derive a short session title from the first user message."""
    s = " ".join((first_text or "").strip().split())
    if not s:
        return "Untitled session"
    if len(s) > TITLE_MAX_CHARS:
        s = s[: TITLE_MAX_CHARS - 1].rstrip() + "…"
    return s


@dataclass(slots=True)
class ChatSession:
    """One saved chat inside a workspace."""

    id: str
    workspace_root: str
    title: str
    created: float
    updated: float
    mode: str = "plan"
    messages: list[BaseMessage] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "workspace_root": self.workspace_root,
            "title": self.title,
            "created": self.created,
            "updated": self.updated,
            "mode": self.mode,
            "messages": messages_to_dict(self.messages),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ChatSession | None":
        try:
            sid = str(raw.get("id") or "")
            root = str(raw.get("workspace_root") or "")
            if not sid or not root:
                return None
            try:
                messages = messages_from_dict(raw.get("messages") or [])
            except Exception:
                logger.warning("Session %s has undecodable messages; starting empty", sid)
                messages = []
            return cls(
                id=sid,
                workspace_root=root,
                title=str(raw.get("title") or "Untitled session"),
                created=float(raw.get("created") or 0.0),
                updated=float(raw.get("updated") or 0.0),
                mode=str(raw.get("mode") or "plan"),
                messages=list(messages),
            )
        except (TypeError, ValueError) as e:
            logger.warning("Skipping malformed session row: %s", e)
            return None


class SessionStore:
    """JSON-backed per-workspace chat sessions. Thread-safe."""

    def __init__(
        self,
        path: Path | None = None,
        max_per_workspace: int = MAX_SESSIONS_PER_WORKSPACE,
    ) -> None:
        if path is None:
            from paths import code_sessions_file

            path = code_sessions_file()
        self.path = path
        self.max_per_workspace = max(1, max_per_workspace)
        self._lock = threading.Lock()
        self._data: dict[str, list[ChatSession]] = {}
        self._load()

    # ------------------------------------------------------------ persistence

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("SessionStore load failed (%s); starting empty", e)
            return
        if not isinstance(raw, dict):
            logger.warning("SessionStore top-level must be a mapping; starting empty")
            return
        for root, rows in raw.items():
            if not isinstance(rows, list):
                continue
            sessions = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                sess = ChatSession.from_dict({**row, "workspace_root": str(root)})
                if sess is not None:
                    sessions.append(sess)
            sessions.sort(key=lambda s: s.updated, reverse=True)
            self._data[str(root)] = sessions[: self.max_per_workspace]

    def _save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        payload = {
            root: [s.to_dict() for s in sessions]
            for root, sessions in self._data.items()
        }
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def _persist(self) -> None:
        try:
            self._save()
        except OSError as e:
            logger.warning("SessionStore save failed: %s", e)

    # ------------------------------------------------------------------ API

    def list(self, workspace_root: str) -> list[ChatSession]:
        """Sessions for a workspace, most recently updated first."""
        with self._lock:
            return list(self._data.get(str(workspace_root), []))

    def get(self, workspace_root: str, session_id: str) -> ChatSession | None:
        with self._lock:
            for s in self._data.get(str(workspace_root), []):
                if s.id == session_id or s.id.startswith(session_id):
                    return s
            return None

    def create(self, workspace_root: str, title: str = "", mode: str = "plan") -> ChatSession:
        now = time.time()
        sess = ChatSession(
            id=uuid.uuid4().hex[:12],
            workspace_root=str(workspace_root),
            title=title or "Untitled session",
            created=now,
            updated=now,
            mode=mode,
        )
        with self._lock:
            sessions = self._data.setdefault(str(workspace_root), [])
            sessions.insert(0, sess)
            del sessions[self.max_per_workspace :]
            self._persist()
        return sess

    def save(
        self,
        workspace_root: str,
        session_id: str,
        messages: list[BaseMessage],
        mode: str,
        title: str | None = None,
    ) -> ChatSession | None:
        """Replace history/mode (and optionally title) for a session."""
        with self._lock:
            sessions = self._data.get(str(workspace_root), [])
            for s in sessions:
                if s.id == session_id:
                    s.messages = list(messages)
                    s.mode = mode
                    if title:
                        s.title = title
                    s.updated = time.time()
                    sessions.sort(key=lambda x: x.updated, reverse=True)
                    self._persist()
                    return s
            return None

    def rename(self, workspace_root: str, session_id: str, title: str) -> ChatSession | None:
        title = " ".join(title.strip().split())
        if not title:
            return None
        with self._lock:
            sessions = self._data.get(str(workspace_root), [])
            for s in sessions:
                if s.id == session_id or s.id.startswith(session_id):
                    s.title = title[:80]
                    s.updated = time.time()
                    self._persist()
                    return s
            return None

    def delete(self, workspace_root: str, session_id: str) -> ChatSession | None:
        with self._lock:
            sessions = self._data.get(str(workspace_root), [])
            for i, s in enumerate(sessions):
                if s.id == session_id or s.id.startswith(session_id):
                    removed = sessions.pop(i)
                    if not sessions:
                        self._data.pop(str(workspace_root), None)
                    self._persist()
                    return removed
            return None

    def prune_workspace(self, workspace_root: str) -> int:
        """Drop all sessions for a workspace that no longer exists on disk."""
        with self._lock:
            sessions = self._data.pop(str(workspace_root), [])
            if sessions:
                self._persist()
            return len(sessions)


__all__ = [
    "ChatSession",
    "SessionStore",
    "auto_title",
    "DEFAULT_SESSIONS_PATH",
    "MAX_SESSIONS_PER_WORKSPACE",
    "TITLE_MAX_CHARS",
]
