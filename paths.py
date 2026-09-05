"""Central runtime-path resolution for Jarvis.

All persistent runtime files live under ``data/`` (plus ``logs/`` for
human-readable logs, kept at root for tailing convenience). Every path
helper falls back to the legacy root location so existing installs keep
working, and migrates the file forward on first use (move, never copy —
no duplication, no data loss).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "logs"


def data_path(name: str) -> Path:
    """Resolve ``data/<name>``, migrating a legacy root-level file if needed.

    - If ``data/<name>`` exists (or neither exists), return it.
    - If only ``<root>/<name>`` exists, move it into ``data/`` and return
      the new path. The move only happens when the target is absent, so
      concurrent first-runs cannot clobber each other (``shutil.move``
      onto an existing file would overwrite — we check first).
    """
    target = DATA_DIR / name
    legacy = PROJECT_ROOT / name
    if not target.exists() and legacy.exists() and legacy.is_file():
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.move(str(legacy), str(target))
                logger.info("Migrated %s -> %s", legacy, target)
        except OSError:
            logger.exception("Failed to migrate %s; using legacy path", legacy)
            return legacy
    return target


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def memory_db() -> Path:
    return data_path("memory.db")


def chat_history_file() -> Path:
    return data_path("chat_history.json")


def chat_threads_db() -> Path:
    return data_path("chat_threads.db")


def code_workspaces_file() -> Path:
    return data_path("code_workspaces.json")


def code_sessions_file() -> Path:
    return data_path("code_sessions.json")


def documents_dir() -> Path:
    """User-uploaded RAG documents (migrated as a directory, not a file)."""
    target = DATA_DIR / "documents"
    legacy = PROJECT_ROOT / "documents"
    if not target.exists() and legacy.is_dir():
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.move(str(legacy), str(target))
                logger.info("Migrated %s -> %s", legacy, target)
        except OSError:
            logger.exception("Failed to migrate %s; using legacy dir", legacy)
            return legacy
    return target


__all__ = [
    "PROJECT_ROOT",
    "DATA_DIR",
    "LOG_DIR",
    "data_path",
    "ensure_data_dir",
    "memory_db",
    "chat_history_file",
    "chat_threads_db",
    "code_workspaces_file",
    "code_sessions_file",
    "documents_dir",
]
