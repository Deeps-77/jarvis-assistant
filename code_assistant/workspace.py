"""Workspace model for the code assistant.

A ``Workspace`` represents one project root the user has opened in the code
assistant UI. Every tool call is validated against the workspace: paths are
resolved to absolute form, traversal outside the root is refused, and any
file/directory matching the deny-glob list is hidden.

Why deny-globs instead of an allow-list?
- Most users want to point the assistant at an existing repo and have it
  "just work" on the source tree, with the obvious junk (.git, build
  artefacts, virtualenvs, the sandbox dir itself) filtered out. An
  allow-list forces the user to enumerate every extension before they can
  do anything useful.

The registry (``WorkspaceRegistry``) persists the user's opened workspaces
to a JSON file so they survive restarts. Each registry row is keyed by the
absolute root path; the user-facing label is the repo's leaf directory name.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path

logger = logging.getLogger(__name__)


DEFAULT_DENY_GLOBS: list[str] = [
    ".git/",
    ".jarvis-sandbox/",
    ".venv/",
    "venv/",
    "__pycache__/",
    "node_modules/",
    "dist/",
    "build/",
    ".idea/",
    ".vscode/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".tox/",
    "*.pyc",
    "*.pyo",
    "*.wasm",
    "*.map",
    "*.min.js",
    "*.min.css",
    "*.lock",
    "package-lock.json",
]


class WorkspaceError(ValueError):
    """Raised when a path fails validation."""


@dataclass(slots=True)
class Workspace:
    """One project root. Tools operate relative to ``root``."""

    root: Path
    name: str = ""
    deny_globs: list[str] = field(default_factory=lambda: list(DEFAULT_DENY_GLOBS))
    max_file_size_bytes: int = 2 * 1024 * 1024  # 2 MB safety cap for reads

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        if not self.root.exists():
            raise WorkspaceError(f"Workspace root does not exist: {self.root}")
        if not self.root.is_dir():
            raise WorkspaceError(f"Workspace root is not a directory: {self.root}")
        if not self.name:
            self.name = self.root.name or self.root.as_posix()
        # Sandbox dir is always implicitly denied, even if the user removed it.
        if ".jarvis-sandbox/" not in self.deny_globs:
            self.deny_globs.append(".jarvis-sandbox/")

    @property
    def sandbox_dir(self) -> Path:
        return self.root / ".jarvis-sandbox"

    @property
    def sandbox_tmp(self) -> Path:
        return self.sandbox_dir / "tmp"

    @property
    def sandbox_trash(self) -> Path:
        return self.sandbox_dir / "trash"

    @property
    def sandbox_pending(self) -> Path:
        return self.sandbox_dir / "pending"

    def ensure_sandbox(self) -> None:
        """Create sandbox subdirs in-place (idempotent)."""
        for sub in (self.sandbox_dir, self.sandbox_tmp, self.sandbox_trash, self.sandbox_pending):
            sub.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------- path validation

    def _to_posix(self, p: Path | str) -> str:
        return str(p).replace("\\", "/") if isinstance(p, Path) else p.replace("\\", "/")

    def is_denied(self, rel_path: str | Path) -> bool:
        """Return True if the relative path matches any deny-glob."""
        rel = self._to_posix(rel_path) if isinstance(rel_path, Path) else self._to_posix(rel_path)
        # Empty path is the root — never denied.
        if not rel or rel == ".":
            return False
        for glob in self.deny_globs:
            if glob.endswith("/"):
                prefix = glob.rstrip("/")
                if rel == prefix or rel.startswith(prefix + "/"):
                    return True
            else:
                if fnmatch(rel, glob):
                    return True
                # Also match against any path component for patterns like "*.pyc".
                if "/" in rel and fnmatch(rel.rsplit("/", 1)[-1], glob):
                    return True
        return False

    def validate_path(self, rel_path: str | Path, *, must_exist: bool = True) -> Path:
        """Resolve ``rel_path`` against the workspace root and validate it.

        - Raises :class:`WorkspaceError` on traversal, denied globs, or
          missing paths.
        - Returns the absolute, resolved ``Path`` on success.
        """
        if rel_path is None or str(rel_path).strip() == "":
            return self.root
        rel_str = self._to_posix(rel_path)
        # Reject absolute paths outright (force relative use).
        if rel_str.startswith("/") or (len(rel_str) >= 2 and rel_str[1] == ":"):
            raise WorkspaceError(f"Paths must be relative to the workspace root, got: {rel_str!r}")
        # Strip leading "./" segments (one or more). Do NOT strip bare dots —
        # a path starting with "." like ".jarvis-sandbox" must be preserved
        # verbatim so the deny-glob check below can match it. Also do NOT
        # strip ".." — that's a traversal signal we want to catch.
        while rel_str.startswith("./"):
            rel_str = rel_str[2:]
        # Detect any ".." segment — refuse rather than try to be clever.
        if any(part == ".." for part in rel_str.split("/")):
            raise WorkspaceError(f"Path traversal is not allowed: {rel_str!r}")
        if self.is_denied(rel_str):
            raise WorkspaceError(f"Path matches a workspace deny-glob: {rel_str!r}")
        candidate = (self.root / rel_str).resolve()
        # Belt and suspenders: ensure the resolved path is still inside root.
        try:
            candidate.relative_to(self.root)
        except ValueError:
            raise WorkspaceError(f"Path resolves outside the workspace root: {rel_str!r}")
        if must_exist and not candidate.exists():
            raise WorkspaceError(f"Path does not exist: {rel_str!r}")
        return candidate

    # ----------------------------------------------------------- listing

    def iter_files(
        self,
        *,
        max_depth: int | None = None,
        include_hidden: bool = False,
    ):
        """Yield ``Path`` objects for every non-denied file under the root.

        Hidden files (names starting with ``.``) are skipped unless
        ``include_hidden`` is True. Deny-globs take precedence over
        ``include_hidden``.
        """
        stack: list[tuple[Path, int]] = [(self.root, 0)]
        while stack:
            current, depth = stack.pop()
            try:
                entries = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except (PermissionError, OSError) as e:
                logger.debug("skip %s: %s", current, e)
                continue
            for entry in entries:
                rel = entry.relative_to(self.root)
                if not include_hidden and any(part.startswith(".") for part in rel.parts):
                    continue
                if self.is_denied(rel):
                    continue
                if entry.is_dir():
                    if max_depth is None or depth + 1 <= max_depth:
                        stack.append((entry, depth + 1))
                elif entry.is_file():
                    yield entry


_LANGUAGE_BY_EXT: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".go": "go", ".rs": "rust", ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp",
    ".java": "java", ".kt": "kotlin", ".swift": "swift",
    ".rb": "ruby", ".php": "php", ".cs": "csharp",
    ".html": "html", ".css": "css", ".scss": "scss", ".sass": "sass", ".less": "less",
    ".md": "markdown", ".rst": "rst", ".txt": "text",
    ".json": "json", ".yaml": "yaml", ".yml": "yaml", ".toml": "toml", ".ini": "ini",
    ".sh": "shell", ".bash": "shell", ".zsh": "shell", ".ps1": "powershell",
    ".sql": "sql", ".xml": "xml", ".svg": "svg",
}


def guess_language(path: Path) -> str:
    return _LANGUAGE_BY_EXT.get(path.suffix.lower(), "text")


# -------------------------------------------------------------- registry


@dataclass
class SavedWorkspace:
    root: str
    name: str
    last_used: str  # ISO timestamp

    def to_dict(self) -> dict:
        return {"root": self.root, "name": self.name, "last_used": self.last_used}


class WorkspaceRegistry:
    """JSON-backed list of recently-opened workspaces.

    Persisted to ``code_workspaces.json`` next to ``code_ui.py``. Used by the
    Chainlit sidebar to show "recent" workspaces and by the workspace picker
    to autocomplete known paths.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path(__file__).parent.parent / "code_workspaces.json"
        self._lock = threading.Lock()
        self._entries: list[SavedWorkspace] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            self._entries = [
                SavedWorkspace(root=e["root"], name=e.get("name", ""), last_used=e.get("last_used", ""))
                for e in raw
                if e.get("root")
            ]
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("WorkspaceRegistry load failed (%s); starting empty", e)
            self._entries = []

    def _save(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps([e.to_dict() for e in self._entries], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def touch(self, workspace: Workspace) -> None:
        """Record/refresh an entry for ``workspace`` and move it to the top."""
        with self._lock:
            key = str(workspace.root)
            now = datetime.now().astimezone().isoformat(timespec="seconds")
            existing_idx = next((i for i, e in enumerate(self._entries) if e.root == key), None)
            entry = SavedWorkspace(root=key, name=workspace.name, last_used=now)
            if existing_idx is not None:
                self._entries.pop(existing_idx)
            self._entries.insert(0, entry)
            # Cap to last 20 so the JSON stays small.
            del self._entries[20:]
            try:
                self._save()
            except OSError as e:
                logger.warning("WorkspaceRegistry save failed: %s", e)

    def remove(self, root: str) -> None:
        with self._lock:
            self._entries = [e for e in self._entries if e.root != root]
            try:
                self._save()
            except OSError as e:
                logger.warning("WorkspaceRegistry save failed: %s", e)

    def list(self) -> list[SavedWorkspace]:
        with self._lock:
            return list(self._entries)


__all__ = [
    "Workspace",
    "WorkspaceError",
    "WorkspaceRegistry",
    "SavedWorkspace",
    "guess_language",
    "DEFAULT_DENY_GLOBS",
]
