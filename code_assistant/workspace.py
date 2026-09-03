"""Workspace management for the Jarvis Code Assistant.

A *Workspace* is the root folder the code brain operates inside. All file
paths exposed to the agent are validated to lie within this root before any
I/O operation is attempted.

Key design goals
----------------
- **Confinement**: ``validate_path`` refuses ``..`` traversal, symlinks that
  escape the root, and any path matching a deny-glob. Every tool function in
  ``code_assistant.tools`` calls this before touching the filesystem.
- **Sandbox dir**: ``<root>/.jarvis-sandbox/`` is created on ``Workspace``
  construction and added to ``.gitignore`` automatically. The sandbox dir is
  used by Phase 2 (write tools + run_in_sandbox), but creating it here keeps
  workspace setup atomic.
- **Registry**: ``WorkspaceRegistry`` persists named workspaces across UI
  sessions to a ``code_workspaces.json`` file next to the project root, so
  users can switch between projects without re-typing paths.
"""

from __future__ import annotations

import fnmatch
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────── deny-glob defaults ──────────────────────────

DEFAULT_DENY_GLOBS: tuple[str, ...] = (
    ".git",
    ".git/*",
    "node_modules",
    "node_modules/*",
    ".venv",
    ".venv/*",
    "__pycache__",
    "__pycache__/*",
    ".jarvis-sandbox",
    ".jarvis-sandbox/*",
    "dist",
    "dist/*",
    "build",
    "build/*",
    ".idea",
    ".idea/*",
    ".vscode",
    ".vscode/*",
    "*.pyc",
    "*.pyo",
    ".DS_Store",
)


def _matches_deny(rel: str, deny_globs: tuple[str, ...]) -> bool:
    """Return True if *rel* (relative POSIX path) matches any deny glob."""
    parts = Path(rel).parts
    for glob in deny_globs:
        # Match against the full relative path
        if fnmatch.fnmatch(rel, glob):
            return True
        # Also match against each individual path component so that
        # ``node_modules`` blocks ``node_modules/lodash/index.js`` too.
        for part in parts:
            if fnmatch.fnmatch(part, glob):
                return True
    return False


# ─────────────────────────────────── Workspace ───────────────────────────────


@dataclass
class Workspace:
    """A validated, confinement-enforced view of a directory on disk.

    Parameters
    ----------
    root:
        Absolute path to the workspace root. Resolved to an absolute path on
        construction; a ``ValueError`` is raised if the directory does not exist.
    name:
        Human-readable label used in the UI sidebar and registry.
    deny_globs:
        Iterable of glob patterns (relative to root) that are hidden from the
        agent. Defaults to ``DEFAULT_DENY_GLOBS``.
    """

    root: Path
    name: str
    deny_globs: tuple[str, ...] = field(default_factory=lambda: DEFAULT_DENY_GLOBS)

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()
        if not self.root.exists():
            raise ValueError(f"Workspace root does not exist: {self.root}")
        if not self.root.is_dir():
            raise ValueError(f"Workspace root is not a directory: {self.root}")
        if isinstance(self.deny_globs, list):
            self.deny_globs = tuple(self.deny_globs)
        self._ensure_sandbox()

    # ----------------------------------------------------------------- paths

    def validate_path(self, rel: str) -> Path:
        """Resolve *rel* within the workspace root and validate it.

        Raises ``PermissionError`` for:
        - ``..`` traversal or any path that resolves outside the root
        - Symlinks that point outside the root
        - Paths matching any deny-glob

        Returns the resolved absolute ``Path`` on success.
        """
        # Normalise: strip leading slashes/backslashes so callers can pass
        # "foo/bar" or "/foo/bar" equally.
        clean = rel.lstrip("/\\").replace("\\", "/")
        if not clean or clean == ".":
            return self.root

        candidate = (self.root / clean).resolve()

        # Confinement check — must be inside root
        try:
            candidate.relative_to(self.root)
        except ValueError:
            raise PermissionError(
                f"Path {rel!r} resolves outside workspace root {self.root}"
            )

        # Deny-glob check (use POSIX relative path for matching)
        try:
            posix_rel = candidate.relative_to(self.root).as_posix()
        except ValueError:
            posix_rel = clean

        if _matches_deny(posix_rel, self.deny_globs):
            raise PermissionError(
                f"Path {rel!r} matches a deny-glob and is not accessible"
            )

        return candidate

    def rel(self, abs_path: Path) -> str:
        """Return *abs_path* relative to the workspace root as a POSIX string."""
        try:
            return abs_path.relative_to(self.root).as_posix()
        except ValueError:
            return str(abs_path)

    # --------------------------------------------------------------- sandbox

    def _ensure_sandbox(self) -> None:
        """Create ``.jarvis-sandbox/`` and add it to ``.gitignore``."""
        sandbox = self.root / ".jarvis-sandbox"
        try:
            sandbox.mkdir(exist_ok=True)
            (sandbox / "tmp").mkdir(exist_ok=True)
            (sandbox / "trash").mkdir(exist_ok=True)
        except OSError as e:
            logger.warning("Could not create sandbox dir %s: %s", sandbox, e)
            return

        gitignore = self.root / ".gitignore"
        marker = ".jarvis-sandbox/"
        try:
            existing = gitignore.read_text(encoding="utf-8") if gitignore.exists() else ""
            if marker not in existing:
                with gitignore.open("a", encoding="utf-8") as fh:
                    if existing and not existing.endswith("\n"):
                        fh.write("\n")
                    fh.write(f"# Jarvis code-assistant sandbox\n{marker}\n")
                logger.info("Added %s to %s", marker, gitignore)
        except OSError as e:
            logger.warning("Could not update .gitignore: %s", e)

    # ------------------------------------------------------------------ repr

    def describe(self) -> str:
        """Short label for the UI sidebar."""
        return f"{self.name} ({self.root})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "root": str(self.root),
            "deny_globs": list(self.deny_globs),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Workspace":
        return cls(
            root=Path(data["root"]),
            name=data.get("name", Path(data["root"]).name),
            deny_globs=tuple(data.get("deny_globs", DEFAULT_DENY_GLOBS)),
        )


# ─────────────────────────────── WorkspaceRegistry ───────────────────────────

_REGISTRY_FILE = Path(__file__).parent.parent / "code_workspaces.json"


class WorkspaceRegistry:
    """Persist a list of named workspaces across UI sessions.

    Backed by ``code_workspaces.json`` in the project root (next to
    ``llm_provider.py``).  Thread-safe via a simple load-modify-save cycle;
    no concurrent writers expected in a single-user setup.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path) if path else _REGISTRY_FILE
        self._workspaces: dict[str, dict[str, Any]] = {}
        self._load()

    # ------------------------------------------------------------------ CRUD

    def add(self, ws: Workspace) -> None:
        """Save or overwrite a workspace entry keyed by name."""
        self._workspaces[ws.name] = ws.to_dict()
        self._save()

    def remove(self, name: str) -> bool:
        if name not in self._workspaces:
            return False
        del self._workspaces[name]
        self._save()
        return True

    def get(self, name: str) -> Workspace | None:
        data = self._workspaces.get(name)
        if not data:
            return None
        try:
            return Workspace.from_dict(data)
        except (ValueError, KeyError) as e:
            logger.warning("Could not load workspace %r: %s", name, e)
            return None

    def list_all(self) -> list[dict[str, Any]]:
        """Return all saved workspace dicts (for UI display)."""
        return list(self._workspaces.values())

    def names(self) -> list[str]:
        return list(self._workspaces.keys())

    # ---------------------------------------------------------- persistence

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._workspaces = raw if isinstance(raw, dict) else {}
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("WorkspaceRegistry load error from %s: %s", self._path, e)

    def _save(self) -> None:
        try:
            self._path.write_text(
                json.dumps(self._workspaces, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning("WorkspaceRegistry save error to %s: %s", self._path, e)


# ─────────────────────────────────── helpers ─────────────────────────────────


def count_workspace_contents(ws: Workspace) -> dict[str, int]:
    """Return rough file/dir counts (used in the system prompt preamble)."""
    files = dirs = 0
    try:
        for entry in ws.root.rglob("*"):
            rel = entry.relative_to(ws.root).as_posix()
            if _matches_deny(rel, ws.deny_globs):
                continue
            if entry.is_dir():
                dirs += 1
            else:
                files += 1
            if files + dirs > 50_000:
                break  # cap to avoid hanging on huge repos
    except OSError:
        pass
    return {"files": files, "dirs": dirs}


__all__ = [
    "Workspace",
    "WorkspaceRegistry",
    "DEFAULT_DENY_GLOBS",
    "count_workspace_contents",
]
