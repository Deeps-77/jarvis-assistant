"""Read-only code-assistant tools (Phase 1).

Each tool is a LangChain ``@tool``-decorated callable that operates on a
module-level ``Workspace`` set by :func:`set_workspace` before the agent
runs. Validation always goes through ``Workspace.validate_path`` so a
misbehaving model cannot escape the project root or read sandbox artefacts.

Phase 1 surface (4 tools, all read-only):
- :func:`list_files`  recursive listing, optional glob
- :func:`read_file`   paginated file read with line numbers
- :func:`grep_files`  case-sensitive/insensitive search, optional regex
- :func:`get_file_info` size, mtime, language guess

Phase 2 will add the write/exec toolset behind an approval gate.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool

from code_assistant.workspace import Workspace, guess_language

logger = logging.getLogger(__name__)

_WORKSPACE: Workspace | None = None


def set_workspace(workspace: Workspace | None) -> None:
    """Inject the active workspace. Called by ``CodeBrain`` before each run."""
    global _WORKSPACE
    _WORKSPACE = workspace


def get_workspace() -> Workspace:
    if _WORKSPACE is None:
        raise RuntimeError("No workspace configured. Call set_workspace(ws) first.")
    return _WORKSPACE


def _config_workspace(config: RunnableConfig | None) -> Workspace:
    """Prefer an injected workspace (set via set_workspace) over config."""
    if _WORKSPACE is not None:
        return _WORKSPACE
    # Fallback for direct tool invocation in tests / scripts.
    cfg = (config or {}).get("configurable") or {}
    root = cfg.get("workspace_root")
    if root:
        return Workspace(Path(root))
    raise RuntimeError("No workspace in tool context.")


# ------------------------------------------------------------ list_files


@tool
def list_files(
    path: str = "",
    pattern: str | None = None,
    max_depth: int = 8,
    max_entries: int = 500,
) -> str:
    """List files in the workspace.

    Args:
        path: Optional subdirectory relative to the workspace root. Empty = root.
        pattern: Optional glob filter against the basename (e.g. "*.py", "test_*").
        max_depth: Cap recursion depth (default 8). 0 = no recursion.
        max_entries: Hard cap on returned entries (default 500) to keep the
            prompt bounded for very large repos.

    Returns:
        A newline-separated list of relative paths, or an explanatory error.
    """
    ws = _config_workspace(None)
    try:
        start = ws.validate_path(path, must_exist=False) if path else ws.root
    except Exception as e:
        return f"ERROR: {e}"
    if start.is_file():
        return start.relative_to(ws.root).as_posix()

    results: list[str] = []
    truncated = False
    stack: list[tuple[Path, int]] = [(start, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            entries = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except (PermissionError, OSError) as e:
            return f"ERROR: cannot list {current}: {e}"
        for entry in entries:
            rel = entry.relative_to(ws.root)
            if any(part.startswith(".") for part in rel.parts):
                continue
            if ws.is_denied(rel):
                continue
            if pattern and not fnmatch(entry.name, pattern):
                continue
            if entry.is_dir():
                if max_depth <= 0 or depth + 1 <= max_depth:
                    stack.append((entry, depth + 1))
            elif entry.is_file():
                results.append(rel.as_posix())
                if len(results) >= max_entries:
                    truncated = True
                    break
        if truncated:
            break
    if not results:
        return "(no matching files)"
    suffix = "\n... (truncated)" if truncated else ""
    return "\n".join(results) + suffix


# ------------------------------------------------------------ read_file


@tool
def read_file(
    path: str,
    start_line: int = 0,
    end_line: int = -1,
    max_bytes: int = 200_000,
) -> str:
    """Read a file from the workspace.

    Args:
        path: Relative path to the file.
        start_line: 0-indexed first line to include (default 0 = from start).
        end_line: 0-indexed last line to include (default -1 = until end).
        max_bytes: Hard cap on returned text (default 200 KB). Larger files
            must be read in chunks via start_line/end_line.

    Returns:
        The file slice with ``<line>: <content>`` markers, or an error.
    """
    ws = _config_workspace(None)
    try:
        target = ws.validate_path(path, must_exist=True)
    except Exception as e:
        return f"ERROR: {e}"
    if not target.is_file():
        return f"ERROR: not a file: {path}"
    size = target.stat().st_size
    if size > ws.max_file_size_bytes:
        return (
            f"ERROR: file is {size:,} bytes which exceeds the workspace cap "
            f"({ws.max_file_size_bytes:,}). Read it in chunks via start_line/end_line."
        )
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return f"ERROR: cannot read {path}: {e}"
    lines = text.splitlines()
    total = len(lines)
    if start_line < 0:
        start_line = 0
    if end_line < 0 or end_line >= total:
        end_line = total - 1
    if start_line > end_line:
        return f"ERROR: start_line ({start_line}) > end_line ({end_line})"
    slice_ = lines[start_line : end_line + 1]
    # Annotate lines and bound by bytes.
    width = len(str(end_line + 1))
    out: list[str] = []
    used = 0
    truncated = False
    for i, line in enumerate(slice_, start=start_line):
        rendered = f"{str(i + 1).rjust(width)} | {line}"
        encoded = rendered.encode("utf-8")
        if used + len(encoded) > max_bytes:
            truncated = True
            break
        out.append(rendered)
        used += len(encoded) + 1
    header = f"# {path}  (lines {start_line + 1}..{end_line + 1} of {total})"
    suffix = (
        f"\n... (truncated at {max_bytes:,} bytes; use start_line/end_line to read more)"
        if truncated
        else ""
    )
    return header + "\n" + "\n".join(out) + suffix


# ------------------------------------------------------------ grep_files


@tool
def grep_files(
    query: str,
    file_pattern: str | None = None,
    case_sensitive: bool = False,
    max_results: int = 80,
    context_lines: int = 0,
) -> str:
    """Search for ``query`` across files in the workspace.

    Args:
        query: Regex pattern (Python ``re`` syntax).
        file_pattern: Optional glob filter on basename, e.g. "*.py".
        case_sensitive: Default False.
        max_results: Cap on matching lines (default 80).
        context_lines: Number of surrounding lines to include around each match.

    Returns:
        ``<path>:<line>:<content>`` lines, grouped by file. Empty message if
        no matches. Falls back to ``rg`` when available for speed.
    """
    ws = _config_workspace(None)
    try:
        pattern = re.compile(query if case_sensitive else f"(?i){query}")
    except re.error as e:
        return f"ERROR: invalid regex {query!r}: {e}"

    rg = shutil.which("rg")
    if rg:
        return _grep_with_ripgrep(ws, rg, query, file_pattern, case_sensitive, max_results, context_lines)

    results: list[str] = []
    truncated = False
    for path in ws.iter_files():
        if file_pattern and not fnmatch(path.name, file_pattern):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                rel = path.relative_to(ws.root).as_posix()
                if context_lines:
                    results.append(f"{rel}:{i}:{line}")
                else:
                    results.append(f"{rel}:{i}:{line}")
                if len(results) >= max_results:
                    truncated = True
                    break
        if truncated:
            break
    if not results:
        return "(no matches)"
    suffix = "\n... (truncated)" if truncated else ""
    return "\n".join(results) + suffix


def _grep_with_ripgrep(
    ws: Workspace,
    rg: str,
    query: str,
    file_pattern: str | None,
    case_sensitive: bool,
    max_results: int,
    context_lines: int,
) -> str:
    cmd = [
        rg,
        "--no-heading",
        "--line-number",
        "--color=never",
        f"--max-count={max_results}",
    ]
    if not case_sensitive:
        cmd.append("-i")
    if context_lines:
        cmd.append(f"-C{context_lines}")
    if file_pattern:
        cmd.extend(["--glob", file_pattern])
    # Add explicit deny-glob --type-add filters via --iglob.
    for g in ws.deny_globs:
        if g.endswith("/"):
            cmd.extend(["--glob", f"!{g[:-1]}"])
        else:
            cmd.extend(["--glob", f"!{g}"])
    cmd.extend([query, "."])
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ws.root),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return "ERROR: ripgrep timed out"
    except OSError as e:
        return f"ERROR: ripgrep failed: {e}"
    if proc.returncode not in (0, 1):  # 1 = no matches
        return f"ERROR: ripgrep: {proc.stderr.strip()[:300]}"
    out = proc.stdout.strip()
    if not out:
        return "(no matches)"
    lines = out.splitlines()
    if len(lines) > max_results:
        lines = lines[:max_results]
        out = "\n".join(lines) + "\n... (truncated)"
    return out


# ------------------------------------------------------------ get_file_info


@tool
def get_file_info(path: str) -> str:
    """Return size, modification time, and language guess for a file."""
    ws = _config_workspace(None)
    try:
        target = ws.validate_path(path, must_exist=True)
    except Exception as e:
        return f"ERROR: {e}"
    if not target.is_file():
        return f"ERROR: not a file: {path}"
    stat = target.stat()
    lang = guess_language(target)
    mtime = datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds")
    return (
        f"path: {target.relative_to(ws.root).as_posix()}\n"
        f"size_bytes: {stat.st_size:,}\n"
        f"modified: {mtime}\n"
        f"language: {lang}\n"
        f"permissions: {oct(stat.st_mode & 0o777)}"
    )


# ------------------------------------------------------------ registry

ALL_TOOLS = [list_files, read_file, grep_files, get_file_info]
ALL_TOOL_NAMES = frozenset(t.name for t in ALL_TOOLS)


def tool_names() -> list[str]:
    return [t.name for t in ALL_TOOLS]


__all__ = [
    "ALL_TOOLS",
    "ALL_TOOL_NAMES",
    "set_workspace",
    "get_workspace",
    "tool_names",
    "list_files",
    "read_file",
    "grep_files",
    "get_file_info",
]
