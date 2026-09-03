"""Read-only file-system tools for the Jarvis Code Assistant (Phase 1).

All tools in this module are *read-only* — they never write to the filesystem.
Phase 2 will add mutating tools (``write_file``, ``edit_file``, ``apply_patch``,
``delete_path``) in a separate section.

Each tool is a plain Python function decorated with ``@tool`` from LangChain.
They close over a ``Workspace`` instance that is injected at startup by
``CodeBrain``.  The closure pattern keeps the agent function signatures simple
(no extra arguments the LLM has to invent) while still passing the workspace
context through at call time.

Read-only toolset
-----------------
- ``list_files``   — directory listing, filtered by deny-globs
- ``read_file``    — paginated file read with optional line range
- ``grep_files``   — full-text / regex search via ripgrep or Python ``re``
- ``get_file_info`` — size, mtime, language guess, line count
"""

from __future__ import annotations

import logging
import mimetypes
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from langchain_core.tools import tool

from code_assistant.workspace import Workspace, _matches_deny

logger = logging.getLogger(__name__)

# Maximum bytes returned by read_file in one call (prevents context overflow).
_MAX_READ_BYTES = 80_000
# Maximum number of grep matches returned per call.
_MAX_GREP_RESULTS = 100
# Maximum number of directory entries returned by list_files.
_MAX_LIST_ENTRIES = 500

# ────────────────────────── language / extension map ─────────────────────────

_EXT_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "jsx",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".swift": "swift",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".sh": "bash",
    ".bash": "bash",
    ".zsh": "zsh",
    ".ps1": "powershell",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".md": "markdown",
    ".rst": "rst",
    ".html": "html",
    ".htm": "html",
    ".css": "css",
    ".scss": "scss",
    ".sql": "sql",
    ".tf": "terraform",
    ".dockerfile": "dockerfile",
    ".xml": "xml",
    ".ini": "ini",
    ".cfg": "ini",
    ".env": "env",
}


def _guess_language(path: Path) -> str:
    suffix = path.suffix.lower()
    lang = _EXT_LANG.get(suffix, "")
    if not lang:
        name = path.name.lower()
        if name in ("dockerfile", "containerfile"):
            return "dockerfile"
        if name.startswith("makefile"):
            return "makefile"
    return lang or "text"


# ──────────────────────────── factory function ───────────────────────────────


def make_read_tools(workspace: Workspace) -> list[Any]:
    """Return a list of LangChain tool objects bound to *workspace*.

    Call this once per ``CodeBrain`` instance. The returned tools are passed
    directly to ``create_react_agent``.
    """

    # ── list_files ────────────────────────────────────────────────────────────

    @tool
    def list_files(
        path: str = "",
        pattern: str = "",
        max_depth: int = 3,
    ) -> str:
        """List files and directories inside the workspace.

        Parameters
        ----------
        path:
            Sub-path within the workspace to list. Empty string means the root.
        pattern:
            Optional glob pattern to filter results (e.g. ``*.py``).
        max_depth:
            How many levels deep to recurse (1 = immediate children only).
            Capped at 6.
        """
        try:
            base = workspace.validate_path(path)
        except PermissionError as e:
            return f"Error: {e}"

        if not base.is_dir():
            return f"Error: '{path}' is not a directory."

        max_depth = min(max(1, max_depth), 6)
        entries: list[str] = []
        root_str = str(workspace.root)

        def _walk(p: Path, depth: int) -> None:
            if len(entries) >= _MAX_LIST_ENTRIES:
                return
            try:
                items = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
            except PermissionError:
                return
            for item in items:
                try:
                    rel = item.relative_to(workspace.root).as_posix()
                except ValueError:
                    continue
                if _matches_deny(rel, workspace.deny_globs):
                    continue
                if pattern and not item.match(pattern):
                    if not (item.is_dir()):
                        continue
                indent = "  " * (depth - 1)
                kind = "/" if item.is_dir() else ""
                size_str = ""
                if item.is_file():
                    try:
                        sz = item.stat().st_size
                        size_str = f"  ({_fmt_size(sz)})"
                    except OSError:
                        pass
                entries.append(f"{indent}{item.name}{kind}{size_str}")
                if item.is_dir() and depth < max_depth:
                    _walk(item, depth + 1)

        _walk(base, 1)

        if not entries:
            return "Directory is empty (or all contents are excluded by deny-globs)."

        header = f"Listing: {workspace.rel(base) or '.'}\n"
        body = "\n".join(entries)
        suffix = f"\n\n(Showing {len(entries)} entries" + (
            f"; truncated at {_MAX_LIST_ENTRIES}" if len(entries) >= _MAX_LIST_ENTRIES else ""
        ) + ")"
        return header + body + suffix

    # ── read_file ─────────────────────────────────────────────────────────────

    @tool
    def read_file(
        path: str,
        start_line: int = 1,
        end_line: int = -1,
    ) -> str:
        """Read the contents of a file, optionally restricted to a line range.

        Parameters
        ----------
        path:
            File path relative to workspace root.
        start_line:
            First line to return (1-indexed). Defaults to 1.
        end_line:
            Last line to return, inclusive (1-indexed). ``-1`` means EOF.
            The returned slice is capped at ~80 KB to protect the context window.
        """
        try:
            abs_path = workspace.validate_path(path)
        except PermissionError as e:
            return f"Error: {e}"

        if not abs_path.exists():
            return f"Error: '{path}' does not exist in the workspace."
        if abs_path.is_dir():
            return f"Error: '{path}' is a directory. Use list_files to browse."

        try:
            raw = abs_path.read_bytes()
        except OSError as e:
            return f"Error reading '{path}': {e}"

        # Detect binary files
        if b"\x00" in raw[:8192]:
            size = len(raw)
            return (
                f"'{path}' appears to be a binary file ({_fmt_size(size)}). "
                "Use get_file_info to inspect metadata."
            )

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.decode("latin-1")
            except UnicodeDecodeError:
                return f"Error: '{path}' could not be decoded as text."

        lines = text.splitlines(keepends=True)
        total_lines = len(lines)

        # Normalise range
        s = max(1, start_line) - 1  # convert to 0-indexed
        e = total_lines if end_line == -1 else min(end_line, total_lines)

        slice_lines = lines[s:e]

        # Cap output size
        content = "".join(slice_lines)
        if len(content.encode()) > _MAX_READ_BYTES:
            content = content.encode()[:_MAX_READ_BYTES].decode(errors="replace")
            truncated = True
        else:
            truncated = False

        lang = _guess_language(abs_path)
        header = (
            f"File: {workspace.rel(abs_path)}\n"
            f"Lines: {s + 1}-{min(e, s + len(slice_lines))} of {total_lines} | "
            f"Language: {lang}\n"
            f"{'─' * 60}\n"
        )
        footer = "\n[TRUNCATED — call read_file again with start_line to continue]" if truncated else ""
        return header + content + footer

    # ── grep_files ────────────────────────────────────────────────────────────

    @tool
    def grep_files(
        query: str,
        regex: bool = False,
        file_pattern: str = "",
        case_sensitive: bool = False,
        max_results: int = 50,
    ) -> str:
        """Search for text across all files in the workspace.

        Tries ripgrep (``rg``) first for speed; falls back to Python ``re``.

        Parameters
        ----------
        query:
            The search string or regex pattern.
        regex:
            If ``True``, treat *query* as a regex. Otherwise, exact literal match.
        file_pattern:
            Optional glob to restrict which files are searched (e.g. ``*.py``).
        case_sensitive:
            Whether the search is case-sensitive. Defaults to ``False``.
        max_results:
            Cap on matches returned (to protect context). Max 100.
        """
        if not query:
            return "Error: query cannot be empty."

        max_results = min(max(1, max_results), _MAX_GREP_RESULTS)

        rg_path = _find_rg()
        if rg_path:
            return _grep_rg(
                rg_path, query, regex, file_pattern, case_sensitive, max_results, workspace
            )
        return _grep_python(query, regex, file_pattern, case_sensitive, max_results, workspace)

    # ── get_file_info ─────────────────────────────────────────────────────────

    @tool
    def get_file_info(path: str) -> str:
        """Return metadata about a file or directory.

        Returns size, modification time, language guess, and line count (for
        text files under 10 MB). Use this before read_file to avoid reading
        huge files blindly.

        Parameters
        ----------
        path:
            File or directory path relative to workspace root.
        """
        try:
            abs_path = workspace.validate_path(path)
        except PermissionError as e:
            return f"Error: {e}"

        if not abs_path.exists():
            return f"Error: '{path}' does not exist in the workspace."

        try:
            stat = abs_path.stat()
        except OSError as e:
            return f"Error: {e}"

        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stat.st_mtime))
        rel = workspace.rel(abs_path)

        if abs_path.is_dir():
            try:
                children = sum(1 for _ in abs_path.iterdir())
            except OSError:
                children = -1
            return (
                f"Path:     {rel}\n"
                f"Type:     directory\n"
                f"Modified: {mtime}\n"
                f"Children: {children} immediate entries\n"
            )

        size = stat.st_size
        lang = _guess_language(abs_path)
        lines_info = ""

        if size < 10 * 1024 * 1024:  # < 10 MB
            try:
                raw = abs_path.read_bytes()
                if b"\x00" not in raw[:8192]:
                    try:
                        text = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        text = raw.decode("latin-1", errors="replace")
                    lines_info = f"Lines:    {text.count(chr(10)) + (1 if text and not text.endswith(chr(10)) else 0)}\n"
            except OSError:
                pass

        return (
            f"Path:     {rel}\n"
            f"Type:     file\n"
            f"Size:     {_fmt_size(size)} ({size} bytes)\n"
            f"Modified: {mtime}\n"
            f"Language: {lang or 'unknown'}\n"
            + lines_info
        )

    return [list_files, read_file, grep_files, get_file_info]


# ────────────────────────────── grep helpers ─────────────────────────────────


def _find_rg() -> str | None:
    """Return the path to ripgrep if available on PATH."""
    for candidate in ("rg", "rg.exe"):
        try:
            result = subprocess.run(
                [candidate, "--version"],
                capture_output=True,
                timeout=2,
            )
            if result.returncode == 0:
                return candidate
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return None


def _grep_rg(
    rg: str,
    query: str,
    regex: bool,
    file_pattern: str,
    case_sensitive: bool,
    max_results: int,
    workspace: Workspace,
) -> str:
    cmd = [rg, "--line-number", "--no-heading", "--color=never"]
    if not case_sensitive:
        cmd.append("--ignore-case")
    if not regex:
        cmd.append("--fixed-strings")
    if file_pattern:
        cmd += ["--glob", file_pattern]
    # Add deny-glob exclusions
    for g in workspace.deny_globs:
        cmd += ["--glob", f"!{g}"]
    cmd += [f"--max-count={max_results}", query, str(workspace.root)]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=15,
            cwd=str(workspace.root),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return f"ripgrep error: {e}"

    if result.returncode not in (0, 1):
        return f"ripgrep error (rc={result.returncode}): {result.stderr[:500]}"

    lines = result.stdout.strip().splitlines()[:max_results]
    if not lines:
        return f"No matches for {query!r}."

    # Convert absolute paths to workspace-relative paths
    out: list[str] = []
    for line in lines:
        try:
            # rg output: /abs/path/file.py:10:content
            colon_idx = line.index(":")
            abs_file = line[:colon_idx]
            rest = line[colon_idx:]
            rel = workspace.rel(Path(abs_file))
            out.append(rel + rest)
        except (ValueError, IndexError):
            out.append(line)

    header = f"grep ({len(out)} match{'es' if len(out) != 1 else ''}): {query!r}\n{'─' * 60}\n"
    return header + "\n".join(out)


def _grep_python(
    query: str,
    regex: bool,
    file_pattern: str,
    case_sensitive: bool,
    max_results: int,
    workspace: Workspace,
) -> str:
    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        pattern = re.compile(query if regex else re.escape(query), flags)
    except re.error as e:
        return f"Invalid regex: {e}"

    matches: list[str] = []

    for abs_path in workspace.root.rglob("*"):
        if len(matches) >= max_results:
            break
        try:
            rel = abs_path.relative_to(workspace.root).as_posix()
        except ValueError:
            continue
        if _matches_deny(rel, workspace.deny_globs):
            continue
        if not abs_path.is_file():
            continue
        if file_pattern and not abs_path.match(file_pattern):
            continue
        if abs_path.stat().st_size > 5 * 1024 * 1024:
            continue  # skip files > 5 MB

        try:
            raw = abs_path.read_bytes()
            if b"\x00" in raw[:4096]:
                continue  # skip binary
            text = raw.decode("utf-8", errors="replace")
        except OSError:
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            if len(matches) >= max_results:
                break
            if pattern.search(line):
                matches.append(f"{rel}:{lineno}:{line.rstrip()}")

    if not matches:
        return f"No matches for {query!r}."

    header = f"grep ({len(matches)} match{'es' if len(matches) != 1 else ''}): {query!r}\n{'─' * 60}\n"
    return header + "\n".join(matches)


# ────────────────────────────── size formatting ───────────────────────────────


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"


__all__ = ["make_read_tools", "_guess_language", "_fmt_size"]
