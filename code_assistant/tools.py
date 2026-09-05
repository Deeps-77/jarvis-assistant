"""Code-assistant tools (Phase 2: read-only + write/exec behind approval).

Each tool is a LangChain ``@tool``-decorated callable that operates on a
module-level ``Workspace`` set by :func:`set_workspace` before the agent
runs. Validation always goes through ``Workspace.validate_path`` so a
misbehaving model cannot escape the project root or read sandbox artefacts.

Read-only toolset (Phase 1, always available):
- :func:`list_files`    recursive listing, optional glob
- :func:`read_file`     paginated file read with line numbers
- :func:`grep_files`    regex search across files
- :func:`get_file_info` size, mtime, language guess

Write/exec toolset (Phase 2, BUILD mode only, every call approval-gated):
- :func:`write_file`    atomic file write with diff preview
- :func:`edit_file`     find/replace with diff preview
- :func:`apply_patch`   apply a unified diff
- :func:`mkdir`         recursive directory creation
- :func:`delete_path`   soft-delete to .jarvis-sandbox/trash
- :func:`run_command`   subprocess in sandbox tmp dir
- :func:`copy_out`      move an artefact from sandbox to workspace

The toolset is split by :data:`code_assistant.modes.TOOLSETS` so the Plan
mode never sees write tools — that's the first line of defence. The second
line is the mode check inside each write tool: even if a tool sneaks in,
it refuses to run unless the brain is in BUILD mode.
"""

from __future__ import annotations

import difflib
import hashlib
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from datetime import datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import InjectedToolArg, tool

from code_assistant.sandbox import (
    DEFAULT_MAX_OUTPUT_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    copy_out as _sandbox_copy_out,
    run_in_sandbox,
)
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


# ------------------------------------------------------- mode + approval gate


class ModeError(PermissionError):
    """Raised when a write tool is invoked outside BUILD mode."""


# Tools whose names appear here trigger the approval gate. The brain reads
# this set before yielding a tool_start event for one of them.
REQUIRES_APPROVAL: set[str] = {
    "write_file",
    "edit_file",
    "apply_patch",
    "mkdir",
    "delete_path",
    "run_command",
    "copy_out",
}

# Module-level "current mode". The brain calls ``set_current_mode`` before
# each run so a stray write call outside BUILD mode raises ModeError.
_CURRENT_MODE: str = "plan"


def set_current_mode(mode: str) -> None:
    global _CURRENT_MODE
    _CURRENT_MODE = (mode or "plan").strip().lower()


def get_current_mode() -> str:
    return _CURRENT_MODE


def _require_build_mode() -> str | None:
    """Refuse write/exec calls outside BUILD mode.

    Returns an error string suitable for tool-return if the mode is wrong,
    or ``None`` when the call is allowed. Tools should ``return _require_build_mode()``
    as their first line — that way the model sees a normal string error
    instead of an exception that LangChain can't render cleanly.
    """
    if _CURRENT_MODE != "build":
        return (
            f"ERROR: Write/exec tools require BUILD mode (current: "
            f"{_CURRENT_MODE}). Switch to BUILD with `/mode build` in the "
            f"code UI before calling {globals().get('_caller_name', 'this tool')}."
        )
    return None


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


# ============================================================ PHASE 2 TOOLS
#
# Write/exec tools. Each one:
#   1. Validates the path (Workspace.validate_path) so traversal / absolute
#      / denied paths still raise.
#   2. Calls _require_build_mode() — refuses to run if the brain is in
#      PLAN mode even if the model somehow ended up with the tool.
#   3. Returns a structured string the brain / UI can parse.
#
# Every tool here is registered in REQUIRES_APPROVAL, so the brain yields an
# approval_required BrainEvent BEFORE it lets the tool run. The UI shows a
# diff card; the user clicks Approve / Reject / Edit & retry.


# ------------------------------------------------------- atomic write helper


def _atomic_write(target: Path, content: str, workspace: Workspace) -> None:
    """Write ``content`` to ``target`` via temp file + ``os.replace``.

    The temp file lives under ``.jarvis-sandbox/pending/`` so the write is
    atomic from the perspective of readers (an incomplete file is never
    visible at ``target``).
    """
    workspace.ensure_sandbox()
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
    tmp_name = f"{int(time.time() * 1000)}_{digest}_{uuid.uuid4().hex[:6]}.partial"
    tmp = workspace.sandbox_pending / tmp_name
    tmp.write_text(content, encoding="utf-8", newline="")
    try:
        os.replace(tmp, target)
    except OSError:
        # Clean up the temp on failure.
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _unified_diff(old: str, new: str, path: str) -> str:
    """Render a unified diff (suitable for the approval card)."""
    old_lines = old.splitlines(keepends=True) if old else []
    new_lines = new.splitlines(keepends=True) if new else []
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        n=3,
    )
    return "".join(diff)


# ------------------------------------------------------------ write_file


@tool
def write_file(path: str, content: str, overwrite: bool = False) -> str:
    """Write ``content`` to ``path`` (atomic). Returns a diff vs the previous version.

    Args:
        path: Relative path inside the workspace.
        content: New file contents.
        overwrite: Required to be True when the file already exists.

    Returns:
        A structured text result containing the diff, byte counts, and
        a marker ``APPLIED: <path>`` on success (the brain parses this to
        update the UI's "applied" log).
    """
    _err = _require_build_mode()

    if _err is not None:

        return _err
    ws = _config_workspace(None)
    try:
        target = ws.validate_path(path, must_exist=False)
    except Exception as e:
        return f"ERROR: {e}"

    existed = target.exists()
    if existed and not overwrite:
        return (
            f"ERROR: {path} already exists. Re-call with overwrite=True to "
            f"replace it, or use edit_file for a targeted change."
        )

    old = target.read_text(encoding="utf-8", errors="replace") if existed else ""
    diff = _unified_diff(old, content, path)
    try:
        _atomic_write(target, content, ws)
    except OSError as e:
        return f"ERROR: write failed: {e}"

    bytes_written = len(content.encode("utf-8"))
    action = "updated" if existed else "created"
    diff_block = diff if diff.strip() else "(no textual change)"
    return (
        f"APPLIED: {path}\n"
        f"action: {action}\n"
        f"bytes: {bytes_written:,}\n"
        f"--- diff ---\n{diff_block}\n--- end diff ---"
    )


# ------------------------------------------------------------ edit_file


@tool
def edit_file(
    path: str,
    find: str,
    replace: str,
    global_replace: bool = False,
) -> str:
    """Replace ``find`` with ``replace`` inside ``path``. Returns the resulting diff.

    Args:
        path: Relative path inside the workspace.
        find: Exact substring to locate. Must appear at least once.
        replace: Replacement text.
        global_replace: Replace every occurrence (default replaces only the
            first).

    Returns:
        A structured result with the diff and ``APPLIED: <path>`` on success.
    """
    _err = _require_build_mode()
    if _err is not None:
        return _err
    ws = _config_workspace(None)
    try:
        target = ws.validate_path(path, must_exist=True)
    except Exception as e:
        return f"ERROR: {e}"
    if not target.is_file():
        return f"ERROR: not a file: {path}"

    old = target.read_text(encoding="utf-8", errors="replace")
    occurrences = old.count(find)
    if occurrences == 0:
        return (
            f"ERROR: find text not found in {path}. Use read_file to confirm "
            f"the exact text (whitespace must match exactly)."
        )
    if occurrences > 1 and not global_replace:
        return (
            f"ERROR: {find[:60]!r} matches {occurrences} places in {path}. "
            f"Pass global_replace=True to replace all, or narrow find to a "
            f"unique snippet."
        )
    new = old.replace(find, replace) if global_replace else old.replace(find, replace, 1)
    diff = _unified_diff(old, new, path)
    try:
        _atomic_write(target, new, ws)
    except OSError as e:
        return f"ERROR: write failed: {e}"
    return (
        f"APPLIED: {path}\n"
        f"action: edited ({occurrences} occurrence{'s' if occurrences != 1 else ''})\n"
        f"--- diff ---\n{diff}\n--- end diff ---"
    )


# ------------------------------------------------------------ apply_patch


@tool
def apply_patch(path: str, unified_diff: str) -> str:
    """Apply a unified diff to ``path``.

    Args:
        path: Relative path inside the workspace.
        unified_diff: A standard unified diff (the kind ``diff -u`` produces).

    Returns:
        A structured result with ``APPLIED: <path>`` on success.

    Notes:
        Implemented manually rather than shelling out to ``patch`` so it
        works on Windows without extra binaries. Only handles single-file
        diffs (one ``---`` / ``+++`` header pair).
    """
    _err = _require_build_mode()

    if _err is not None:

        return _err
    ws = _config_workspace(None)
    try:
        target = ws.validate_path(path, must_exist=False)
    except Exception as e:
        return f"ERROR: {e}"

    old = target.read_text(encoding="utf-8", errors="replace") if target.exists() else ""
    new, parse_err = _apply_unified_diff_text(old, unified_diff, path)
    if parse_err:
        return f"ERROR: patch parse failed: {parse_err}"
    diff = _unified_diff(old, new, path)
    try:
        _atomic_write(target, new, ws)
    except OSError as e:
        return f"ERROR: write failed: {e}"
    return (
        f"APPLIED: {path}\n"
        f"action: patched\n"
        f"--- diff ---\n{diff}\n--- end diff ---"
    )


def _apply_unified_diff_text(old: str, patch: str, path: str) -> tuple[str, str | None]:
    """Tiny unified-diff applier. Returns (new_content, error_or_None)."""
    lines = patch.splitlines()
    # Find header pair
    src_header = None
    i = 0
    n = len(lines)
    while i < n:
        if lines[i].startswith("--- ") and i + 1 < n and lines[i + 1].startswith("+++ "):
            src_header = lines[i][4:].split("\t", 1)[0].strip()
            dst_header = lines[i + 1][4:].split("\t", 1)[0].strip()
            i += 2
            break
        i += 1
    if src_header is None:
        return old, "no --- /+++ header pair found"

    # Expect @@ hunk header
    if i >= n or not lines[i].startswith("@@"):
        return old, "no @@ hunk header found"

    old_lines = old.splitlines(keepends=True)
    if old and not old.endswith("\n"):
        old_lines.append("\n")
    out: list[str] = []
    cur_old = 0  # 1-indexed
    while i < n:
        line = lines[i]
        if not line.startswith("@@"):
            break
        try:
            header = line.split("@@")[1].strip()
            old_part = header.split(" ")[0].lstrip("-")
            old_start = int(old_part.split(",")[0])
        except (IndexError, ValueError):
            return old, f"malformed @@ header: {line!r}"
        # Copy untouched lines up to the hunk start
        target = old_start - 1  # 0-indexed
        while cur_old < target:
            if cur_old < len(old_lines):
                out.append(old_lines[cur_old])
            cur_old += 1
        i += 1
        # Process hunk body
        while i < n:
            l = lines[i]
            if l.startswith("@@") or (l.startswith("--- ") and i + 1 < n and lines[i + 1].startswith("+++ ")):
                break
            if l.startswith("+"):
                out.append(l[1:] + "\n")
            elif l.startswith("-"):
                cur_old += 1
            elif l.startswith(" "):
                if cur_old < len(old_lines):
                    out.append(old_lines[cur_old])
                cur_old += 1
            elif l == "\\ No newline at end of file":
                pass
            else:
                return old, f"unexpected patch line: {l!r}"
            i += 1
    # Copy remaining old lines
    while cur_old < len(old_lines):
        out.append(old_lines[cur_old])
        cur_old += 1
    return "".join(out), None


# ------------------------------------------------------------ mkdir


@tool
def mkdir(path: str) -> str:
    """Recursively create a directory at ``path`` (and any missing parents)."""
    _err = _require_build_mode()

    if _err is not None:

        return _err
    ws = _config_workspace(None)
    try:
        target = ws.validate_path(path, must_exist=False)
    except Exception as e:
        return f"ERROR: {e}"
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return f"ERROR: mkdir failed: {e}"
    return f"APPLIED: {path} (directory)"


# ------------------------------------------------------------ delete_path


@tool
def delete_path(path: str, confirm: bool = False) -> str:
    """Soft-delete ``path`` to ``.jarvis-sandbox/trash/<timestamp>_<name>``.

    Args:
        path: Relative path inside the workspace.
        confirm: Must be True to actually delete (defence in depth).
    """
    _err = _require_build_mode()

    if _err is not None:

        return _err
    ws = _config_workspace(None)
    if not confirm:
        return "ERROR: pass confirm=True to delete."
    try:
        target = ws.validate_path(path, must_exist=True)
    except Exception as e:
        return f"ERROR: {e}"
    if target == ws.root:
        return "ERROR: refusing to delete the workspace root."
    ws.ensure_sandbox()
    stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S")
    trash_dest = ws.sandbox_trash / f"{stamp}_{target.name}"
    try:
        shutil.move(str(target), str(trash_dest))
    except OSError as e:
        return f"ERROR: delete failed: {e}"
    rel = trash_dest.relative_to(ws.root).as_posix()
    return f"APPLIED: {path} (moved to {rel})"


# ------------------------------------------------------------ run_command


@tool
def run_command(
    command: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> str:
    """Run a single shell command inside the sandbox tmp directory.

    Args:
        command: One executable invocation. No shell metacharacters — use
            a single command with flags and quoted args, e.g.
            ``python -m pytest -q``. To run multiple commands, call this
            tool multiple times.
        timeout: Hard timeout in seconds (default 30).
        max_output_bytes: Per-stream output cap (default 200 KB).

    Returns:
        Rendered ``RunResult`` text — stdout/stderr, exit code, duration.
    """
    _err = _require_build_mode()
    if _err is not None:
        return _err
    ws = _config_workspace(None)
    try:
        result = run_in_sandbox(
            ws,
            command,
            cwd="sandbox_tmp",
            timeout=timeout,
            max_output_bytes=max_output_bytes,
        )
    except ValueError as e:
        return f"ERROR: {e}"
    except FileNotFoundError as e:
        return f"ERROR: executable not found: {e}"
    return result.to_text()


# ------------------------------------------------------------ copy_out


@tool
def copy_out(sandbox_relative_path: str, dest_relative_path: str = "") -> str:
    """Move a file produced in the sandbox tmp dir into the workspace.

    Args:
        sandbox_relative_path: Path under ``.jarvis-sandbox/tmp/``.
        dest_relative_path: Destination inside the workspace. Defaults to
            the same basename at the workspace root.

    Returns:
        ``APPLIED: <dest>`` on success.
    """
    _err = _require_build_mode()

    if _err is not None:

        return _err
    ws = _config_workspace(None)
    try:
        dest = _sandbox_copy_out(ws, sandbox_relative_path, dest_relative_path)
    except (ValueError, FileNotFoundError, FileExistsError) as e:
        return f"ERROR: {e}"
    return f"APPLIED: {dest}"


# ------------------------------------------------------------ registry

ALL_TOOLS = [
    list_files,
    read_file,
    grep_files,
    get_file_info,
    write_file,
    edit_file,
    apply_patch,
    mkdir,
    delete_path,
    run_command,
    copy_out,
]
ALL_TOOL_NAMES = frozenset(t.name for t in ALL_TOOLS)


def tool_names() -> list[str]:
    return [t.name for t in ALL_TOOLS]


__all__ = [
    "ALL_TOOLS",
    "ALL_TOOL_NAMES",
    "REQUIRES_APPROVAL",
    "ModeError",
    "set_current_mode",
    "get_current_mode",
    "set_workspace",
    "get_workspace",
    "tool_names",
    "list_files",
    "read_file",
    "grep_files",
    "get_file_info",
    "write_file",
    "edit_file",
    "apply_patch",
    "mkdir",
    "delete_path",
    "run_command",
    "copy_out",
]
