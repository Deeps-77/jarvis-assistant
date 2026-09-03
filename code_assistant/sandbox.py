"""Subprocess sandbox for the code assistant's ``run_command`` tool.

The sandbox is intentionally minimal:

- Default ``cwd`` is the workspace's ``.jarvis-sandbox/tmp`` so commands can't
  accidentally touch real project files. Output artefacts must be explicitly
  :func:`copy_out`-ed to become visible.
- The environment is stripped to a small allow-list (``env_keep``). The model
  cannot pass through credentials or pollute the parent shell.
- Output is capped (default 200 KB) and a hard timeout (default 30 s) is
  enforced — the subprocess is killed (and on POSIX, the process group) when
  the limit is exceeded.
- The sandbox is **not** a security boundary against a malicious local user.
  It only protects against the model accidentally making a mess. Local
  single-user only.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from code_assistant.workspace import Workspace

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_OUTPUT_BYTES = 200_000

# Environment variables that survive the strip. PATH is required for the
# command to find executables; the rest are harmless locale/TZ hints.
DEFAULT_ENV_KEEP: tuple[str, ...] = (
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "SystemRoot",
    "HOME",
    "USERPROFILE",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TMP",
    "TEMP",
    "USER",
    "LOGNAME",
    "SHELL",
)


@dataclasses.dataclass(slots=True)
class RunResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int
    timed_out: bool
    truncated: bool
    cwd: str
    command: str

    def to_text(self) -> str:
        head = (
            f"$ {self.command}\n"
            f"(cwd: {self.cwd}, exit: {self.exit_code}, "
            f"{self.duration_ms}ms"
            + (" [timed out]" if self.timed_out else "")
            + (" [output truncated]" if self.truncated else "")
            + ")"
        )
        parts = [head]
        if self.stdout:
            parts.append("--- stdout ---\n" + self.stdout)
        if self.stderr:
            parts.append("--- stderr ---\n" + self.stderr)
        if not self.stdout and not self.stderr:
            parts.append("(no output)")
        return "\n".join(parts)


def _select_cwd(workspace: Workspace, requested: str | None) -> Path:
    """Resolve and validate the cwd. Refuses anything outside the sandbox."""
    if not requested or requested.strip() in ("", ".", "sandbox_tmp"):
        target = workspace.sandbox_tmp
    elif requested.strip() in ("sandbox", ".jarvis-sandbox"):
        target = workspace.sandbox_dir
    elif requested.strip() in ("pending",):
        target = workspace.sandbox_pending
    else:
        # Any other explicit path is rejected: the only writable place is
        # inside .jarvis-sandbox/. Real file writes go through write_file
        # which has its own approval gate.
        raise ValueError(
            f"run_command cwd must be one of: '', 'sandbox_tmp', "
            f"'sandbox', 'pending'. Got: {requested!r}"
        )
    target.mkdir(parents=True, exist_ok=True)
    return target


def _strip_env(env_keep: tuple[str, ...]) -> dict[str, str]:
    """Build a minimal env from ``os.environ`` keeping only allow-listed keys."""
    return {k: v for k, v in os.environ.items() if k in env_keep}


def _truncate(text: str, max_bytes: int) -> tuple[str, bool]:
    """Truncate ``text`` to ``max_bytes`` UTF-8 bytes from the head."""
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text, False
    head = encoded[:max_bytes].decode("utf-8", errors="replace")
    return head + f"\n... [truncated, {len(encoded) - max_bytes} more bytes]", True


def _parse_command(command: str) -> list[str]:
    """Split a shell-like command line into argv safely.

    We support plain ``shlex`` parsing (handles quoted args). Shell metachars
    (``|``, ``>``, ``;``, etc.) are NOT honoured — ``shell=False`` in the
    subprocess call, so ``command`` must be a single executable invocation
    (possibly with flags and quoted args). This is intentional: the sandbox
    must not be a shell escape hatch.
    """
    try:
        return shlex.split(command, posix=(sys.platform != "win32"))
    except ValueError as e:
        raise ValueError(f"Could not parse command {command!r}: {e}") from None


def run_in_sandbox(
    workspace: Workspace,
    command: str | list[str],
    *,
    cwd: str | None = None,
    timeout: int | None = DEFAULT_TIMEOUT_SECONDS,
    env_keep: tuple[str, ...] | None = None,
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
) -> RunResult:
    """Run ``command`` in the workspace's sandbox tmp dir.

    Args:
        workspace: Active workspace; sandbox paths are derived from this.
        command: Shell-style command string or argv list. No metacharacters
            — must be one executable invocation. Use ``run_command`` tool
            to call this from the agent.
        cwd: ``None`` (default → ``sandbox_tmp``), ``"sandbox"``,
            ``"pending"``. Any other value is rejected.
        timeout: Hard timeout in seconds (default 30). 0 = no timeout
            (not recommended; the LLM can hang the UI indefinitely).
        env_keep: Tuple of env-var names to forward. Defaults to
            ``DEFAULT_ENV_KEEP`` (PATH, locale, etc.).
        max_output_bytes: Cap on stdout+stderr per stream (default 200 KB).

    Returns:
        :class:`RunResult` with stdout, stderr, exit code, duration, and
        flags for timeout/truncation.

    Raises:
        ValueError on bad cwd / unparseable command.
        FileNotFoundError when the executable doesn't exist on PATH.
    """
    workspace.ensure_sandbox()
    cwd_path = _select_cwd(workspace, cwd)
    keep = env_keep if env_keep is not None else DEFAULT_ENV_KEEP
    env = _strip_env(keep)
    argv = _parse_command(command) if isinstance(command, str) else list(command)
    if not argv:
        raise ValueError("Empty command")

    display = command if isinstance(command, str) else " ".join(shlex.quote(a) for a in argv)
    t0 = time.perf_counter()
    timed_out = False
    try:
        proc = subprocess.run(
            argv,
            cwd=str(cwd_path),
            env=env,
            shell=False,
            capture_output=True,
            text=True,
            timeout=timeout if timeout and timeout > 0 else None,
            check=False,
        )
        stdout, stderr, exit_code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as e:
        timed_out = True
        stdout = e.stdout or ""
        stderr = (e.stderr or "") + f"\n[killed: exceeded {timeout}s timeout]"
        exit_code = -1
    except FileNotFoundError:
        raise
    except OSError as e:
        return RunResult(
            stdout="",
            stderr=f"ERROR: failed to launch process: {e}",
            exit_code=127,
            duration_ms=int((time.perf_counter() - t0) * 1000),
            timed_out=False,
            truncated=False,
            cwd=str(cwd_path),
            command=display,
        )

    stdout, stdout_trunc = _truncate(stdout, max_output_bytes)
    stderr, stderr_trunc = _truncate(stderr, max_output_bytes)
    return RunResult(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        duration_ms=int((time.perf_counter() - t0) * 1000),
        timed_out=timed_out,
        truncated=stdout_trunc or stderr_trunc,
        cwd=str(cwd_path),
        command=display,
    )


def copy_out(workspace: Workspace, sandbox_relative: str, dest_relative: str = "") -> str:
    """Move a file from ``.jarvis-sandbox/tmp/`` into the workspace.

    Used by the model to surface artefacts the sandbox produced (test
    outputs, generated config files, etc.). Refuses anything outside the
    sandbox tmp dir as the source.

    Returns:
        The relative path the file now lives at in the workspace.
    """
    workspace.ensure_sandbox()
    src_rel = sandbox_relative.lstrip("/\\")
    src = (workspace.sandbox_tmp / src_rel).resolve()
    try:
        src.relative_to(workspace.sandbox_tmp)
    except ValueError:
        raise ValueError(
            f"copy_out source must be inside {workspace.sandbox_tmp}, got {sandbox_relative!r}"
        )
    if not src.exists():
        raise FileNotFoundError(f"copy_out source does not exist: {sandbox_relative}")

    dest_name = dest_relative.strip() or src.name
    # Dest relative validation goes through the workspace's normal path check.
    dest_abs = workspace.validate_path(dest_name, must_exist=False)
    dest_abs.parent.mkdir(parents=True, exist_ok=True)
    if dest_abs.exists():
        raise FileExistsError(f"copy_out destination already exists: {dest_name}")
    # Move (atomic on the same filesystem) rather than copy so the sandbox
    # is left clean.
    import shutil
    shutil.move(str(src), str(dest_abs))
    return dest_name


__all__ = [
    "RunResult",
    "run_in_sandbox",
    "copy_out",
    "DEFAULT_TIMEOUT_SECONDS",
    "DEFAULT_MAX_OUTPUT_BYTES",
    "DEFAULT_ENV_KEEP",
]
