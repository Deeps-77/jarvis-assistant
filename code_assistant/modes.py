"""Plan/Build mode gating for the code assistant.

Two modes drive every assistant turn:

- :attr:`Mode.PLAN` — read-only. The agent can list, read, and grep files but
  cannot create, edit, delete, or execute anything. The user reviews the
  proposed plan in the UI before flipping to Build.

- :attr:`Mode.BUILD` — full toolset (Phase 2+). Exposes the read-only tools
  plus the write/exec toolset, every mutating call guarded by an approval
  gate in the UI.

The mode is the **single source of truth** for tool availability. Three
layers enforce it:

1. :func:`filter_tools` selects which tools the agent's ReAct graph sees.
2. Each write tool calls :func:`code_assistant.tools._require_build_mode`
   on entry and raises :class:`code_assistant.tools.ModeError` if the
   brain isn't in BUILD mode. The LangChain ReAct loop catches the error
   and feeds it back into the conversation so the model self-corrects.
3. The brain's event stream pauses on every tool whose name is in
   :data:`code_assistant.tools.REQUIRES_APPROVAL` and yields an
   ``approval_required`` event for the UI to confirm.
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable

from langchain_core.tools import BaseTool


class Mode(str, Enum):
    PLAN = "plan"
    BUILD = "build"

    @classmethod
    def parse(cls, raw: str | None) -> "Mode":
        if not raw:
            return cls.PLAN
        try:
            return cls(raw.strip().lower())
        except ValueError:
            return cls.PLAN


# Tool subsets per mode. Names must match ``BaseTool.name``.
PLAN_TOOL_NAMES: frozenset[str] = frozenset(
    {"list_files", "read_file", "grep_files", "get_file_info"}
)
# BUILD includes everything — read-only tools PLUS the write/exec toolset.
# Write tools are individually gated by ``_require_build_mode`` AND the
# approval flow.
BUILD_TOOL_NAMES: frozenset[str] = frozenset(
    {
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
    }
)

TOOLSETS: dict[Mode, frozenset[str]] = {
    Mode.PLAN: PLAN_TOOL_NAMES,
    Mode.BUILD: BUILD_TOOL_NAMES,
}


SYSTEM_PROMPT_BASE = """You are Jarvis Code Assistant, an expert software engineer working inside the user's local project.

You operate inside a single Workspace: every path you pass to a tool is resolved
relative to that root. Paths must NOT be absolute and must NOT contain '..'.
The runtime refuses both. If you need something outside the workspace, ask the
user to add it to the deny-glob list or open a wider workspace.

Behaviour rules (always):
- Be precise. Cite filenames and line numbers when you refer to code.
- Quote exact code before commenting on it; do not paraphrase.
- When you propose code changes, show the exact hunks you would apply.
- Never invent file contents. Use the read_file tool first if unsure.
- If the user asks something you can't answer from the workspace, say so
  instead of guessing.
- Prefer the smallest tool that gets the job done.
"""


PLAN_PROMPT_SUFFIX = """

CURRENT MODE: PLAN (read-only)

You have access to these tools ONLY:
- list_files: enumerate files in a directory (with optional glob filter)
- read_file: paginated file read (use start_line/end_line for big files)
- grep_files: regex search across files (optional glob filter)
- get_file_info: size, mtime, language for one file

You must NOT:
- Create, modify, move, or delete any file
- Run any shell command, code, tests, or builds
- Claim to have done something you only described

Format your final answer as a concrete plan the user can review before flipping
to Build mode:
1. Goal (one sentence)
2. Files to change (with the relevant function/class/section name)
3. For each change: the exact code hunk (or a precise description if the file is new)
4. Risks and tests to run
5. Open questions for the user
"""


BUILD_PROMPT_SUFFIX = """

CURRENT MODE: BUILD (full toolset with approval gate)

Read tools (always available, no approval needed):
- list_files, read_file, grep_files, get_file_info

Write/exec tools (every call is approval-gated in the UI before it runs):
- write_file: create or replace a file (atomic via temp+rename)
- edit_file: targeted find/replace (refuses ambiguous matches)
- apply_patch: apply a unified diff
- mkdir: recursive directory creation
- delete_path: soft-delete to .jarvis-sandbox/trash
- run_command: single shell command in .jarvis-sandbox/tmp/ (env stripped,
  hard timeout, output capped)
- copy_out: move a sandbox-produced file into the workspace

Rules for using write/exec tools:
1. Always inspect the file first with read_file or grep_files before editing.
2. Prefer edit_file over write_file when the change is small and local.
3. run_command does NOT support shell metacharacters (|, >, ;, &&). Pass one
   executable invocation per call. For multi-step workflows, call run_command
   multiple times.
4. Files produced by run_command live in .jarvis-sandbox/tmp/. To make them
   visible in the workspace, call copy_out.
5. Never claim a write succeeded unless the tool returned "APPLIED: <path>".
   A USER_REJECTED response means the user declined — adapt the plan or ask.
"""


SYSTEM_PROMPTS: dict[Mode, str] = {
    Mode.PLAN: SYSTEM_PROMPT_BASE + PLAN_PROMPT_SUFFIX,
    Mode.BUILD: SYSTEM_PROMPT_BASE + BUILD_PROMPT_SUFFIX,
}


def filter_tools(tools: Iterable[BaseTool], mode: Mode) -> list[BaseTool]:
    """Return the subset of ``tools`` allowed in ``mode``."""
    allowed = TOOLSETS.get(mode, PLAN_TOOL_NAMES)
    return [t for t in tools if t.name in allowed]


def workspace_preamble(workspace_root: str, mode: Mode) -> str:
    """Short system-prefix the brain injects so the model knows the root."""
    return (
        f"\n\nWorkspace root: {workspace_root}\n"
        f"Active mode: {mode.value.upper()}\n"
        f"All file paths must be relative to the workspace root."
    )


__all__ = [
    "Mode",
    "TOOLSETS",
    "PLAN_TOOL_NAMES",
    "BUILD_TOOL_NAMES",
    "SYSTEM_PROMPTS",
    "filter_tools",
    "workspace_preamble",
]
