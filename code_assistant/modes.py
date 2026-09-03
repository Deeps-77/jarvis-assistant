"""Mode definitions for the Jarvis Code Assistant.

Two modes:

- **PLAN** (Phase 1+): Read-only. The agent may inspect the codebase and
  propose changes, but it must never apply them. The toolbelt is restricted to
  ``list_files``, ``read_file``, ``grep_files``, and ``get_file_info``.

- **BUILD** (Phase 2+): Read + write + sandbox execution. The agent may apply
  diffs, create files, and run commands — each mutating action is gated by an
  approval step in the UI.

``filter_tools`` is the single authoritative gating point: ``CodeBrain`` calls
it on every request to ensure the LLM only *sees* tools appropriate for the
current mode. Write tools in Phase 2 also double-check the mode at call time
(belt-and-suspenders).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

# ─────────────────────────────────── Mode ────────────────────────────────────


class Mode(str, Enum):
    """Operating mode for a code-assistant session."""

    PLAN = "plan"
    BUILD = "build"

    def label(self) -> str:
        return {"plan": "📋 Plan", "build": "🔨 Build"}[self.value]

    def description(self) -> str:
        return {
            "plan": "Read-only analysis and planning. No files will be modified.",
            "build": "Full read + write access with approval gate for each change.",
        }[self.value]


# ─────────────────────────────── system prompts ──────────────────────────────

_PLAN_SYSTEM = """\
You are Jarvis, an expert software engineer operating in PLAN mode.

Your role is to:
1. Read and understand the codebase using the provided tools.
2. Analyse problems, explain code, identify bugs, and suggest improvements.
3. Propose specific, actionable changes — but NEVER apply them. Describe what
   you would change and why, using diffs or code blocks in your response.

Constraints (PLAN mode):
- You MAY use: list_files, read_file, grep_files, get_file_info.
- You MUST NOT write, edit, delete, or execute anything.
- If asked to make a change, explain the change in detail and ask the user to
  switch to BUILD mode to apply it.

Be concise but complete. When reading files, start with the most relevant
sections; use get_file_info first for large files to check line counts.
"""

_BUILD_SYSTEM = """\
You are Jarvis, an expert software engineer operating in BUILD mode.

You have full access to read and modify the workspace. Every mutating action
(write_file, edit_file, apply_patch, run_command, delete_path) requires user
approval before execution — the UI will pause and show a diff card.

Guidelines:
- Read before you write: understand the existing code before proposing changes.
- Prefer edit_file / apply_patch over write_file for existing files.
- After any write, offer to run a sanity check (e.g. py_compile) in the sandbox.
- If a change is large, break it into reviewable steps.
"""

SYSTEM_PROMPTS: dict[Mode, str] = {
    Mode.PLAN: _PLAN_SYSTEM,
    Mode.BUILD: _BUILD_SYSTEM,
}

# ─────────────────────────────── tool gating ─────────────────────────────────

# Names of tools that are available in each mode.
# Phase 2 will add write tool names to PLAN_TOOL_NAMES stays unchanged.
PLAN_TOOL_NAMES: frozenset[str] = frozenset(
    {"list_files", "read_file", "grep_files", "get_file_info"}
)

BUILD_TOOL_NAMES: frozenset[str] = frozenset(
    PLAN_TOOL_NAMES
    | {
        # Phase 2 additions (registered here for completeness; not yet implemented):
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


def filter_tools(all_tools: list[Any], mode: Mode) -> list[Any]:
    """Return only the tools permitted in *mode*.

    ``all_tools`` is the list returned by ``make_read_tools`` (Phase 1) plus,
    in Phase 2, ``make_write_tools`` as well. Each element must be a LangChain
    ``BaseTool`` with a ``.name`` attribute.
    """
    allowed = TOOLSETS[mode]
    return [t for t in all_tools if t.name in allowed]


__all__ = [
    "Mode",
    "SYSTEM_PROMPTS",
    "TOOLSETS",
    "PLAN_TOOL_NAMES",
    "BUILD_TOOL_NAMES",
    "filter_tools",
]
