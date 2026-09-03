"""Jarvis Code Assistant — workspace-oriented code work behind a ReAct agent.

This package is the Phase-1 read-only slice of a larger feature. It is
deliberately additive to the existing chat brain (``core.py``) and the
existing tools (``tools.py``): no existing module is modified.

Layers:

    code_assistant.workspace    Workspace dataclass + path validation
    code_assistant.tools        Read-only LangChain @tool functions
    code_assistant.modes        Plan/Build mode enum + tool gating
    code_assistant.brain        CodeBrain: ReAct agent that streams events
    code_assistant.config       YAML harness config + overrides
    code_ui.py                  Chainlit frontend on port 8500 (Phase 1+)
"""

from code_assistant.brain import ApprovalDecision

__all__ = ["workspace", "tools", "modes", "brain", "config", "ApprovalDecision"]
