"""Jarvis Code Assistant — workspace-oriented code work behind a ReAct agent.

Additive to the chat brain (``core.py``) and the chat tools
(``chat_tools.py``): no existing module is modified.

Layers:

    code_assistant.workspace    Workspace dataclass + path validation + registry
    code_assistant.tools        LangChain @tool functions (read + write + exec)
    code_assistant.modes        Plan/Build mode enum + tool gating + prompts
    code_assistant.brain        CodeBrain: manual ReAct loop, approval gate,
                                thinking stream, empty-round nudge
    code_assistant.sessions     Per-workspace chat sessions (save/resume/delete)
    code_assistant.config       YAML harness config + overrides
    code_ui/                    Chainlit frontend on port 8500 (package;
                                ``code_ui.py`` is a thin entry shim)
"""

from code_assistant.brain import ApprovalDecision
from code_assistant.sessions import ChatSession, SessionStore

__all__ = [
    "workspace",
    "tools",
    "modes",
    "brain",
    "sessions",
    "config",
    "ApprovalDecision",
    "ChatSession",
    "SessionStore",
]
