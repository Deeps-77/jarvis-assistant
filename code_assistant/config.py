"""Harness customisation via ``code_assistant.yaml``.

A single optional YAML file (next to ``code_ui.py``) lets the operator tune
the code assistant without touching Python. Every setting here has an
``os.environ`` fallback (``CODE_LLM_*`` / ``CODE_SANDBOX_TIMEOUT`` / ...);
the env wins when both are set, except where noted.

Schema (all keys optional)::

    code_assistant:
      llm:
        provider: ollama           # or "openai"
        model: qwen2.5-coder:3b
        base_url: http://localhost:11434
        temperature: 0.1
        num_ctx: 16384
        timeout: 600
        keep_alive: -1
        max_tokens: 0              # 0 = provider default
        api_key: ""                # openai only; can be set via env CODE_LLM_API_KEY

      sandbox:
        timeout_seconds: 30
        max_output_bytes: 200000
        env_keep: [PATH, LANG, HOME, ...]   # additive: merged with DEFAULT_ENV_KEEP

      modes:
        plan:
          tools: [list_files, read_file, grep_files, get_file_info]   # "all" for everything
          extra_system_prompt: ""   # appended to the built-in prompt
        build:
          tools: all
          extra_system_prompt: ""

      harness:
        max_tool_rounds: 6
        max_history_messages: 24
        require_approval_for: [write_file, edit_file, ...]   # additive to REQUIRES_APPROVAL

The loader is intentionally tolerant: unknown keys are logged and ignored
so a stale config file does not brick the UI.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "code_assistant.yaml"


# ------------------------------------------------------- dataclass model


@dataclass(slots=True)
class LLMOverride:
    provider: str = ""
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    temperature: float | None = None
    num_ctx: int | None = None
    timeout: int | None = None
    keep_alive: str = ""
    max_tokens: int | None = None


@dataclass(slots=True)
class SandboxOverride:
    timeout_seconds: int | None = None
    max_output_bytes: int | None = None
    env_keep: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ModeOverride:
    tools: list[str] | str | None = None  # list of names, or "all"
    extra_system_prompt: str = ""


@dataclass(slots=True)
class HarnessOverride:
    max_tool_rounds: int | None = None
    max_history_messages: int | None = None
    require_approval_for: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CodeAssistantConfig:
    llm: LLMOverride = field(default_factory=LLMOverride)
    sandbox: SandboxOverride = field(default_factory=SandboxOverride)
    modes: dict[str, ModeOverride] = field(default_factory=dict)
    harness: HarnessOverride = field(default_factory=HarnessOverride)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_overrides(self) -> bool:
        return (self.llm.provider or self.llm.model or self.llm.base_url or 
                self.llm.api_key or self.llm.temperature is not None or
                self.llm.num_ctx is not None or self.llm.timeout is not None or
                self.llm.keep_alive or self.llm.max_tokens is not None or
                self.sandbox.timeout_seconds is not None or 
                self.sandbox.max_output_bytes is not None or
                self.sandbox.env_keep or
                self.modes or
                self.harness.max_tool_rounds is not None or
                self.harness.max_history_messages is not None or
                self.harness.require_approval_for)


# -------------------------------------------------------------- loader


def load_config(path: Path | None = None) -> CodeAssistantConfig:
    """Load + validate ``code_assistant.yaml``. Returns defaults if file missing."""
    target = path or DEFAULT_CONFIG_PATH
    if not target.exists():
        return CodeAssistantConfig()
    try:
        import yaml  # local import so the rest of the package works without PyYAML
    except ImportError:
        logger.warning(
            "PyYAML not installed — code_assistant.yaml ignored. "
            "Install with `uv add pyyaml` to enable YAML config."
        )
        return CodeAssistantConfig()
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as e:
        logger.warning("Failed to parse %s (%s); using defaults", target, e)
        return CodeAssistantConfig()
    if not isinstance(raw, dict):
        logger.warning("Top-level of %s must be a mapping, got %s; using defaults", target, type(raw).__name__)
        return CodeAssistantConfig()

    return _parse(raw)


def _parse(raw: dict[str, Any]) -> CodeAssistantConfig:
    """Translate the raw mapping into dataclasses. Unknown keys are logged."""
    out = CodeAssistantConfig(raw=raw)
    block = raw.get("code_assistant") or raw
    if not isinstance(block, dict):
        logger.warning("'code_assistant' must be a mapping; got %s", type(block).__name__)
        return out

    # ---- llm ----
    llm_raw = block.get("llm")
    if isinstance(llm_raw, dict):
        llm = out.llm
        for k in (
            "provider", "model", "base_url", "api_key",
            "keep_alive",
        ):
            v = llm_raw.get(k)
            if isinstance(v, str) and v:
                setattr(llm, k, v)
        for k in ("temperature",):
            v = llm_raw.get(k)
            if isinstance(v, (int, float)):
                setattr(llm, k, float(v))
        for k in ("num_ctx", "timeout", "max_tokens"):
            v = llm_raw.get(k)
            if isinstance(v, int) and v >= 0:
                setattr(llm, k, v)
        # api_key can also come from env at apply time
        if not llm.api_key:
            llm.api_key = os.environ.get("CODE_LLM_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")

    # ---- sandbox ----
    sb_raw = block.get("sandbox")
    if isinstance(sb_raw, dict):
        sb = out.sandbox
        v = sb_raw.get("timeout_seconds")
        if isinstance(v, int) and v > 0:
            sb.timeout_seconds = v
        v = sb_raw.get("max_output_bytes")
        if isinstance(v, int) and v > 0:
            sb.max_output_bytes = v
        v = sb_raw.get("env_keep")
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            sb.env_keep = [str(x) for x in v]

    # ---- modes ----
    modes_raw = block.get("modes")
    if isinstance(modes_raw, dict):
        for mode_name, mode_raw in modes_raw.items():
            if mode_name not in ("plan", "build"):
                logger.warning("Unknown mode %r in config; skipping", mode_name)
                continue
            if not isinstance(mode_raw, dict):
                continue
            mo = ModeOverride()
            t = mode_raw.get("tools")
            if t == "all" or t is None:
                mo.tools = "all" if t == "all" else None
            elif isinstance(t, list) and all(isinstance(x, str) for x in t):
                mo.tools = [str(x) for x in t]
            else:
                logger.warning("modes.%s.tools must be 'all' or a list of names", mode_name)
            prompt = mode_raw.get("extra_system_prompt")
            if isinstance(prompt, str):
                mo.extra_system_prompt = prompt
            out.modes[mode_name] = mo

    # ---- harness ----
    h_raw = block.get("harness")
    if isinstance(h_raw, dict):
        h = out.harness
        v = h_raw.get("max_tool_rounds")
        if isinstance(v, int) and 1 <= v <= 50:
            h.max_tool_rounds = v
        v = h_raw.get("max_history_messages")
        if isinstance(v, int) and 1 <= v <= 200:
            h.max_history_messages = v
        v = h_raw.get("require_approval_for")
        if isinstance(v, list) and all(isinstance(x, str) for x in v):
            h.require_approval_for = [str(x) for x in v]

    return out


__all__ = [
    "CodeAssistantConfig",
    "LLMOverride",
    "SandboxOverride",
    "ModeOverride",
    "HarnessOverride",
    "load_config",
    "DEFAULT_CONFIG_PATH",
]
