"""Backend-agnostic LLM provider for Jarvis.

A thin abstraction over LangChain chat models so the chat brain and the
code-assistant brain can both call Ollama or OpenAI through the same
``get_chat_model()`` entry point, and so every call can be wrapped with the
shared ``TokenTracker`` callback without callers caring which backend they hit.

Why not just use ChatOllama / ChatOpenAI directly?
- The two return different metadata shapes; the provider layer normalises both
  so ``TokenTracker`` can read usage from a single code path.
- Tool-calling defaults, temperature, context window, and timeouts differ in
  ways that benefit from being declared once in a config dataclass instead of
  re-typed at every call site.
- A future Anthropic / Gemini provider drops in here without touching the
  brains (``core.py`` and ``code_assistant/brain.py``).

Public surface:
- :class:`LLMConfig` — dataclass holding every knob a provider needs.
- :class:`OllamaProvider` / :class:`OpenAIProvider` — call ``.get_chat_model()``
  to obtain a LangChain ``BaseChatModel`` bound to the supplied callbacks.
- :func:`build_provider` — convenience: ``build_provider(config, callbacks=[...])``.

The config can be loaded two ways:

1. From the environment (``LLMConfig.from_env()``) — used at startup.
2. From an optional ``code_assistant.yaml`` (Phase 3, via ``code_assistant.config``).

Existing ``core.py`` keeps importing ``ChatOllama`` directly until the optional
``USE_SHARED_LLM_CONFIG=true`` flag flips it onto this provider — zero behaviour
change for the chat brain unless the operator opts in.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)


# Default model per provider; only used when the env doesn't specify one.
DEFAULT_OLLAMA_MODEL = "qwen2.5-coder:3b-instruct-q4_K_M"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"

# Cheap static price table (USD per 1M tokens) for the usage dashboard.
# Ollama is local → free. OpenAI numbers are the public list prices; refresh
# quarterly from https://openai.com/api/pricing/.
_OPENAI_PRICES_PER_1M: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "o4-mini": {"input": 1.10, "output": 4.40},
}


def estimate_cost_usd(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    """Best-effort USD cost estimate. Returns 0 for local providers or unknown models."""
    if provider != "openai":
        return 0.0
    prices = _OPENAI_PRICES_PER_1M.get(model)
    if not prices:
        return 0.0
    return (input_tokens / 1_000_000) * prices["input"] + (output_tokens / 1_000_000) * prices["output"]


@dataclass(slots=True)
class LLMConfig:
    """All knobs needed to instantiate a chat model from any supported provider.

    Defaults are tuned for the code-assistant path; ``from_env("")`` mirrors the
    legacy chat-brain behaviour (``OLLAMA_MODEL``, no api key, etc.).
    """

    provider: str = "ollama"
    model: str = DEFAULT_OLLAMA_MODEL
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    api_key: str = ""
    temperature: float = 0.1
    num_ctx: int = 16384
    timeout: int = 600
    keep_alive: str = "-1"
    max_tokens: int = 0  # 0 = provider default
    # Capability flags (auto-detected or manually set via config/env)
    supports_vision: bool = False
    supports_structured_output: bool = False
    supports_function_calling: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.provider = self.provider.strip().lower()
        if self.provider not in ("ollama", "openai"):
            raise ValueError(
                f"Unsupported LLM provider '{self.provider}'. Use 'ollama' or 'openai'."
            )
        if self.max_tokens < 0:
            self.max_tokens = 0
        # Pick a sensible default base URL per provider when the caller left
        # the dataclass default in place. The dataclass default is the Ollama
        # one (because that's the historical Jarvis default), so without this
        # an OpenAIConfig() would silently inherit an Ollama URL.
        ollama_default = DEFAULT_OLLAMA_BASE_URL
        openai_default = DEFAULT_OPENAI_BASE_URL
        if self.provider == "ollama" and self.base_url == openai_default:
            self.base_url = ollama_default
        elif self.provider == "openai" and self.base_url == ollama_default:
            self.base_url = openai_default

    @classmethod
    def from_env(cls, prefix: str = "CODE_") -> "LLMConfig":
        """Build a config from environment variables.

        With ``prefix="CODE_"``:
            CODE_LLM_PROVIDER, CODE_LLM_MODEL, CODE_LLM_BASE_URL,
            CODE_LLM_API_KEY, CODE_LLM_TEMPERATURE, CODE_LLM_NUM_CTX,
            CODE_LLM_TIMEOUT, CODE_LLM_KEEP_ALIVE, CODE_LLM_MAX_TOKENS.

        With ``prefix=""`` (chat brain compatibility):
            OLLAMA_MODEL is read; provider is forced to ``ollama``; api_key and
            openai-specific knobs are ignored.
        """
        def _get(name: str, default: str = "") -> str:
            return os.environ.get(f"{prefix}{name}", default).strip()

        provider = _get("LLM_PROVIDER", "ollama").lower()
        if not provider:
            provider = "ollama"

        if provider == "ollama":
            model = _get("LLM_MODEL") or os.environ.get("OLLAMA_MODEL", "").strip() or DEFAULT_OLLAMA_MODEL
            base_url = _get("LLM_BASE_URL") or os.environ.get("OLLAMA_BASE_URL", "").strip() or DEFAULT_OLLAMA_BASE_URL
            api_key = ""
        else:
            model = _get("LLM_MODEL") or DEFAULT_OPENAI_MODEL
            base_url = _get("LLM_BASE_URL") or DEFAULT_OPENAI_BASE_URL
            api_key = _get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "").strip()
            if not api_key:
                raise ValueError(
                    f"LLM provider '{provider}' requires CODE_LLM_API_KEY (or OPENAI_API_KEY)."
                )

        def _int(name: str, default: int) -> int:
            raw = _get(name)
            if not raw:
                return default
            try:
                return int(raw)
            except ValueError:
                logger.warning("Invalid int for %s%s=%r; using default %d", prefix, name, raw, default)
                return default

        def _float(name: str, default: float) -> float:
            raw = _get(name)
            if not raw:
                return default
            try:
                return float(raw)
            except ValueError:
                logger.warning("Invalid float for %s%s=%r; using default %s", prefix, name, raw, default)
                return default

        return cls(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=_float("LLM_TEMPERATURE", 0.1),
            num_ctx=_int("LLM_NUM_CTX", 16384),
            timeout=_int("LLM_TIMEOUT", 600),
            keep_alive=_get("LLM_KEEP_ALIVE", "-1") or "-1",
            max_tokens=_int("LLM_MAX_TOKENS", 0),
        )

    def describe(self) -> str:
        """Short human-readable label for logs and UI."""
        label = f"{self.provider}:{self.model}"
        if self.provider == "openai":
            label += f" @ {self.base_url}"
        return label


class LLMProvider:
    """Base interface. Subclasses build a LangChain chat model on demand."""

    config: LLMConfig

    def get_chat_model(self, callbacks: list[BaseCallbackHandler] | None = None) -> BaseChatModel:
        raise NotImplementedError

    @staticmethod
    def extract_usage(response: Any) -> tuple[int, int]:
        """Normalise a model response into ``(input_tokens, output_tokens)``.

        Subclasses or callers can fall back to provider-specific metadata;
        ``TokenTracker`` calls this on every ``on_llm_end`` so all backends are
        counted uniformly.
        """
        llm_output = getattr(response, "llm_output", None) or {}
        usage = llm_output.get("token_usage") if isinstance(llm_output, dict) else None
        if isinstance(usage, dict):
            return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))

        # Fallback: scrape from generations[0][i].generation_info (Ollama style).
        total_in = 0
        total_out = 0
        for gen_list in getattr(response, "generations", []) or []:
            for gen in gen_list:
                info = getattr(gen, "generation_info", None) or {}
                total_in += int(info.get("prompt_eval_count", 0) or 0)
                total_out += int(info.get("eval_count", 0) or 0)
        return total_in, total_out


class OllamaProvider(LLMProvider):
    """Local Ollama backend via ``langchain_ollama.ChatOllama``."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._detect_capabilities()

    def _detect_capabilities(self) -> None:
        """Auto-detect capabilities based on model name."""
        model = self.config.model.lower()
        # Vision models often have "vision" in name or are known vision models
        vision_keywords = ["vision", "llava", "bakllava", "moondream", "granite3.2-vision"]
        self.config.supports_vision = any(kw in model for kw in vision_keywords)
        # Structured output: most modern Ollama models support it via function calling
        self.config.supports_structured_output = True
        self.config.supports_function_calling = True

    @staticmethod
    def _normalize_keep_alive(raw: str) -> int | str:
        """Ollama expects ``keep_alive`` as int minutes OR a duration string with a unit.

        ``-1`` as a bare string is invalid (Ollama tries to parse it as a Go
        duration like ``"-1m"``). Map common shorthand to the canonical forms:

        - ``"-1"``, ``"forever"``, ``"never"`` → ``-1`` (int, keep loaded forever)
        - ``"0"``, ``"now"`` → ``0`` (int, unload immediately)
        - ``"5"`` → ``"5m"``
        - anything else → returned as-is (already has a unit, e.g. ``"30s"``)
        """
        s = (raw or "").strip()
        if not s:
            return -1
        low = s.lower()
        if low in ("-1", "forever", "never", "infinite"):
            return -1
        if low in ("0", "now", "off"):
            return 0
        if s.lstrip("-").isdigit():
            # bare number → treat as minutes
            return f"{s}m"
        return s

    def get_chat_model(self, callbacks: list[BaseCallbackHandler] | None = None) -> BaseChatModel:
        from langchain_ollama import ChatOllama

        kwargs: dict[str, Any] = dict(
            model=self.config.model,
            base_url=self.config.base_url,
            temperature=self.config.temperature,
            num_ctx=self.config.num_ctx,
            timeout=self.config.timeout,
            keep_alive=self._normalize_keep_alive(self.config.keep_alive),
        )
        if self.config.max_tokens:
            kwargs["num_predict"] = self.config.max_tokens
        if callbacks:
            kwargs["callbacks"] = callbacks
        return ChatOllama(**kwargs)


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible backend via ``langchain_openai.ChatOpenAI``.

    Works against ``api.openai.com`` and any OpenAI-compatible host
    (Together, Groq, OpenRouter, local llama.cpp server, etc.) by overriding
    ``CODE_LLM_BASE_URL``.
    """

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._detect_capabilities()

    def _detect_capabilities(self) -> None:
        """Auto-detect capabilities based on model name."""
        model = self.config.model.lower()
        # Vision: gpt-4o, gpt-4o-mini, gpt-4-vision, gpt-4-turbo
        vision_models = [
            "gpt-4o", "gpt-4.1", "gpt-4-turbo", "gpt-4-vision",
            "o4-mini", "o3-mini", "o1-mini"
        ]
        self.config.supports_vision = any(vm in model for vm in vision_models)
        # Structured outputs: gpt-4o-2024-08-06+, gpt-4o-mini, o1, o3, o4 series
        structured_models = ["gpt-4o", "gpt-4.1", "o1", "o3", "o4"]
        self.config.supports_structured_output = any(sm in model for sm in structured_models)
        self.config.supports_function_calling = True

    def get_chat_model(self, callbacks: list[BaseCallbackHandler] | None = None) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = dict(
            model=self.config.model,
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            temperature=self.config.temperature,
            timeout=self.config.timeout,
        )
        if self.config.max_tokens:
            kwargs["max_tokens"] = self.config.max_tokens
        if callbacks:
            kwargs["callbacks"] = callbacks
        return ChatOpenAI(**kwargs)


def build_provider(
    config: LLMConfig | None = None,
    callbacks: list[BaseCallbackHandler] | None = None,
) -> tuple[LLMProvider, BaseChatModel]:
    """Construct (provider, chat_model) in one call.

    If ``config`` is omitted, loads :func:`LLMConfig.from_env` with the
    ``CODE_`` prefix. Pass an explicit config in tests or when reading from a
    YAML file (Phase 3).
    """
    cfg = config or LLMConfig.from_env()
    if cfg.provider == "openai":
        provider: LLMProvider = OpenAIProvider(cfg)
    else:
        provider = OllamaProvider(cfg)
    return provider, provider.get_chat_model(callbacks=callbacks)


__all__ = [
    "LLMConfig",
    "LLMProvider",
    "OllamaProvider",
    "OpenAIProvider",
    "build_provider",
    "estimate_cost_usd",
    "DEFAULT_OLLAMA_MODEL",
    "DEFAULT_OPENAI_MODEL",
]
