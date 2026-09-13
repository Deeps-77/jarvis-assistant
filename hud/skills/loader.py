"""Drop-in skill loader with per-file crash isolation and risk gating."""

from __future__ import annotations

import importlib.util
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

RISKS = ("readonly", "system")
SKILL_DIR = Path(__file__).resolve().parent / "packs"


@dataclass(slots=True)
class Skill:
    name: str
    version: str
    description: str
    risk: str
    tools: list = field(default_factory=list)
    source: str = ""
    # Module-level direct_match(text) -> (tool_name, args) | None callables.
    # Deterministic routes for intents weak models fumble through tool choice.
    matchers: list = field(default_factory=list)


_registry: dict[str, Skill] = {}
_context: dict = {}  # per-turn context (chat_key, owner); see set_turn_context
_system_armed: bool = False  # HUD checkbox state; see set_system_armed
_providers: dict = {}  # process-lifetime providers (screen_grabber, on_goodbye)


def set_system_armed(armed: bool) -> None:
    """Arm/disarm system-tier skills (HUD checkbox). Env can force it on."""
    global _system_armed
    _system_armed = bool(armed) or os.environ.get("HUD_SKILLS_ALLOW", "").strip().lower() in (
        "1", "true", "yes", "system", "all",
    )


def set_turn_context(**fields) -> None:
    """Set context for the active turn (chat_key, owner, on_goodbye...)."""
    _context.clear()
    _context.update(fields)


def get_context(key: str, default=None):
    return _context.get(key, default)


def set_provider(name: str, fn) -> None:
    """Register a process-lifetime provider (screen_grabber, on_goodbye)."""
    if fn is None:
        _providers.pop(name, None)
    else:
        _providers[name] = fn


def get_provider(name: str, default=None):
    return _providers.get(name, default)


def _validate_manifest(mod, path: Path) -> Skill | None:
    manifest = getattr(mod, "SKILL", None)
    if not isinstance(manifest, dict):
        logger.warning("Skill %s has no SKILL dict; skipped", path.name)
        return None
    name = str(manifest.get("name") or path.stem).strip()
    risk = str(manifest.get("risk") or "readonly").strip().lower()
    if risk not in RISKS:
        logger.warning("Skill %s has unknown risk %r; skipped", path.name, risk)
        return None
    tools = manifest.get("tools") or []
    if not isinstance(tools, list):
        logger.warning("Skill %s tools is not a list; skipped", path.name)
        return None
    valid = []
    for tool in tools:
        tname = getattr(tool, "name", "")
        if not tname or not getattr(tool, "description", ""):
            logger.warning("Skill %s has a tool without name/description; dropped", name)
            continue
        valid.append(tool)
    if not valid:
        logger.warning("Skill %s exposes no valid tools; skipped", path.name)
        return None
    matchers = []
    direct_match = getattr(mod, "direct_match", None)
    if callable(direct_match):
        matchers.append(direct_match)
    return Skill(
        name=name,
        version=str(manifest.get("version") or "0"),
        description=str(manifest.get("description") or ""),
        risk=risk,
        tools=valid,
        source=path.name,
        matchers=matchers,
    )


def load_skills(skill_dir: Path | None = None) -> dict[str, Skill]:
    """(Re)load all skill packs. One bad file never breaks the others."""
    global _registry
    _registry = {}
    directory = Path(skill_dir) if skill_dir else SKILL_DIR
    if not directory.is_dir():
        return _registry
    for path in sorted(directory.glob("*.py")):
        if path.name.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(f"hud_skill_{path.stem}", path)
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot build spec for {path.name}")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception:
            logger.warning("Skill %s failed to import; skipped", path.name, exc_info=True)
            continue
        try:
            skill = _validate_manifest(mod, path)
        except Exception:
            logger.warning("Skill %s manifest invalid; skipped", path.name, exc_info=True)
            continue
        if skill is None:
            continue
        if skill.name in _registry:
            logger.warning("Duplicate skill name %r (%s); keeping first", skill.name, path.name)
            continue
        _registry[skill.name] = skill
        logger.info("Skill loaded: %s v%s [%s] (%d tools)", skill.name, skill.version, skill.risk, len(skill.tools))
    return _registry


def turn_tools() -> list:
    """Tools for the current turn: readonly always, system only when armed."""
    out = []
    for skill in _registry.values():
        if skill.risk == "system" and not _system_armed:
            continue
        out.extend(skill.tools)
    return out


def direct_routes() -> list:
    """Deterministic matchers for the current turn.

    Each matcher maps text -> (tool_name, args) | None. Only matchers from
    skills active this turn are returned, so armed-off system skills can
    never fire directly. Callers resolve tool_name against their own belt.
    """
    out = []
    for skill in _registry.values():
        if skill.risk == "system" and not _system_armed:
            continue
        out.extend(skill.matchers)
    return out


def describe() -> str:
    """Human-readable skill catalog for 'what can you do?' answers."""
    if not _registry:
        return "No skills installed."
    lines = []
    for skill in sorted(_registry.values(), key=lambda s: s.name):
        gate = "" if skill.risk == "readonly" else " (needs system-skills toggle)"
        lines.append(f"- {skill.name}: {skill.description}{gate}")
    return "\n".join(lines)


__all__ = ["RISKS", "SKILL_DIR", "Skill", "describe", "direct_routes", "get_context", "get_provider", "load_skills", "set_provider", "set_system_armed", "set_turn_context", "turn_tools"]
