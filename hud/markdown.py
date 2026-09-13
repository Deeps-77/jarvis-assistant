"""Qt-safe markdown rendering for the HUD transcript.

The shared Telegram renderer (``main.render_markdown_blocks``) keeps
single newlines as literal ``\\n`` (softbreaks) — Telegram honors those,
but QTextEdit collapses them into spaces, turning chat replies into one
wall of text ("2026\\n1. ..." becomes "20261. ...").

This wrapper converts newlines to ``<br/>`` everywhere except inside
``<pre>`` code blocks (where QTextEdit honors them natively).
"""

from __future__ import annotations

import html as _html
import re

_PRE_RE = re.compile(r"(?is)(<pre.*?</pre>)")


def _nl_to_br(fragment: str) -> str:
    parts = _PRE_RE.split(fragment)
    for i in range(0, len(parts), 2):
        parts[i] = parts[i].replace("\n", "<br/>")
    return "".join(parts)


def render_chat_html(md_text: str) -> str:
    """Render markdown to a QTextEdit-ready HTML string."""
    try:
        from main import render_markdown_blocks

        blocks = render_markdown_blocks(md_text or "")
    except Exception:
        return _html.escape(md_text or "").replace("\n", "<br/>")
    # Blocks join with a blank line: the renderer emits bare fragments
    # (no <p> wrappers), so without this, "para\n1. list" collapses to
    # "para1. list" — the other half of the reported wall-of-text.
    return "<br/><br/>".join(_nl_to_br(b) for b in blocks if b.strip())


__all__ = ["render_chat_html"]
