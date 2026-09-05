"""Approval-gate UX: cards, replies, outcome chips."""
from __future__ import annotations

from typing import Any

import chainlit as cl

from code_assistant.brain import ApprovalDecision


def _looks_like_json_tool_call(text: str) -> bool:
    """True if the final reply is just a serialised tool call (small-model artifact)."""
    s = (text or "").strip()
    if not s.startswith("{"):
        return False
    return '"name"' in s and ('"arguments"' in s or '"args"' in s)


# ------------------------------------------------------------ approval UX


def _format_args(args: dict[str, Any]) -> str:
    """Render tool args as a compact, readable block."""
    import json
    try:
        return json.dumps(args, indent=2, ensure_ascii=False)[:2000]
    except (TypeError, ValueError):
        return repr(args)[:2000]


async def _request_approval(answer: cl.Message, payload: dict[str, Any]) -> ApprovalDecision:
    """Show the approval card and block until the user responds.

    Uses Chainlit's ``AskUserMessage`` so the approval flow blocks the
    current handler; the brain's ``_approval_event`` stays set until the
    response comes back.
    """
    name = payload.get("name", "?")
    args = payload.get("args", {})
    args_block = _format_args(args)
    card = (
        f"### ⚠️ Approval required\n\n"
        f"**Tool:** `{name}`\n\n"
        f"**Arguments:**\n```json\n{args_block}\n```\n\n"
        "Reply with one of:\n"
        "- `approve` — run the tool with these arguments\n"
        "- `reject` — decline (model will adapt)\n"
        "- `reject <reason>` — decline with an explanation for the model\n"
        "- `edit` — approve with edited args (paste a JSON object matching the schema)\n"
    )
    # Stream the card into the current answer so the user sees it inline.
    answer.content = (answer.content or "") + "\n\n" + card
    await answer.update()

    response = await cl.AskUserMessage(
        content="Approve this tool call?",
        timeout=180,
    ).send()

    if response is None:
        return ApprovalDecision(decision="reject", reason="timeout")

    text = (response or "").strip()
    low = text.lower()
    if low in ("approve", "yes", "y", "ok"):
        return ApprovalDecision(decision="approve")
    if low.startswith("edit"):
        rest = text[4:].strip()
        # The user might paste JSON. Try to parse.
        import json
        try:
            edited = json.loads(rest) if rest else args
        except json.JSONDecodeError:
            await cl.Message(
                content="Couldn't parse edit JSON. Rejecting with reason."
            ).send()
            return ApprovalDecision(
                decision="reject",
                reason=f"edit was unparseable: {rest[:120]}",
            )
        return ApprovalDecision(decision="edit", args=edited)
    if low.startswith("reject"):
        reason = text[len("reject"):].strip() or "user declined"
        return ApprovalDecision(decision="reject", reason=reason)
    # Default: treat unknown reply as rejection with explanation.
    return ApprovalDecision(decision="reject", reason=f"unrecognised reply: {text[:120]}")


def _approval_outcome_chip(decision: ApprovalDecision) -> str:
    """Render a small status chip so the conversation log shows the decision."""
    if decision.decision == "approve":
        return "> ✅ _approved_\n\n"
    if decision.decision == "edit":
        return "> ✏️ _approved with edited args_\n\n"
    return f"> ❌ _rejected ({decision.reason[:80]})_\n\n"
