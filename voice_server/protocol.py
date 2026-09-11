"""WebSocket wire protocol for the voice console.

Binary frames (client -> server): raw int16 mono PCM @16kHz.
Binary frames (server -> client): WAV bytes (single sentence) for playback.

JSON frames (both directions), ``{"type": ..., ...}``:

- client -> server: ``{"type": "start"}`` (begin listening),
  ``{"type": "stop"}`` (pause listening; pending speech is flushed as a
  final turn, session stays connected — only a real disconnect ends it),
  ``{"type": "barge"}`` (user cut in),
  ``{"type": "text", "text": ...}`` (type-fallback box),
  ``{"type": "threshold", "value": float}`` (live-tune the VAD energy gate),
  ``{"type": "mute", "muted": bool}`` (speaker mute state, informational).
- server -> client: ``{"type": "state", "state": ...}`` (idle, listening,
  endpointing, thinking, speaking), ``{"type": "transcript", "text": ...}``
  (final user transcript), ``{"type": "partial", "speechMs": int}``
  (speech detected but not yet endpointed — throttled heartbeat),
  ``{"type": "reply", "text": ...}`` (assistant text for the transcript
  feed), ``{"type": "audio_done", "id": ...}`` (sentence fully sent),
  ``{"type": "error", "message": ...}``.
"""

from __future__ import annotations

STATES = ("idle", "listening", "endpointing", "thinking", "speaking")

CLIENT_TYPES = ("start", "stop", "barge", "text", "threshold", "mute")
SERVER_TYPES = ("state", "transcript", "partial", "reply", "audio_done", "error")


def server_msg(msg_type: str, **fields) -> dict:
    """Build a server -> client JSON frame."""
    if msg_type not in SERVER_TYPES:
        raise ValueError(f"unknown server message type: {msg_type!r}")
    return {"type": msg_type, **fields}


__all__ = ["STATES", "CLIENT_TYPES", "SERVER_TYPES", "server_msg"]
