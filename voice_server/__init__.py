"""Jarvis standalone voice console (FastAPI + WebSocket, :8600).

Always-listening, fully offline, ephemeral by construction: no chat
threads, no steps, no persistence — audio in, audio out, RAM only.

Entry: ``python -m voice_server.server`` (or the ``Voice`` option in
``run.ps1``). Browser UI at ``http://localhost:8600/``.
"""
