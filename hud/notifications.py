"""HUD toast notifications + file-based watcher for scheduled reminders.

ToastWidget: non-modal floating toast that auto-dismisses.
NotificationWatcher: polls ``data/pending_notifications.json`` and emits
signals when new entries appear (written by the reminders skill).
"""

from __future__ import annotations

import json
import logging
import time

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

TOAST_DURATION_MS = 8000
POLL_INTERVAL_MS = 2000


def _read_notifications(path) -> list[dict]:
    """Read and return the pending notifications list. Never raises."""
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _write_notifications(path, entries: list[dict]) -> None:
    """Atomically write the notifications list."""
    try:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        logger.debug("Failed to write notifications file", exc_info=True)


def _remove_shown(path, shown_ts: list[float]) -> None:
    """Remove entries whose ``ts`` matches any in shown_ts."""
    entries = _read_notifications(path)
    shown_set = set(shown_ts)
    remaining = [e for e in entries if e.get("ts") not in shown_set]
    _write_notifications(path, remaining)


# --------------------------------------------------------------- toast widget


class ToastWidget(QFrame):
    """Non-modal floating toast that auto-dismisses after a delay."""

    def __init__(self, title: str, message: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            "QFrame { background: #0a1119; border: 1px solid #12303f; border-radius: 8px; }"
            "QLabel { color: #cfe9f5; background: transparent; }"
            "QPushButton { color: #5f7d8f; background: transparent; border: none; "
            "font-size: 14px; padding: 2px 6px; }"
            "QPushButton:hover { color: #00d4ff; }"
        )
        self.setFixedWidth(320)
        self.setMinimumHeight(60)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 8, 12, 8)
        outer.setSpacing(2)

        top = QHBoxLayout()
        top.setSpacing(8)
        title_lbl = QLabel(f"<b>{title}</b>")
        title_lbl.setStyleSheet("color: #00d4ff; font-size: 12px;")
        top.addWidget(title_lbl)
        top.addStretch()
        close_btn = QPushButton("\u00d7")
        close_btn.setFixedSize(20, 20)
        close_btn.clicked.connect(self._dismiss)
        top.addWidget(close_btn)
        outer.addLayout(top)

        msg_lbl = QLabel(message)
        msg_lbl.setWordWrap(True)
        msg_lbl.setStyleSheet("font-size: 11px;")
        outer.addWidget(msg_lbl)

        # Auto-dismiss timer.
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(TOAST_DURATION_MS)
        self._timer.timeout.connect(self._dismiss)
        self._timer.start()

    def _dismiss(self) -> None:
        self._timer.stop()
        self.hide()
        self.deleteLater()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Position at top-right of parent window.
        if self.parent():
            pw = self.parent()
            x = pw.width() - self.width() - 16
            y = 16
            self.move(x, y)


# ----------------------------------------------------------- notification watcher


class NotificationWatcher(QWidget):
    """Polls the pending notifications file and emits signals for new entries."""

    notification = pyqtSignal(str, str)  # (title, message)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._shown_ts: list[float] = []
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(POLL_INTERVAL_MS)

    @property
    def _path(self):
        from paths import data_path
        return data_path("pending_notifications.json")

    def _poll(self) -> None:
        entries = _read_notifications(self._path)
        if not entries:
            return
        for entry in entries:
            ts = entry.get("ts", 0)
            if ts in self._shown_ts:
                continue
            title = entry.get("title", "Jarvis")
            message = entry.get("message", "")
            if message:
                self._shown_ts.append(ts)
                self.notification.emit(title, message)
        # Prune old shown timestamps (keep last 200).
        if len(self._shown_ts) > 200:
            self._shown_ts = self._shown_ts[-200:]
        # Clean up shown entries from file.
        if self._shown_ts:
            _remove_shown(self._path, self._shown_ts)


__all__ = ["ToastWidget", "NotificationWatcher"]
