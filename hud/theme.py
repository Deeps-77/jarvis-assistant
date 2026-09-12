"""HUD theme: central palette + dark stylesheet + live accent.

Reimplemented from scratch: a cyan-on-black default, an Iron Man
amber/gold alt, one hue knob that re-derives the accent family, and a
single stylesheet builder. The orb reads this module at paint time so
an accent change recolors it on the next frame with no restart.
"""

from __future__ import annotations

import colorsys

BG = "#05080e"
PANEL = "#0a1119"
BORDER = "#12303f"
TEXT = "#cfe9f5"
DIM = "#5f7d8f"
ACCENT = "#00d4ff"
ACCENT_DIM = "#007a99"
DANGER = "#ff5470"
OK = "#3ddc84"

# Iron Man / gold theme
AMBER = "#ff9500"
AMBER_DIM = "#994200"


def set_accent_hue(hue: int) -> str:
    """Set accent from a hue angle 0..359. Returns the new accent hex."""
    global ACCENT, ACCENT_DIM
    try:
        h = float(hue) % 360.0
    except (TypeError, ValueError):
        return ACCENT
    r, g, b = colorsys.hsv_to_rgb(h / 360.0, 1.0, 1.0)
    ACCENT = "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
    r2, g2, b2 = colorsys.hsv_to_rgb(h / 360.0, 0.9, 0.55)
    ACCENT_DIM = "#{:02x}{:02x}{:02x}".format(int(r2 * 255), int(g2 * 255), int(b2 * 255))
    return ACCENT


def build_stylesheet() -> str:
    return f"""
    QMainWindow, QWidget {{ background: {BG}; color: {TEXT}; font-family: 'Segoe UI', 'Inter', sans-serif; }}
    QTextEdit {{
        background: {PANEL}; color: {TEXT};
        border: 1px solid {BORDER}; border-radius: 6px; padding: 6px;
        font-family: 'Consolas', 'Courier New', monospace; font-size: 11px;
        selection-background-color: {ACCENT_DIM};
    }}
    QLineEdit {{
        background: {PANEL}; color: {TEXT};
        border: 1px solid {BORDER}; border-radius: 6px; padding: 6px;
        font-size: 12px;
    }}
    QLineEdit:focus {{ border: 1px solid {ACCENT}; }}
    QPushButton {{
        background: transparent; color: {ACCENT};
        border: 1px solid {ACCENT_DIM}; border-radius: 6px; padding: 6px 12px;
        font-size: 11px; letter-spacing: 0.5px;
    }}
    QPushButton:hover {{ border: 1px solid {ACCENT}; background: rgba(0,212,255,0.07); }}
    QPushButton:checked {{ background: {ACCENT}; color: {BG}; font-weight: bold; }}
    QPushButton:disabled {{ color: {DIM}; border-color: {BORDER}; }}
    QPushButton#danger {{ color: {DANGER}; border-color: {DANGER}; }}
    QComboBox {{
        background: {PANEL}; color: {TEXT};
        border: 1px solid {BORDER}; border-radius: 6px; padding: 4px 8px;
    }}
    QComboBox::drop-down {{ border: none; }}
    QComboBox QAbstractItemView {{ background: {PANEL}; color: {TEXT}; border: 1px solid {BORDER}; }}
    QLabel {{ background: transparent; }}
    QLabel#live {{ color: #8a93a6; font-style: italic; font-size: 10px; }}
    QLabel#metrics {{ color: {DIM}; font-size: 9px; font-family: 'Consolas', monospace; }}
    QLabel#header {{ color: {ACCENT}; font-size: 13px; letter-spacing: 3px; font-weight: bold; }}
    QLabel#clock {{ color: {DIM}; font-size: 10px; font-family: 'Consolas', monospace; }}
    QProgressBar {{
        background: {PANEL}; border: 1px solid {BORDER}; border-radius: 5px; max-height: 8px;
    }}
    QProgressBar::chunk {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {ACCENT_DIM}, stop:1 {ACCENT}); border-radius: 4px; }}
    QSplitter::handle {{ background: {BORDER}; width: 2px; }}
    QSlider::groove:horizontal {{ background: {PANEL}; height: 4px; border-radius: 2px; border: 1px solid {BORDER}; }}
    QSlider::handle:horizontal {{
        background: {ACCENT}; width: 12px; height: 12px; border-radius: 6px; margin: -5px 0;
    }}
    QSlider::sub-page:horizontal {{ background: {ACCENT_DIM}; border-radius: 2px; }}
    QScrollBar:vertical {{ background: {BG}; width: 6px; }}
    QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 3px; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
    """


def apply_theme(app) -> None:
    """Apply the current palette application-wide (live, no restart)."""
    try:
        app.setStyleSheet(build_stylesheet())
    except Exception:
        pass


__all__ = [
    "BG", "PANEL", "BORDER", "TEXT", "DIM",
    "ACCENT", "ACCENT_DIM", "DANGER", "OK", "AMBER", "AMBER_DIM",
    "set_accent_hue", "build_stylesheet", "apply_theme",
]
