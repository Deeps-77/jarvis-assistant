"""Persona state machine behind the HUD orb. Qt-free, fully unit-tested.

States mirror the voice protocol (idle/listening/endpointing/thinking/speaking/wake).
The window maps (state, glow, level) to paint parameters at 60fps; all timing
math lives here so tests never need a QApplication.

v2: Added `wake` state for always-on standby, smooth color interpolation
via `fade_to()`, and per-state ring-count / particle-count metadata.
"""

from __future__ import annotations

import math

STATES = ("idle", "wake", "listening", "endpointing", "thinking", "speaking")

# Base pulse speed (radians/sec) per state.
STATE_SPEED = {
    "idle":         0.6,
    "wake":         1.2,   # gentle, attentive pulse
    "listening":    2.0,
    "endpointing":  4.5,
    "thinking":     7.0,
    "speaking":     3.0,
}

# Orb core color per state (R, G, B).
STATE_COLOR = {
    "idle":         (50,  90,  120),
    "wake":         (30, 120, 200),   # cool blue standby
    "listening":    (40, 210, 110),   # green — open ear
    "endpointing":  (240, 200, 60),   # amber — processing
    "thinking":     (160, 100, 255),  # purple — cognition
    "speaking":     (0,  200, 255),   # bright cyan — output
}

# Number of concentric rings drawn per state.
STATE_RINGS = {
    "idle":         1,
    "wake":         2,
    "listening":    2,
    "endpointing":  3,
    "thinking":     4,
    "speaking":     3,
}

# Particle count multiplier per state (0 = no particles).
STATE_PARTICLES = {
    "idle":         0,
    "wake":         8,
    "listening":    16,
    "endpointing":  24,
    "thinking":     32,
    "speaking":     20,
}


class PersonaState:
    """Alive-orb model: state + energy level + animation phase + color fade."""

    def __init__(self) -> None:
        self.state = "idle"
        self.level = 0.0          # mic/playback energy 0..1
        self._phase = 0.0
        # Color fade: interpolate from _from_color toward state color
        self._from_color: tuple[int, int, int] = STATE_COLOR["idle"]
        self._fade_t: float = 1.0  # 1.0 = transition complete

    def set_state(self, state: str) -> str:
        prev = self.state
        self.state = state if state in STATES else "idle"
        if self.state != prev:
            # Kick off smooth color transition from current rendered color.
            self._from_color = self.color_raw()
            self._fade_t = 0.0
        return self.state

    def set_level(self, level: float) -> float:
        try:
            v = float(level)
        except (TypeError, ValueError):
            v = 0.0
        self.level = min(1.0, max(0.0, v))
        return self.level

    def tick(self, dt: float) -> float:
        """Advance animation by dt seconds. Returns glow 0..1."""
        try:
            step = float(dt)
        except (TypeError, ValueError):
            step = 0.0
        if step < 0:
            step = 0.0
        self._phase += step * STATE_SPEED.get(self.state, 0.8)
        # Advance fade (transition duration ≈ 0.4 s)
        self._fade_t = min(1.0, self._fade_t + step / 0.4)
        return self.glow()

    def glow(self) -> float:
        base = 0.5 + 0.5 * math.sin(self._phase)
        return min(1.0, 0.25 + 0.55 * base + 0.45 * self.level)

    def color_raw(self) -> tuple[int, int, int]:
        """Target color (no fade)."""
        return STATE_COLOR.get(self.state, STATE_COLOR["idle"])

    def color(self) -> tuple[int, int, int]:
        """Smoothly interpolated current color."""
        t = self._ease(self._fade_t)
        fr = self._from_color
        to = self.color_raw()
        return (
            int(fr[0] + (to[0] - fr[0]) * t),
            int(fr[1] + (to[1] - fr[1]) * t),
            int(fr[2] + (to[2] - fr[2]) * t),
        )

    def rings(self) -> int:
        return STATE_RINGS.get(self.state, 1)

    def particles(self) -> int:
        return STATE_PARTICLES.get(self.state, 0)

    @staticmethod
    def _ease(t: float) -> float:
        """Smooth-step easing: s-curve 0→1."""
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)


__all__ = [
    "STATES", "STATE_COLOR", "STATE_SPEED", "STATE_RINGS", "STATE_PARTICLES",
    "PersonaState",
]
