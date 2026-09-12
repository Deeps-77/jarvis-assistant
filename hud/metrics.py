"""HUD telemetry: CPU/mem/net via psutil, GPU via NVML ctypes, no subprocess.

Same zero-subprocess approach as desktop HUDs use: try NVIDIA's nvml.dll
directly through ctypes, degrade to N/A (-1.0) anywhere else. All calls
are cheap and non-raising; the window polls snapshot() on a 1.5s QTimer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_last_net: tuple[float, int, int] | None = None  # (timestamp, sent, recv)


@dataclass(slots=True)
class Snapshot:
    cpu: float = 0.0
    mem: float = 0.0
    net_mbps: float = 0.0
    gpu: float = -1.0  # -1 = unavailable


def _nvml_gpu() -> float:
    """NVIDIA GPU utilisation %, or -1.0 when unavailable. No subprocess."""
    try:
        import ctypes

        class _Util(ctypes.Structure):
            _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

        for dll_name in ("nvml", r"C:\Windows\System32\nvml.dll"):
            try:
                lib = ctypes.WinDLL(dll_name)
                lib.nvmlInit_v2()
                break
            except Exception:
                continue
        else:
            return -1.0
        dev = ctypes.c_void_p()
        lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(dev))
        util = _Util()
        lib.nvmlDeviceGetUtilizationRates(dev, ctypes.byref(util))
        return float(util.gpu)
    except Exception:
        return -1.0


def snapshot() -> Snapshot:
    """Capture current telemetry. Never raises."""
    global _last_net
    snap = Snapshot()
    try:
        import psutil

        snap.cpu = float(psutil.cpu_percent(interval=None))
        snap.mem = float(psutil.virtual_memory().percent)
        counters = psutil.net_io_counters()
        now = time.monotonic()
        if _last_net is not None and counters is not None:
            prev_t, prev_sent, prev_recv = _last_net
            dt = now - prev_t
            if dt > 0:
                delta = (counters.bytes_sent - prev_sent) + (counters.bytes_recv - prev_recv)
                snap.net_mbps = max(0.0, (delta / dt) / (1024 * 1024))
        if counters is not None:
            _last_net = (now, counters.bytes_sent, counters.bytes_recv)
    except Exception:
        logger.debug("telemetry snapshot failed", exc_info=True)
    snap.gpu = _nvml_gpu()
    return snap


def format_snapshot(snap: Snapshot) -> str:
    gpu = "N/A" if snap.gpu < 0 else f"{snap.gpu:.0f}%"
    return (
        f"CPU {snap.cpu:.0f}% · MEM {snap.mem:.0f}% · "
        f"GPU {gpu} · NET {snap.net_mbps:.1f} MB/s"
    )


__all__ = ["Snapshot", "snapshot", "format_snapshot"]
