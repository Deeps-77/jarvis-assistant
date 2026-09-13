"""Read-only system introspection: performance Q&A and disk stats."""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from security import sanitize_external

logger = logging.getLogger(__name__)


@tool
def system_status() -> str:
    """Reports PC performance: CPU, memory, GPU, battery, busiest processes.

    Use when asked how the computer is doing, CPU/RAM usage, temperature
    proxies, or what's eating resources.
    """
    try:
        import psutil

        from hud.metrics import format_snapshot, snapshot

        lines = [format_snapshot(snapshot())]
        try:
            battery = psutil.sensors_battery()
            if battery is not None:
                lines.append(
                    f"Battery: {battery.percent:.0f}%"
                    f" ({'charging' if battery.power_plugged else 'on battery'})"
                )
        except Exception:
            pass
        try:
            top = sorted(psutil.process_iter(["name", "cpu_percent"]), key=lambda p: p.info.get("cpu_percent") or 0.0, reverse=True)[:5]
            names = ", ".join(f"{p.info.get('name') or '?'} ({(p.info.get('cpu_percent') or 0.0):.0f}%)" for p in top)
            lines.append(f"Busiest processes: {names}")
        except Exception:
            pass
        return sanitize_external("\n".join(lines), "system status")
    except Exception as e:
        logger.warning("system_status failed: %s", e)
        return f"ERROR: couldn't read system status ({e})."


@tool
def desk_stats() -> str:
    """Reports disk usage per drive and system uptime.

    Use when asked about free space, disk usage, or how long the PC has
    been running.
    """
    try:
        import datetime

        import psutil

        lines = []
        for part in psutil.disk_partitions(all=False):
            try:
                usage = psutil.disk_usage(part.mountpoint)
            except OSError:
                continue
            lines.append(
                f"{part.device} {usage.free / 1024**3:.1f} GB free of "
                f"{usage.total / 1024**3:.1f} GB ({usage.percent:.0f}% used)"
            )
        try:
            uptime = datetime.datetime.now() - datetime.datetime.fromtimestamp(psutil.boot_time())
            hours, rem = divmod(int(uptime.total_seconds()), 3600)
            lines.append(f"Uptime: {hours}h {rem // 60}m")
        except Exception:
            pass
        if not lines:
            return "ERROR: no disk information available."
        return sanitize_external("\n".join(lines), "disk stats")
    except Exception as e:
        logger.warning("desk_stats failed: %s", e)
        return f"ERROR: couldn't read disk stats ({e})."


SKILL = {
    "name": "system_info",
    "version": "1.0",
    "description": "PC performance Q&A (CPU/RAM/GPU/battery/processes) and disk stats.",
    "risk": "readonly",
    "tools": [system_status, desk_stats],
}
