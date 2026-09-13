"""Desktop wallpaper changer (system tier: armed via HUD checkbox)."""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_IMAGE_EXTS = frozenset({".png", ".jpg", ".jpeg", ".bmp"})


@tool
def set_wallpaper(image_path: str) -> str:
    """Sets the desktop wallpaper to an image file.

    Pass the full path to a PNG/JPG/BMP image that already exists. If the
    user doesn't name a file, ask which image to use instead of guessing.
    """
    raw = (image_path or "").strip().strip('"').strip("'")
    if not raw:
        return "ERROR: tell me which image file to use as wallpaper."
    path = Path(raw).expanduser()
    if not path.is_file():
        return f"ERROR: '{raw}' doesn't exist. Give me the full path to an image."
    if path.suffix.lower() not in _IMAGE_EXTS:
        return f"ERROR: '{path.suffix}' isn't a supported image (PNG/JPG/BMP)."
    if os.name != "nt":
        return "ERROR: wallpaper changing is supported on Windows only."
    try:
        SPI_SETDESKWALLPAPER = 20
        ok = ctypes.windll.user32.SystemParametersInfoW(SPI_SETDESKWALLPAPER, 0, str(path), 3)
    except Exception as e:
        logger.warning("set_wallpaper failed: %s", e)
        return f"ERROR: couldn't set the wallpaper ({e})."
    if not ok:
        return "ERROR: Windows refused the wallpaper change."
    logger.info("wallpaper set to %s", path)
    return f"Wallpaper set to {path.name}."


SKILL = {
    "name": "wallpaper",
    "version": "1.0",
    "description": "Set the desktop wallpaper to a chosen image file.",
    "risk": "system",
    "tools": [set_wallpaper],
}
