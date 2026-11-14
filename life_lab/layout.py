"""Responsive layout decisions for both ordinary and high-DPI desktops."""
from __future__ import annotations

import math


def logical_width(pixel_width: int, screen_dpi: float = 96.0) -> float:
    """Convert physical viewport pixels to a 96-DPI logical comparison width.

    Tk's font size follows system scaling, but widget geometry may use physical
    pixels on an aware Windows process. Use the conservative value for panels.
    """
    if not math.isfinite(screen_dpi) or screen_dpi < 72 or screen_dpi > 768:
        screen_dpi = 96.0
    return pixel_width * 96.0 / max(96.0, screen_dpi)


def window_dimensions(width: int, height: int) -> tuple[int, int]:
    return min(1570, max(780, width-50)), min(950, max(490, height-85))


def permitted_panels(width: int, dpi: float = 96.0) -> int:
    effective = logical_width(width, dpi)
    if effective < 1060:
        return 0
    if effective < 1370:
        return 1
    return 2


def dialog_dimensions(width: int, height: int, desired_w: int, desired_h: int) -> tuple[int, int]:
    return min(desired_w, max(340, width-40)), min(desired_h, max(360, height-90))
