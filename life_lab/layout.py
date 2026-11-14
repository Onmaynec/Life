"""Responsive layout decisions in Tk logical pixels (independent of UI state)."""
from __future__ import annotations


def window_dimensions(width: int, height: int) -> tuple[int, int]:
    return min(1570, max(780, width-50)), min(950, max(490, height-85))


def permitted_panels(width: int) -> int:
    if width < 1060:
        return 0  # Sidebars accessible using buttons but not automatically shown.
    if width < 1370:
        return 1
    return 2


def dialog_dimensions(width: int, height: int, desired_w: int, desired_h: int) -> tuple[int, int]:
    return min(desired_w, max(340, width-40)), min(desired_h, max(360, height-90))
