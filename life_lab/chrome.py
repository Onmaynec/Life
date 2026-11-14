"""Windows native caption styling with conservative DWM fallbacks.

Keep the system border, accessibility keys, resize handles and window controls.
No custom frameless overlays are used. Unsupported DWM attributes are no-ops.
"""
from __future__ import annotations

import sys


def _rgb(background: str) -> int:
    if not isinstance(background, str) or len(background) != 7 or not background.startswith("#"):
        raise ValueError("Цвет должен быть в формате #RRGGBB")
    r, g, b = (int(background[i:i+2],16) for i in (1,3,5))
    return r | (g << 8) | (b << 16)


def dark_titlebar(window: object, background: str = "#101114") -> bool:
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes
        window.update_idletasks()
        user32 = ctypes.windll.user32
        ancestor = user32.GetAncestor
        ancestor.argtypes = [wintypes.HWND, wintypes.UINT]
        ancestor.restype = wintypes.HWND
        hwnd = ancestor(window.winfo_id(), 2) or window.winfo_id()  # GA_ROOT
        if not hwnd:
            return False
        dwm = ctypes.windll.dwmapi.DwmSetWindowAttribute
        dwm.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
        dwm.restype = ctypes.c_long

        def set_attr(attribute: int, value: int) -> bool:
            data = wintypes.DWORD(value)
            return dwm(hwnd, attribute, ctypes.byref(data), ctypes.sizeof(data)) == 0

        # Attr 20 on modern Windows, 19 on older Windows 10 builds.
        dark = set_attr(20, 1) or set_attr(19, 1)
        # Explicit caption and text are supported by some Win11 builds.
        caption = set_attr(35, _rgb(background))
        if caption:
            set_attr(36, 0xFFFFFF)  # readable white native caption text
        return dark or caption
    except (OSError, ValueError, AttributeError, TypeError, OverflowError):
        return False
