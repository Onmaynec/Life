"""Native Windows title bar colours, with no borderless-window emulation."""
from __future__ import annotations

import sys


def dark_titlebar(window: object, background: str = "#101114") -> bool:
    """Apply DWM dark caption when available; safe no-op outside Windows.

    Windows 11 also supports explicit caption colour (attribute 35). The native
    resize border and window controls remain owned by the OS.
    """
    if sys.platform != "win32":
        return False
    try:
        import ctypes
        from ctypes import wintypes
        window.update_idletasks()
        # Tk may give the child content HWND; GetParent resolves its toplevel.
        parent = ctypes.windll.user32.GetParent
        parent.argtypes = [wintypes.HWND]
        parent.restype = wintypes.HWND
        hwnd = parent(window.winfo_id()) or window.winfo_id()
        dwm = ctypes.windll.dwmapi.DwmSetWindowAttribute
        dwm.argtypes = [wintypes.HWND,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD]
        dwm.restype = ctypes.c_long
        active = wintypes.BOOL(1)
        result = dwm(hwnd, 20, ctypes.byref(active), ctypes.sizeof(active))
        if result != 0:
            result = dwm(hwnd, 19, ctypes.byref(active), ctypes.sizeof(active))
        r, g, b = (int(background[i:i+2], 16) for i in (1, 3, 5))
        caption = wintypes.DWORD(r | (g << 8) | (b << 16))
        dwm(hwnd, 35, ctypes.byref(caption), ctypes.sizeof(caption))
        return result == 0
    except (OSError, ValueError, AttributeError, TypeError):
        return False
