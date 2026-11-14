"""Small, local UI preferences; never part of a simulation save."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

from .visuals import PALETTES, STYLES


def settings_path() -> Path:
    if sys.platform == "win32":
        base = Path(os.getenv("APPDATA") or Path.home() / "AppData" / "Roaming")
    else:
        base = Path(os.getenv("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "Life" / "preferences.json"


@dataclass(frozen=True)
class Preferences:
    palette: str = "Графит"
    style: str = "Свечение"
    grid: bool = True
    trails: bool = True
    speed: int = 8
    skin: str = "Графит / циан"
    animation: bool = True
    overlay: bool = True
    chart_fill: bool = True
    autosave: bool = True

    @classmethod
    def from_dict(cls, data: object) -> "Preferences":
        if not isinstance(data, dict):
            return cls()
        palette = data.get("palette", "Графит")
        # Preferences written before v0.5 had no skin and used coloured presets.
        # Start upgraded installations with the requested neutral dark palette.
        if "skin" not in data and palette in ("Аврора", "Биосфера", "Ультрафиолет"):
            palette = "Графит"
        style = data.get("style", "Свечение")
        grid = data.get("grid", True)
        trails = data.get("trails", True)
        speed = data.get("speed", 8)
        from .skins import SKINS
        selected_skin = data.get("skin", "Графит / циан")
        animation = data.get("animation", True)
        overlay = data.get("overlay", True)
        chart_fill = data.get("chart_fill", True)
        autosave = data.get("autosave", True)
        return cls(palette=palette if isinstance(palette, str) and palette in PALETTES else "Графит",
                   style=style if isinstance(style, str) and style in STYLES else "Свечение",
                   grid=grid if type(grid) is bool else True,
                   trails=trails if type(trails) is bool else True,
                   speed=speed if type(speed) is int and 1 <= speed <= 30 else 8,
                   skin=selected_skin if isinstance(selected_skin, str) and selected_skin in SKINS else "Графит / циан",
                   animation=animation if type(animation) is bool else True,
                   overlay=overlay if type(overlay) is bool else True,
                   chart_fill=chart_fill if type(chart_fill) is bool else True,
                   autosave=autosave if type(autosave) is bool else True)

    def as_dict(self) -> dict[str, str | int | bool]:
        return {"palette": self.palette, "style": self.style, "grid": self.grid,
                "trails": self.trails, "speed": self.speed,
                "skin": self.skin, "animation": self.animation,
                "overlay": self.overlay, "chart_fill": self.chart_fill,
                "autosave": self.autosave}


def load_preferences(path: Path | None = None) -> Preferences:
    path = path or settings_path()
    try:
        return Preferences.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, UnicodeError):
        return Preferences()


def save_preferences(prefs: Preferences, path: Path | None = None) -> None:
    path = path or settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    try:
        temp.write_text(json.dumps(prefs.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
