"""Interactive packaging check: Tk, research widgets and a paired scenario.

This is not a Windows 10/11 compatibility certificate. The JSON report records
which runner actually hosted the UI test, including Tk and display geometry.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import tempfile
import time

from .history import ExperimentHistory
from .instruments import ResearchLog
from .scenarios import Intervention
from .world import Ecosystem


def run_ui_smoke(out: str | Path | None = None) -> dict[str, object]:
    previous = {key: os.environ.get(key) for key in
                ("APPDATA", "LOCALAPPDATA", "XDG_CONFIG_HOME", "XDG_STATE_HOME")}
    with tempfile.TemporaryDirectory(prefix="life-ui-acceptance-") as temp:
        for key in previous:
            os.environ[key] = temp
        app = None
        try:
            from .app import LifeLabApp
            app = LifeLabApp()
            app.root.geometry("1000x690")
            app.root.update()
            world = Ecosystem(30, 22, seed=31)
            world.randomize(seed=31)
            log = ResearchLog()
            log.capture(world)
            app._install_world(world, ExperimentHistory.start(world), log)
            app.open_scenario_studio()
            studio = app.scenario_window
            if studio is None:
                raise AssertionError("Окно сценариев не открылось")
            studio.window.geometry("990x680+15+20")
            studio.duration.set("9")
            studio.interval.set("3")
            studio.pending = [Intervention(0, "moisture", -22)]
            studio._refresh_events()
            studio.start()
            studio.metric.set("Влажность")
            deadline = time.monotonic() + 35
            while studio.session is not None and not studio.session.done:
                app.root.update()
                time.sleep(0.012)
                if time.monotonic() > deadline:
                    raise TimeoutError("Графический сценарий не завершился")
            app.root.update()
            if studio.session is None or studio.session.completed != 9:
                raise AssertionError("Графический сценарий не дал ожидаемый результат")
            if app.world.generation != 0:
                raise AssertionError("Сценарий изменил исходный мир")
            report: dict[str, object] = {
                "python": platform.python_version(), "system": platform.system(),
                "release": platform.release(), "version": platform.version(),
                "tk": str(app.root.tk.call("info", "patchlevel")),
                "screen": [app.root.winfo_screenwidth(), app.root.winfo_screenheight()],
                "dpi": round(float(app.screen_dpi), 2),
                "title": app.root.title(), "scenarios_steps": studio.session.completed,
                "scenario_source_generation": app.world.generation,
                "pass": True,
            }
            if out:
                directory = Path(out)
                directory.mkdir(parents=True, exist_ok=True)
                # Attempt an actual screenshot; the runner may have no capturable desktop.
                try:
                    from PIL import ImageGrab
                    studio.window.lift()
                    app.root.update()
                    x, y = studio.window.winfo_rootx(), studio.window.winfo_rooty()
                    w, h = studio.window.winfo_width(), studio.window.winfo_height()
                    ImageGrab.grab(bbox=(x, y, x+w, y+h)).save(directory/"scenario-studio.png")
                    report["screenshot"] = "scenario-studio.png"
                except (OSError, RuntimeError, ValueError) as error:
                    report["screenshot_error"] = str(error)
                (directory/"ui-report.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")
            return report
        finally:
            if app is not None and app.root.winfo_exists():
                try:
                    if app.scenario_window and app.scenario_window.window.winfo_exists():
                        app.scenario_window.close()
                    app.root.destroy()
                except Exception:
                    pass
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
