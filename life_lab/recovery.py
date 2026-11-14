"""Local recovery checkpoints for long-running Life sessions.

The automatic checkpoint also preserves the latest cleanly closed session.
An explicit restore point is retained until the user overwrites it. Both use
exactly the normal .lifelab format; no simulation-specific fork of serialization.
"""
from __future__ import annotations

from datetime import datetime
import os
from pathlib import Path
import sys

from .history import ExperimentHistory
from .instruments import ResearchLog
from .storage import load_document, load_research, save_world
from .world import CellularAutomaton, Ecosystem


def state_directory() -> Path:
    if sys.platform == "win32":
        base = Path(os.getenv("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    else:
        base = Path(os.getenv("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return base / "Life" / "recovery"


class RecoveryStore:
    def __init__(self, directory: Path | None = None) -> None:
        self.directory = Path(directory) if directory is not None else state_directory()

    def file(self, *, manual: bool = False) -> Path:
        return self.directory / ("checkpoint.lifelab" if manual else "autosave.lifelab")

    def available(self, *, manual: bool = False) -> bool:
        path = self.file(manual=manual)
        return path.is_file() and path.stat().st_size > 0

    def timestamp(self, *, manual: bool = False) -> str:
        path = self.file(manual=manual)
        return datetime.fromtimestamp(path.stat().st_mtime).strftime("%d.%m.%Y %H:%M")

    def save(self, world: CellularAutomaton | Ecosystem,
             history: ExperimentHistory, research: ResearchLog,
             *, manual: bool = False) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.file(manual=manual)
        save_world(world, path, history, research)
        return path

    def restore(self, *, manual: bool = False) -> tuple[CellularAutomaton | Ecosystem, ExperimentHistory, ResearchLog]:
        path = self.file(manual=manual)
        # Finish validating every component before callers replace the live world.
        world, history = load_document(path)
        research = load_research(path)
        return world, history, research

    def discard(self, *, manual: bool = False) -> None:
        self.file(manual=manual).unlink(missing_ok=True)
