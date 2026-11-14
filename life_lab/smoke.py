"""Packaged Windows distribution smoke checks (also runnable from source).

Run Life.exe --smoke-test. Creates all temporary artifacts outside the project
and its installed distribution; no user settings or active worlds are modified.
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from .patterns import encode_rle, decode_rle
from .preferences import Preferences, load_preferences, save_preferences
from .instruments import ResearchLog
from .storage import load_world, save_world
from .world import CellularAutomaton, Ecosystem


def run_smoke() -> None:
    with tempfile.TemporaryDirectory(prefix="life-smoke-") as folder:
        root = Path(folder)
        world = Ecosystem(32, 24, seed=17)
        world.randomize(0.2, seed=17)
        world.step()
        save_world(world, root/"world.lifelab", research=ResearchLog())
        loaded = load_world(root/"world.lifelab")
        assert np.array_equal(loaded.cells, world.cells)
        assert loaded.generation == world.generation
        assert np.array_equal(loaded.lineage.ids,world.lineage.ids)
        classic = CellularAutomaton(30, 20)
        classic.paint(4,4)
        text = encode_rle(classic.cells)
        grid, _rule = decode_rle(text)
        assert np.count_nonzero(grid) == 1
        Image.fromarray(np.uint8(world.cells*70),"L").save(root/"preview.png")
        assert (root/"preview.png").stat().st_size > 0
        log = ResearchLog()
        log.capture(world)
        log.export_csv(root/"metrics.csv")
        assert (root/"metrics.csv").stat().st_size > 100
        prefs = Preferences()
        save_preferences(prefs, root/"prefs.json")
        assert load_preferences(root/"prefs.json") == prefs


if __name__ == "__main__":
    run_smoke()
