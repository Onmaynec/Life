"""Save/load LIFE//LAB worlds in a compressed .lifelab container."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .world import CellularAutomaton, Ecosystem


FORMAT_VERSION = 1


def save_world(world: CellularAutomaton | Ecosystem, path: str | Path) -> None:
    meta: dict[str, Any] = {
        "version": FORMAT_VERSION,
        "mode": world.mode,
        "generation": world.generation,
        "wrap": world.wrap,
        "width": world.width,
        "height": world.height,
        "total_births": world.total_births,
        "total_deaths": world.total_deaths,
    }
    arrays: dict[str, Any] = {"cells": world.cells}
    if isinstance(world, CellularAutomaton):
        meta["rule"] = world.rule
    else:
        arrays.update(energy=world.energy, trait=world.trait, nutrients=world.nutrients)
        meta["mutations"] = world.mutations
        meta["rng_state"] = world._rng.bit_generator.state
    with open(path, "wb") as f:
        np.savez_compressed(f, metadata=np.array(json.dumps(meta)), **arrays)


def load_world(path: str | Path) -> CellularAutomaton | Ecosystem:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("version") != FORMAT_VERSION:
            raise ValueError("Неподдерживаемая версия сохранения")
        w, h = int(metadata["width"]), int(metadata["height"])
        CellularAutomaton._validate_dimensions(w, h)
        cells = archive["cells"]
        if cells.shape != (h, w) or not np.isin(cells, [0, 1, 2, 3]).all():
            raise ValueError("Повреждённое состояние клеток")
        mode = metadata.get("mode")
        if mode == "classic":
            if np.max(cells) > 1:
                raise ValueError("В классическом режиме клетка должна быть 0 или 1")
            world: CellularAutomaton | Ecosystem = CellularAutomaton(
                w, h, metadata.get("rule", "B3/S23"), bool(metadata.get("wrap", False)))
        elif mode == "ecosystem":
            world = Ecosystem(w, h, wrap=bool(metadata.get("wrap", False)))
            for key in ("energy", "trait", "nutrients"):
                arr = archive[key]
                if arr.shape != cells.shape or not np.isfinite(arr).all():
                    raise ValueError(f"Повреждённый массив: {key}")
                getattr(world, key)[:] = arr
            world.mutations = int(metadata.get("mutations", 0))
            if "rng_state" in metadata:
                world._rng.bit_generator.state = metadata["rng_state"]
        else:
            raise ValueError("Неизвестный режим симуляции")
        world.cells[:] = cells
        world.generation = max(0, int(metadata.get("generation", 0)))
        world.total_births = max(0, int(metadata.get("total_births", 0)))
        world.total_deaths = max(0, int(metadata.get("total_deaths", 0)))
        return world
