"""Save/load LIFE//LAB worlds in a compressed .lifelab container."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .world import CellularAutomaton, Ecosystem
from .history import ExperimentHistory
from .instruments import ResearchLog
from .ecology import EcologySettings


FORMAT_VERSION = 4


def save_world(world: CellularAutomaton | Ecosystem, path: str | Path,
               history: ExperimentHistory | None = None,
               research: ResearchLog | None = None) -> None:
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
        arrays.update(energy=world.energy, trait=world.trait, sight=world.sight,
                      fertility=world.fertility, nutrients=world.nutrients,
                      moisture=world.moisture, age=world.age,
                      seeds=world.seeds, organic=world.organic, toxins=world.toxins,
                      altitude=world.altitude)
        meta["ecology"] = world.ecology.as_dict()
        meta["mutations"] = world.mutations
        meta["rng_state"] = world._rng.bit_generator.state
    if history is not None:
        arrays["history"] = np.array(json.dumps(history.records, ensure_ascii=False))
    if research is not None:
        arrays["research"] = np.array(json.dumps(research.records, ensure_ascii=False))
    with open(path, "wb") as f:
        np.savez_compressed(f, metadata=np.array(json.dumps(meta)), **arrays)


def load_document(path: str | Path) -> tuple[CellularAutomaton | Ecosystem, ExperimentHistory]:
    """Read the simulation and any recorded measurements from a .lifelab file."""
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("version") not in (1, 2, 3, FORMAT_VERSION):
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
            if metadata["version"] >= 2:
                for key in ("sight", "fertility"):
                    arr = archive[key]
                    if arr.shape != cells.shape or not np.isfinite(arr).all():
                        raise ValueError(f"Повреждённый массив: {key}")
                    getattr(world, key)[:] = arr
            if metadata["version"] >= 4:
                for key in ("moisture", "age", "seeds", "organic", "toxins", "altitude"):
                    arr = archive[key]
                    if arr.shape != cells.shape or not np.isfinite(arr).all():
                        raise ValueError(f"Повреждённый массив: {key}")
                    getattr(world, key)[:] = arr
                world.ecology = EcologySettings.from_dict(metadata.get("ecology", {}))
            world.mutations = int(metadata.get("mutations", 0))
            if "rng_state" in metadata:
                world._rng.bit_generator.state = metadata["rng_state"]
        else:
            raise ValueError("Неизвестный режим симуляции")
        world.cells[:] = cells
        world.generation = max(0, int(metadata.get("generation", 0)))
        world.total_births = max(0, int(metadata.get("total_births", 0)))
        world.total_deaths = max(0, int(metadata.get("total_deaths", 0)))
        if metadata["version"] >= 3 and "history" in archive:
            history = ExperimentHistory.from_records(json.loads(str(archive["history"].item())))
            if history.records and history.records[-1]["generation"] > world.generation:
                raise ValueError("История содержит поколения из будущего")
        else:
            history = ExperimentHistory.start(world)
        return world, history


def load_world(path: str | Path) -> CellularAutomaton | Ecosystem:
    """Compatibility API for callers only interested in simulation state."""
    return load_document(path)[0]


def load_research(path: str | Path) -> ResearchLog:
    """Load optional v0.5 research history; earlier worlds start with an empty journal."""
    with np.load(path, allow_pickle=False) as archive:
        if "research" not in archive:
            return ResearchLog()
        records = json.loads(str(archive["research"].item()))
        return ResearchLog.from_records(records)
