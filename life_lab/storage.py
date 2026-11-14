"""Save/load LIFE//LAB worlds in a compressed .lifelab container."""
from __future__ import annotations

import json
import os
import tempfile
from zipfile import ZipFile, BadZipFile
from pathlib import Path
from typing import Any

import numpy as np

from .world import CellularAutomaton, Ecosystem
from .history import ExperimentHistory
from .instruments import ResearchLog
from .ecology import EcologySettings
from .lineage import Pedigree


FORMAT_VERSION = 5
# Limits also apply to malformed/malicious NPZ archives, not only decoded worlds.
# Valid 700x500 ecosystem saves with genealogy fit comfortably below these.
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
MAX_MEMBER_BYTES = 256 * 1024 * 1024
MAX_EXPANDED_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 48


def _check_archive_size(path: str | Path) -> None:
    """Reject oversized NPZ inputs before NumPy decompresses any member."""
    source = Path(path)
    if source.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Файл мира превышает допустимый размер")
    try:
        with ZipFile(source, "r") as archive:
            members = archive.infolist()
            if not members or len(members) > MAX_ARCHIVE_MEMBERS:
                raise ValueError("Неправильное количество элементов сохранения")
            if any(item.file_size > MAX_MEMBER_BYTES or item.file_size < 0
                   for item in members):
                raise ValueError("Слишком большой элемент сохранения")
            if sum(item.file_size for item in members) > MAX_EXPANDED_BYTES:
                raise ValueError("Превышен лимит распакованных данных")
            if len({item.filename for item in members}) != len(members):
                raise ValueError("Повторяющиеся элементы сохранения")
    except BadZipFile as error:
        raise ValueError("Некорректный архив сохранения") from error


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
        meta["lineage"] = world.lineage.to_dict()
        arrays["animal_ids"] = world.lineage.ids
    if history is not None:
        arrays["history"] = np.array(json.dumps(history.records, ensure_ascii=False))
    if research is not None:
        arrays["research"] = np.array(json.dumps(research.records, ensure_ascii=False))
    # Write a complete archive beside the destination before replacing it.
    # A failed write must never truncate an earlier valid world.
    target = Path(path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=target.parent,
                                         prefix=f".{target.name}.", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, metadata=np.array(json.dumps(meta)), **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_document(path: str | Path) -> tuple[CellularAutomaton | Ecosystem, ExperimentHistory]:
    """Read the simulation and any recorded measurements from a .lifelab file."""
    _check_archive_size(path)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"].item()))
        if metadata.get("version") not in (1, 2, 3, 4, FORMAT_VERSION):
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
            if metadata["version"] >= 5:
                world.lineage = Pedigree.from_dict(archive["animal_ids"], metadata["lineage"], cells)
        else:
            raise ValueError("Неизвестный режим симуляции")
        world.cells[:] = cells
        world.generation = max(0, int(metadata.get("generation", 0)))
        if isinstance(world, Ecosystem) and metadata["version"] < 5:
            world.rebuild_lineage()
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
    _check_archive_size(path)
    with np.load(path, allow_pickle=False) as archive:
        if "research" not in archive:
            return ResearchLog()
        records = json.loads(str(archive["research"].item()))
        return ResearchLog.from_records(records)
