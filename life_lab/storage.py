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

_ARRAY_LIMITS = {
    "energy": (-10.0, 100.0, False),
    "trait": (0.65, 1.45, False),
    "sight": (1.0, 3.0, False),
    "fertility": (0.70, 1.35, False),
    "nutrients": (0.0, 100.0, False),
    "moisture": (0.0, 100.0, False),
    "age": (0, 65535, True),
    "seeds": (0, 255, True),
    "organic": (0.0, 100.0, False),
    "toxins": (0.0, 100.0, False),
    "altitude": (0.0, 1.0, False),
}


def _restore_array(archive: Any, world: Ecosystem, key: str) -> None:
    """Validate an NPZ layer before casting it into the model's fixed dtype."""
    try:
        arr = archive[key]
    except KeyError as error:
        raise ValueError(f"В сохранении отсутствует массив: {key}") from error
    target = getattr(world, key)
    low, high, integer = _ARRAY_LIMITS[key]
    if (arr.shape != target.shape or arr.dtype.kind not in "iuf"
            or not np.isfinite(arr).all()):
        raise ValueError(f"Повреждённый массив: {key}")
    if integer and arr.dtype.kind not in "iu":
        raise ValueError(f"Массив {key} должен содержать целые значения")
    if np.any(arr < low) or np.any(arr > high):
        raise ValueError(f"Значения массива {key} выходят за допустимый диапазон")
    converted = arr.astype(target.dtype, copy=False)
    if target.dtype.kind == "f" and not np.isfinite(converted).all():
        raise ValueError(f"Переполнение массива {key} при загрузке")
    target[:] = converted


def _metadata_counter(metadata: dict[str, Any], key: str) -> int:
    value = metadata.get(key, 0)
    if type(value) is not int or value < 0:
        raise ValueError(f"Некорректный счётчик сохранения: {key}")
    return value


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
    """Read a world only after validating every layer and its destination dtype."""
    _check_archive_size(path)
    with np.load(path, allow_pickle=False) as archive:
        try:
            metadata = json.loads(str(archive["metadata"].item()))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError("Повреждённые метаданные сохранения") from error
        if not isinstance(metadata, dict):
            raise ValueError("Метаданные сохранения должны быть объектом")
        version = metadata.get("version")
        if type(version) is not int or version not in (1, 2, 3, 4, FORMAT_VERSION):
            raise ValueError("Неподдерживаемая версия сохранения")
        w, h = metadata.get("width"), metadata.get("height")
        if type(w) is not int or type(h) is not int:
            raise ValueError("Размеры мира должны быть целыми числами")
        CellularAutomaton._validate_dimensions(w, h)
        wrap = metadata.get("wrap", False)
        if type(wrap) is not bool:
            raise ValueError("Неверный параметр границ мира")
        try:
            cells = archive["cells"]
        except KeyError as error:
            raise ValueError("В сохранении отсутствует поле клеток") from error
        if (cells.shape != (h, w) or cells.dtype.kind not in "iu"
                or not np.isin(cells, [0, 1, 2, 3]).all()):
            raise ValueError("Повреждённое состояние клеток")
        mode = metadata.get("mode")
        if mode == "classic":
            if np.max(cells) > 1:
                raise ValueError("В классическом режиме клетка должна быть 0 или 1")
            rule = metadata.get("rule", "B3/S23")
            if not isinstance(rule, str):
                raise ValueError("Неверное правило классического автомата")
            world: CellularAutomaton | Ecosystem = CellularAutomaton(w, h, rule, wrap)
        elif mode == "ecosystem":
            world = Ecosystem(w, h, wrap=wrap)
            for key in ("energy", "trait", "nutrients"):
                _restore_array(archive, world, key)
            if version >= 2:
                for key in ("sight", "fertility"):
                    _restore_array(archive, world, key)
            if version >= 4:
                for key in ("moisture", "age", "seeds", "organic", "toxins", "altitude"):
                    _restore_array(archive, world, key)
                world.ecology = EcologySettings.from_dict(metadata.get("ecology", {}))
            world.mutations = _metadata_counter(metadata, "mutations")
            if "rng_state" in metadata:
                try:
                    world._rng.bit_generator.state = metadata["rng_state"]
                except (TypeError, ValueError, KeyError) as error:
                    raise ValueError("Повреждённое состояние генератора случайных чисел") from error
            if version >= 5:
                try:
                    world.lineage = Pedigree.from_dict(
                        archive["animal_ids"], metadata.get("lineage"), cells)
                except (KeyError, TypeError, ValueError) as error:
                    raise ValueError("Повреждённая родословная сохранения") from error
        else:
            raise ValueError("Неизвестный режим симуляции")
        world.cells[:] = cells
        world.generation = _metadata_counter(metadata, "generation")
        if isinstance(world, Ecosystem) and version < 5:
            world.rebuild_lineage()
        world.total_births = _metadata_counter(metadata, "total_births")
        world.total_deaths = _metadata_counter(metadata, "total_deaths")
        if version >= 3 and "history" in archive:
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
