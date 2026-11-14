"""Read-only comparisons of living genetic lineages and spatial trait maps."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from .world import Ecosystem

TRAITS = {"Метаболизм": "trait", "Зрение": "sight", "Плодовитость": "fertility"}
TRAIT_RANGES = {"metabolism": (0.65, 1.45), "sight": (1.0, 3.0), "fertility": (0.70, 1.35)}


def living_families(world: Ecosystem) -> dict[int, dict[str, Any]]:
    """One pass over live IDs, with averages computed only for extant animals."""
    if not isinstance(world, Ecosystem):
        raise ValueError("Родословные доступны только в экосистеме")
    result: dict[int, dict[str, Any]] = {}
    for yy, xx in np.argwhere(world.lineage.ids):
        y, x = int(yy), int(xx)
        rec = world.lineage.get(y,x)
        if rec is None:
            continue
        group = result.setdefault(rec.founder, {"founder":rec.founder, "count":0,
                "offspring":0, "metabolism":[], "sight":[], "fertility":[]})
        group["count"] += 1
        group["offspring"] += rec.offspring
        for name, arr in (("metabolism", world.trait), ("sight", world.sight),
                          ("fertility", world.fertility)):
            group[name].append(float(arr[y,x]))
    return result


def compare_families(world: Ecosystem, first: int, second: int) -> list[dict[str, Any]]:
    if type(first) is not int or type(second) is not int or first <= 0 or second <= 0:
        raise ValueError("Нужны ID двух основателей")
    groups = living_families(world)
    # Complete known archive, not only the living subset. Pruning can remove
    # old ancestors, so this is explicitly "known children", not full history.
    descendants: dict[int,int] = {}
    for rec in world.lineage.records.values():
        if rec.parent:
            descendants[rec.founder] = descendants.get(rec.founder,0) + 1
    result = []
    for uid in (first, second):
        group = groups.get(uid)
        row = {"generation":world.generation, "founder":uid,
               "living":group["count"] if group else 0,
               "offspring":descendants.get(uid,0)}
        for trait in ("metabolism", "sight", "fertility"):
            row[trait] = float(np.mean(group[trait])) if group else None
            row[trait+"_distribution"] = tuple(np.histogram(group[trait], bins=8, range=TRAIT_RANGES[trait])[0].tolist()) if group else tuple([0]*8)
        result.append(row)
    return result


def history(world: Ecosystem, founders: tuple[int, int], every: int = 1) -> list[dict[str, int]]:
    """An O(records + generations) census reconstructed from birth/death events.

    Very old ancestors may be pruned from the bounded archive. Counted history
    is then explicitly a partial census rather than an invented full lineage.
    """
    if type(every) is not int or every < 1:
        raise ValueError("Шаг должен быть положительным")
    if not isinstance(world, Ecosystem):
        raise ValueError("Нужен мир экосистемы")
    last = world.generation
    unique = set(founders)
    changes = {founder: {} for founder in unique}
    for rec in world.lineage.records.values():
        if rec.founder not in unique:
            continue
        delta = changes[rec.founder]
        delta[rec.born] = delta.get(rec.born, 0) + 1
        if rec.died is not None:
            delta[rec.died] = delta.get(rec.died, 0) - 1
    ordered = {f: sorted(events.items()) for f, events in changes.items()}
    counts = {f: 0 for f in unique}
    positions = {f: 0 for f in unique}
    rows = []
    for generation in sorted(set(list(range(0, last+1, every)) + [last])):
        for founder in founders:
            events = ordered[founder]
            while positions[founder] < len(events) and events[positions[founder]][0] <= generation:
                counts[founder] += events[positions[founder]][1]
                positions[founder] += 1
            rows.append({"generation":generation, "founder":founder, "living":counts[founder]})
    return rows


def trait_map(world: Ecosystem, trait: str, founder: int | None = None) -> np.ndarray:
    """A NaN means no animal of the chosen lineage; source arrays are immutable."""
    if trait not in TRAITS:
        raise ValueError("Неизвестный наследуемый признак")
    output = np.full(world.cells.shape, np.nan, dtype=np.float32)
    if founder is None:
        mask = world.cells >= 2
    else:
        mask = np.zeros(world.cells.shape, dtype=bool)
        for yy, xx in np.argwhere(world.lineage.ids):
            rec = world.lineage.get(int(yy), int(xx))
            if rec is not None and rec.founder == founder:
                mask[yy,xx] = True
    output[mask] = getattr(world,TRAITS[trait])[mask]
    return output


def export_family_history(path: str | Path, world: Ecosystem, first: int, second: int) -> None:
    rows = history(world,(first,second))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=("generation","founder","living"))
        writer.writeheader()
        writer.writerows(rows)
