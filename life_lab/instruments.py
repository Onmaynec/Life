"""Deterministic measurements and spatial data for Life's research workspace."""
from __future__ import annotations

from collections import deque
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from .world import Ecosystem, neighbour_counts


MEASURES = (
    "generation", "population", "occupancy", "plants", "herbivores", "predators",
    "births", "deaths", "net_growth", "diversity", "entropy", "mean_neighbours",
    "lonely", "crowded", "clusters", "largest_cluster", "cluster_generation", "edge_contacts",
    "centroid_x", "centroid_y", "dispersion", "soil", "moisture", "energy",
    "herbivore_energy", "predator_energy", "mean_age", "max_age",
    "mean_metabolism", "mean_sight", "mean_fertility", "sunlight", "temperature",
    "plant_biomass", "animal_biomass", "predator_prey_ratio", "birth_rate", "death_rate",
    "organic", "toxins", "seed_bank", "altitude",
    "families", "lineage_depth", "mean_lineage_depth", "pedigree_records",
)


def _clusters(mask: np.ndarray, wrap: bool = False) -> tuple[int, int]:
    """4-connected components using the same edge topology as the simulation."""
    h, w = mask.shape
    points = np.argwhere(mask)
    seen = np.zeros(mask.shape, dtype=bool)
    count = largest = 0
    for yy, xx in points:
        y, x = int(yy), int(xx)
        if seen[y, x]:
            continue
        count += 1
        group = 0
        pending = deque([(y, x)])
        seen[y, x] = True
        while pending:
            cy, cx = pending.pop()
            group += 1
            for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                if wrap:
                    ny, nx = ny % h, nx % w
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny,nx]:
                    seen[ny,nx] = True
                    pending.append((ny,nx))
        largest = max(largest, group)
    return count, largest


def measure(world: Any, events: dict[str, Any] | None = None,
            *, include_clusters: bool = True) -> dict[str, float | int | None]:
    """Independent read-only measurements. Empty populations use null for averages."""
    cells = world.cells
    mask = cells != 0
    pop = int(mask.sum())
    h,w = cells.shape
    counts = np.bincount(cells.ravel(), minlength=4)
    kinds = counts[1:4] if isinstance(world, Ecosystem) else counts[1:2]
    shares = kinds[kinds > 0].astype(float) / max(1,pop)
    entropy = -float(np.sum(shares * np.log2(shares))) if len(shares) else 0.0
    neighbours = neighbour_counts(mask, world.wrap)
    yx = np.argwhere(mask)
    cy, cx = (float(yx[:,0].mean()),float(yx[:,1].mean())) if pop else (0.0,0.0)
    spread = float(np.sqrt(np.mean((yx[:,0]-cy)**2 + (yx[:,1]-cx)**2))) if pop else 0.0
    components, largest = _clusters(mask, world.wrap) if include_clusters else (None, None)
    e = events or {}
    rec: dict[str,float | int | None] = {
        "generation": world.generation, "population": pop,
        "occupancy": pop / (h*w) * 100, "plants": int(counts[1]),
        "herbivores": int(counts[2]) if isinstance(world,Ecosystem) else 0,
        "predators": int(counts[3]) if isinstance(world,Ecosystem) else 0,
        "births": int(e.get("births",0)), "deaths": int(e.get("deaths",0)),
        "net_growth": int(e.get("births",0))-int(e.get("deaths",0)),
        "diversity": int(np.count_nonzero(kinds)), "entropy": entropy,
        "mean_neighbours": float(neighbours[mask].mean()) if pop else 0.0,
        "lonely": int(np.count_nonzero(mask & (neighbours < 2))),
        "crowded": int(np.count_nonzero(mask & (neighbours > 3))),
        "clusters": components, "largest_cluster": largest,
        "cluster_generation": world.generation if include_clusters else None,
        "edge_contacts": int(mask[0,:].sum()+mask[-1,:].sum()+mask[:,0].sum()+mask[:,-1].sum()),
        "centroid_x": cx, "centroid_y": cy, "dispersion": spread,
        "soil": None, "moisture": None, "energy": None,
        "organic": None, "toxins": None, "seed_bank": None, "altitude": None,
        "herbivore_energy": None, "predator_energy": None,
        "mean_age": None, "max_age": None, "mean_metabolism": None,
        "mean_sight": None, "mean_fertility": None,
        "sunlight": None, "temperature": None,
        "families": None, "lineage_depth": None,
        "mean_lineage_depth": None, "pedigree_records": None,
        "plant_biomass": int(counts[1]), "animal_biomass": 0,
        "predator_prey_ratio": None, "birth_rate": 100 * int(e.get("births",0))/max(1,pop),
        "death_rate": 100 * int(e.get("deaths",0))/max(1,pop),
    }
    if isinstance(world,Ecosystem):
        pedigree = world.lineage.summary()
        rec.update(families=pedigree["families"], lineage_depth=pedigree["max_depth"],
                   mean_lineage_depth=pedigree["mean_depth"],
                   pedigree_records=pedigree["recorded"])
        animals = cells >= 2
        herb = cells == 2
        pred = cells == 3
        rec.update(soil=float(world.nutrients.mean()),
                   moisture=float(world.moisture.mean()),
                   organic=float(world.organic.mean()),
                   toxins=float(world.toxins.mean()),
                   seed_bank=int(np.count_nonzero(world.seeds)),
                   altitude=float(world.altitude.mean()),
                   energy=float(world.energy[animals].mean()) if animals.any() else None,
                   herbivore_energy=float(world.energy[herb].mean()) if herb.any() else None,
                   predator_energy=float(world.energy[pred].mean()) if pred.any() else None,
                   mean_age=float(world.age[mask].mean()) if pop else None,
                   max_age=int(world.age.max()),
                   mean_metabolism=float(world.trait[animals].mean()) if animals.any() else None,
                   mean_sight=float(world.sight[animals].mean()) if animals.any() else None,
                   mean_fertility=float(world.fertility[animals].mean()) if animals.any() else None,
                   animal_biomass=float(world.energy[animals].sum()),
                   predator_prey_ratio=float(pred.sum()/max(1,herb.sum())))
        if world.ecology.enabled:
            from .ecology import climate_values
            climate=climate_values(world.generation,world.ecology)
            rec["sunlight"] = float(climate["sunlight"])
            rec["temperature"] = float(climate["temperature"])
    return rec


SPATIAL_UNITS = {
    "Плотность": "доля поля", "Типы клеток": "код вида", "Соседи": "соседей",
    "Энергия": "ед. энергии", "Минералы": "ед. минералов", "Влага": "%",
    "Возраст": "поколений", "Метаболизм": "коэффициент", "Зрение": "клеток",
    "Плодовитость": "коэффициент", "Семена": "количество", "Органика": "ед.",
    "Токсины": "ед.", "Рельеф": "высота", "Глубина родословной": "поколений",
}


def spatial_field(world: Any, name: str, *, max_side: int = 50,
                  aggregation: str = "mean") -> np.ndarray:
    """Read-only spatial aggregation; preserve all source cells and output bounds.

    ``mean`` is useful for concentrations, ``max`` exposes rare local peaks,
    ``sum`` shows absolute quantities per tile. Edge tiles may be smaller.
    """
    if type(max_side) is not int or not 2 <= max_side <= 200:
        raise ValueError("Детализация должна быть от 2 до 200")
    if aggregation not in ("mean", "max", "sum"):
        raise ValueError("Неизвестный тип агрегации")
    # Select a single source layer. Constructing every layer is expensive on
    # large worlds, especially the neighbourhood count and pedigree lookup.
    if name == "Плотность":
        source = world.cells > 0
    elif name == "Типы клеток":
        source = world.cells
    elif name == "Соседи":
        source = neighbour_counts(world.cells > 0, world.wrap)
    elif not isinstance(world, Ecosystem):
        raise ValueError("Неизвестный пространственный показатель")
    elif name == "Глубина родословной":
        source = np.zeros_like(world.cells, dtype=np.float32)
        for yy, xx in np.argwhere(world.cells >= 2):
            record = world.lineage.get(int(yy), int(xx))
            if record is not None:
                source[yy, xx] = record.depth
    elif name in ("Метаболизм", "Зрение", "Плодовитость"):
        attribute = {"Метаболизм": "trait", "Зрение": "sight",
                     "Плодовитость": "fertility"}[name]
        source = np.where(world.cells >= 2, getattr(world, attribute), 0)
    else:
        attribute = {"Энергия": "energy", "Минералы": "nutrients",
                     "Влага": "moisture", "Возраст": "age", "Семена": "seeds",
                     "Органика": "organic", "Токсины": "toxins",
                     "Рельеф": "altitude"}.get(name)
        if attribute is None:
            raise ValueError("Неизвестный пространственный показатель")
        source = getattr(world, attribute)
    arr = np.asarray(source, dtype=np.float32)
    h, w = arr.shape
    sy, sx = max(1, math.ceil(h / max_side)), max(1, math.ceil(w / max_side))
    # Reduce each tile in NumPy. `reduceat` keeps the smaller edge tiles
    # instead of padding with zeros (which would skew averages).
    starts_y = np.arange(0, h, sy)
    starts_x = np.arange(0, w, sx)
    if aggregation == "max":
        return np.maximum.reduceat(np.maximum.reduceat(arr, starts_y, axis=0), starts_x, axis=1)
    total = np.add.reduceat(np.add.reduceat(arr, starts_y, axis=0), starts_x, axis=1)
    if aggregation == "sum":
        return total
    sizes_y = np.diff(np.append(starts_y, h))
    sizes_x = np.diff(np.append(starts_x, w))
    return (total / (sizes_y[:, None] * sizes_x[None, :])).astype(np.float32, copy=False)



class ResearchLog:
    """Bounded measurement history independent of the stable v0.3 CSV schema."""
    def __init__(self, records: list[dict[str,float | int | None]] | None = None):
        self.records = list(records or [])[-2000:]

    def capture(self, world: Any, events: dict[str,Any] | None = None) -> None:
        deep = world.generation % 5 == 0 or not self.records
        data = measure(world, events, include_clusters=deep)
        if self.records and self.records[-1]["generation"] == data["generation"]:
            self.records[-1]=data
        else:
            self.records.append(data)
        if len(self.records)>2000:
            del self.records[:-2000]

    def export_csv(self,path: str | Path) -> None:
        with open(path,"w",newline="",encoding="utf-8-sig") as stream:
            writer=csv.DictWriter(stream,fieldnames=MEASURES)
            writer.writeheader()
            writer.writerows(self.records)

    @classmethod
    def from_records(cls, data: Any) -> "ResearchLog":
        if not isinstance(data,list) or len(data)>2000:
            raise ValueError("Повреждённый исследовательский журнал")
        prev = -1
        normalized: list[dict[str, float | int | None]] = []
        legacy_cluster_schema = bool(data) and isinstance(data[0], dict) and "cluster_generation" not in data[0]
        for entry in data:
            if not isinstance(entry,dict):
                raise ValueError("Неизвестные показатели исследования")
            entry = dict(entry)
            old_keys = {"families", "lineage_depth", "mean_lineage_depth",
                        "pedigree_records", "cluster_generation"}
            if not set(entry).issubset(MEASURES) or (set(MEASURES)-set(entry)) - old_keys:
                raise ValueError("Неизвестные показатели исследования")
            for key in old_keys:
                entry.setdefault(key, None)
            g = entry.get("generation")
            if type(g) is not int or g <= prev:
                raise ValueError("Неверные поколения")
            # v1.0.0 copied previous cluster values into unsampled generations.
            if legacy_cluster_schema:
                entry["cluster_generation"] = g if g % 5 == 0 else None
            if entry["cluster_generation"] is not None:
                if type(entry["cluster_generation"]) is not int or entry["cluster_generation"] > g:
                    raise ValueError("Неверное поколение измерения кластеров")
            else:
                entry["clusters"] = None
                entry["largest_cluster"] = None
            prev = g
            for value in entry.values():
                if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                    raise ValueError("Неверные измерения")
            normalized.append(entry)
        return cls(normalized)
