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
    "lonely", "crowded", "clusters", "largest_cluster", "edge_contacts",
    "centroid_x", "centroid_y", "dispersion", "soil", "moisture", "energy",
    "herbivore_energy", "predator_energy", "mean_age", "max_age",
    "mean_metabolism", "mean_sight", "mean_fertility", "sunlight", "temperature",
    "plant_biomass", "animal_biomass", "predator_prey_ratio", "birth_rate", "death_rate",
    "organic", "toxins", "seed_bank", "altitude",
)


def _clusters(mask: np.ndarray) -> tuple[int, int]:
    """4-connected component count and largest component, bounded local traversal."""
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
    components, largest = _clusters(mask) if include_clusters else (None, None)
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
        "edge_contacts": int(mask[0,:].sum()+mask[-1,:].sum()+mask[:,0].sum()+mask[:,-1].sum()),
        "centroid_x": cx, "centroid_y": cy, "dispersion": spread,
        "soil": None, "moisture": None, "energy": None,
        "organic": None, "toxins": None, "seed_bank": None, "altitude": None,
        "herbivore_energy": None, "predator_energy": None,
        "mean_age": None, "max_age": None, "mean_metabolism": None,
        "mean_sight": None, "mean_fertility": None,
        "sunlight": None, "temperature": None,
        "plant_biomass": int(counts[1]), "animal_biomass": 0,
        "predator_prey_ratio": None, "birth_rate": 100 * int(e.get("births",0))/max(1,pop),
        "death_rate": 100 * int(e.get("deaths",0))/max(1,pop),
    }
    if isinstance(world,Ecosystem):
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


def spatial_field(world: Any, name: str, *, max_side: int = 50) -> np.ndarray:
    """Downsample bounded 2D source data for plotting without altering the world."""
    fields = {"Плотность": (world.cells > 0).astype(np.float32),
              "Типы клеток": world.cells.astype(np.float32),
              "Соседи": neighbour_counts(world.cells > 0,world.wrap).astype(np.float32)}
    if isinstance(world,Ecosystem):
        fields.update({"Энергия":world.energy,"Минералы":world.nutrients,
                       "Влага":world.moisture,"Возраст":world.age.astype(np.float32),
                       "Семена":world.seeds.astype(np.float32),
                       "Органика":world.organic,"Токсины":world.toxins,
                       "Рельеф":world.altitude,
                       "Метаболизм":np.where(world.cells >= 2,world.trait,0),
                       "Зрение":np.where(world.cells >= 2,world.sight,0),
                       "Плодовитость":np.where(world.cells >= 2,world.fertility,0)})
    if name not in fields:
        raise ValueError("Неизвестный пространственный показатель")
    arr=fields[name]
    # Average over tiles (rather than picking one pixel); preserves spatial structure.
    h,w=arr.shape
    sy,sx=max(1, math.ceil(h/max_side)),max(1, math.ceil(w/max_side))
    H,W=math.ceil(h/sy),math.ceil(w/sx)
    out=np.empty((H,W),dtype=np.float32)
    for y in range(H):
        for x in range(W):
            out[y,x]=float(arr[y*sy:min(h,(y+1)*sy),x*sx:min(w,(x+1)*sx)].mean())
    return out


class ResearchLog:
    """Bounded measurement history independent of the stable v0.3 CSV schema."""
    def __init__(self, records: list[dict[str,float | int | None]] | None = None):
        self.records = list(records or [])[-2000:]

    def capture(self, world: Any, events: dict[str,Any] | None = None) -> None:
        deep = world.generation % 5 == 0 or not self.records
        data=measure(world,events,include_clusters=deep)
        if not deep and self.records:
            data["clusters"] = self.records[-1].get("clusters")
            data["largest_cluster"] = self.records[-1].get("largest_cluster")
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
        prev=-1
        for entry in data:
            if not isinstance(entry,dict) or set(entry) != set(MEASURES):
                raise ValueError("Неизвестные показатели исследования")
            g=entry.get("generation")
            if type(g) is not int or g <= prev:
                raise ValueError("Неверные поколения")
            prev=g
            for value in entry.values():
                if value is not None and (type(value) not in (int,float) or not math.isfinite(value)):
                    raise ValueError("Неверные измерения")
        return cls(data)
