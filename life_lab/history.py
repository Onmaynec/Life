"""Experiment measurements and CSV export; no GUI dependencies."""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from .world import CellularAutomaton, Ecosystem


MAX_RECORDS = 10000
GENES = ("metabolism", "sight", "fertility")
COLUMNS = (
    "generation", "population", "plants", "herbivores", "predators",
    "births", "deaths", "total_births", "total_deaths", "mutations",
    "herbivores_metabolism", "herbivores_sight", "herbivores_fertility",
    "predators_metabolism", "predators_sight", "predators_fertility",
)


class ExperimentHistory:
    """The last 10,000 measured generations, including the initial state."""

    def __init__(self, records: list[dict[str, int | float | None]] | None = None):
        self.records: list[dict[str, int | float | None]] = list(records or [])[-MAX_RECORDS:]

    @classmethod
    def from_records(cls, records: Any) -> "ExperimentHistory":
        if not isinstance(records, list) or len(records) > MAX_RECORDS:
            raise ValueError("Повреждена история эксперимента")
        cleaned = []
        last_generation = -1
        for record in records:
            if not isinstance(record, dict) or set(record) != set(COLUMNS):
                raise ValueError("Неверные столбцы истории")
            generation = record["generation"]
            if type(generation) is not int or generation < 0 or generation <= last_generation:
                raise ValueError("Неверный порядок поколений")
            last_generation = generation
            for column, value in record.items():
                if value is None and column.startswith(("herbivores_", "predators_")):
                    continue
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"Неверное значение истории: {column}")
            cleaned.append(dict(record))
        return cls(cleaned)

    def record(self, world: CellularAutomaton | Ecosystem, events: dict[str, int] | None = None) -> None:
        counts = world.counts()
        record: dict[str, int | float | None] = {
            "generation": world.generation,
            "population": counts["population"],
            "plants": counts.get("plants", 0),
            "herbivores": counts.get("herbivores", 0),
            "predators": counts.get("predators", 0),
            "births": (events or {}).get("births", 0),
            "deaths": (events or {}).get("deaths", 0),
            "total_births": world.total_births,
            "total_deaths": world.total_deaths,
            "mutations": counts.get("mutations", 0),
        }
        for species in ("herbivores", "predators"):
            for gene in GENES:
                record[f"{species}_{gene}"] = None
        if isinstance(world, Ecosystem):
            for species, kind in (("herbivores", 2), ("predators", 3)):
                mask = world.cells == kind
                if mask.any():
                    for gene, array in (("metabolism", world.trait),
                                        ("sight", world.sight), ("fertility", world.fertility)):
                        record[f"{species}_{gene}"] = float(array[mask].mean())
        if self.records and self.records[-1]["generation"] == world.generation:
            self.records[-1] = record
        else:
            self.records.append(record)
            if len(self.records) > MAX_RECORDS:
                del self.records[:len(self.records) - MAX_RECORDS]

    @classmethod
    def start(cls, world: CellularAutomaton | Ecosystem) -> "ExperimentHistory":
        result = cls()
        result.record(world)
        return result

    def export_csv(self, path: str | Path) -> None:
        """Excel-readable CSV; missing species averages are empty fields."""
        with open(path, "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.records)
