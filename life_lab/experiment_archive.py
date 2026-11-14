"""Validated, read-only experiment archives and cross-session comparisons.

JSON experiment exports are reports, not .lifelab world files. No untrusted
archive is executed, and failed imports never change the open collection.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

from .experiment_suite import ExperimentPlan, PARAMETERS, SERIES_COLUMNS, SUMMARY_COLUMNS

MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_ROWS = 100_000
_METRICS = ("population", "plants", "herbivores", "predators", "soil", "moisture", "mean_energy")


def _numeric(value: Any, nullable: bool = False) -> bool:
    return (nullable and value is None) or (type(value) in (int, float) and math.isfinite(value))


@dataclass(frozen=True)
class ImportedStudy:
    label: str
    plan: ExperimentPlan
    complete: bool
    cancelled: bool
    summary: tuple[dict[str, Any], ...]
    series: tuple[dict[str, Any], ...]

    @property
    def completed_runs(self) -> int:
        return len(self.summary)

    def compatible_with(self, other: "ImportedStudy") -> tuple[bool, str]:
        p, q = self.plan, other.plan
        # The changed parameter and studied range may differ intentionally.
        for key in ("mode", "start_mode", "width", "height", "wrap", "rule",
                    "steps", "repetitions", "sample_every"):
            if getattr(p, key) != getattr(q, key):
                return False, f"Различается {key}"
        for key in ("density", "plants", "herbivores", "predators", "ecology"):
            if getattr(p, key) != getattr(q, key) and key != p.parameter and key != q.parameter:
                return False, f"Различается {key}"
        return True, "Сопоставимые исходные условия (seed и диапазон могут различаться)"

    def curve(self, metric: str, variant: int = 1) -> list[tuple[int, float]]:
        """Mean over repetitions of one variant; never mix different parameter values."""
        if metric not in _METRICS:
            raise ValueError("Неизвестная метрика")
        if type(variant) is not int or not 1 <= variant <= self.plan.variants:
            raise ValueError("Номер варианта выходит за пределы серии")
        grouped: dict[int, list[float]] = {}
        for row in self.series:
            value = row[metric]
            if row["variant"] == variant and _numeric(value):
                grouped.setdefault(row["step"], []).append(float(value))
        return [(step, sum(v) / len(v)) for step, v in sorted(grouped.items())]


def import_study(path: str | Path) -> ImportedStudy:
    file = Path(path)
    if file.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("Архив эксперимента слишком велик")
    try:
        document = json.loads(file.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Повреждённый JSON эксперимента") from exc
    if not isinstance(document, dict) or document.get("format") != "life.experiment_suite" or type(document.get("version")) is not int or document["version"] != 1:
        raise ValueError("Неизвестный формат или версия эксперимента")
    if type(document.get("completed")) is not bool or type(document.get("cancelled")) is not bool:
        raise ValueError("Повреждённый статус эксперимента")
    params = document.get("plan")
    if not isinstance(params, dict) or set(params) != set(ExperimentPlan.__dataclass_fields__):
        raise ValueError("Неверный план эксперимента")
    try:
        plan = ExperimentPlan(**params)
        spec = PARAMETERS[plan.parameter]
        if plan.mode not in ("classic", "ecosystem") or spec.mode != plan.mode:
            raise ValueError("Неверный режим")
        if (type(plan.variants) is not int or not 2 <= plan.variants <= 12
                or type(plan.repetitions) is not int or not 1 <= plan.repetitions <= 12
                or type(plan.steps) is not int or not 1 <= plan.steps <= 1500
                or type(plan.sample_every) is not int or not 1 <= plan.sample_every <= 500
                or type(plan.seed) is not int or not 0 <= plan.seed < 2**64
                or plan.start_mode not in ("fresh", "snapshot")
                or not isinstance(plan.ecology, dict)
                or not all(_numeric(v) for v in (plan.minimum, plan.maximum))
                or not spec.minimum <= plan.minimum < plan.maximum <= spec.maximum):
            raise ValueError("Параметры плана вне допустимого диапазона")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Неверный план эксперимента") from exc
    rows = {}
    for name, columns in (("summary", SUMMARY_COLUMNS), ("series", SERIES_COLUMNS)):
        data = document.get(name)
        if not isinstance(data, list) or len(data) > MAX_ROWS:
            raise ValueError(f"Повреждённые данные {name}")
        clean = []
        for row in data:
            if not isinstance(row, dict) or set(row) != set(columns):
                raise ValueError(f"Неверные колонки {name}")
            if not all(_numeric(value, nullable=(key in ("herbivore_extinction", "predator_extinction", "mean_energy", "soil", "moisture")))
                       if key != "parameter" else type(value) is str and value == plan.parameter
                       for key, value in row.items()):
                raise ValueError(f"Неверные значения {name}")
            variant, repeat = row["variant"], row["repetition"]
            if (type(variant) is not int or type(repeat) is not int
                    or not 1 <= variant <= plan.variants or not 1 <= repeat <= plan.repetitions
                    or type(row["seed"]) is not int or row["seed"] != plan.seed + repeat - 1
                    or (name == "series" and (type(row["step"]) is not int or not 0 <= row["step"] <= plan.steps))):
                raise ValueError(f"Неверный индекс {name}")
            clean.append(dict(row))
        rows[name] = tuple(clean)
    keys = [(r["variant"], r["repetition"]) for r in rows["summary"]]
    if len(keys) != len(set(keys)):
        raise ValueError("Повторяются результаты запуска")
    sample_keys = [(r["variant"], r["repetition"], r["step"]) for r in rows["series"]]
    if len(sample_keys) != len(set(sample_keys)):
        raise ValueError("Повторяются точки измерения")
    if document["completed"] and len(keys) != plan.variants * plan.repetitions:
        raise ValueError("Завершённая серия содержит не все повторы")
    return ImportedStudy(file.name, plan, document["completed"], document["cancelled"], rows["summary"], rows["series"])


class StudyCollection:
    """Atomic imports, at most 16 independently saved experiments."""
    def __init__(self) -> None:
        self.studies: list[ImportedStudy] = []

    def add(self, path: str | Path) -> ImportedStudy:
        if len(self.studies) >= 16:
            raise ValueError("Открыто слишком много серий (максимум 16)")
        study = import_study(path)  # validate before mutating collection
        self.studies.append(study)
        return study

    def rows(self, metric: str) -> list[dict[str, Any]]:
        if metric not in _METRICS:
            raise ValueError("Неизвестная метрика")
        return [{"source": study.label, "mode": study.plan.mode,
                 "parameter": study.plan.parameter, "variant": row["variant"],
                 "repetition": row["repetition"], "seed": row["seed"],
                 "step": row["step"], "generation": row["generation"],
                 "metric": metric, "value": row[metric]}
                for study in self.studies for row in study.series]

    def export_csv(self, path: str | Path, metric: str) -> None:
        cols = ("source", "mode", "parameter", "variant", "repetition", "seed", "step", "generation", "metric", "value")
        with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=cols)
            writer.writeheader()
            writer.writerows(self.rows(metric))
