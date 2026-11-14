"""Read-only chronological alerts derived from a research measurement journal.

Alerts are observations, not simulation events: when measurements are sparse,
we only know that a threshold was crossed *by* the sampled generation.
No world RNG is accessed and no history is mutated.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
import csv
import math


_THRESHOLDS = {"low": 0.55, "normal": 0.35, "high": 0.20}
_SPECIES = {"plants": "Растения", "herbivores": "Травоядные", "predators": "Хищники"}


@dataclass(frozen=True)
class TimelineEvent:
    generation: int
    category: str
    severity: str
    title: str
    description: str


def _number(row: Mapping[str, object], key: str) -> float | None:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def events_between(previous: Mapping[str, object], current: Mapping[str, object],
                   sensitivity: str = "normal") -> list[TimelineEvent]:
    """Compute changes between actual samples, without interpolation.

    A missing measurement is unknown, not zero. The input order is checked
    because exported or restored journals can contain irregular intervals.
    """
    if sensitivity not in _THRESHOLDS:
        raise ValueError("Неизвестная чувствительность")
    before, after = _number(previous, "generation"), _number(current, "generation")
    if before is None or after is None or before < 0 or after <= before:
        raise ValueError("Поколения должны возрастать")
    if not before.is_integer() or not after.is_integer():
        raise ValueError("Номер поколения должен быть целым")
    generation = int(after)
    result: list[TimelineEvent] = []

    def add(category: str, severity: str, title: str, description: str) -> None:
        result.append(TimelineEvent(generation, category, severity, title, description))

    p0, p1 = _number(previous, "population"), _number(current, "population")
    if p0 is not None and p1 is not None and p0 >= 20 and p1 >= 0:
        change = (p1 - p0) / p0
        if change <= -_THRESHOLDS[sensitivity]:
            add("Популяция", "Критично" if change <= -0.5 else "Внимание",
                "Резкое сокращение", f"Численность: {int(p0)} → {int(p1)} ({change:+.0%})")
        elif change >= _THRESHOLDS[sensitivity]:
            add("Популяция", "Информация", "Быстрый рост",
                f"Численность: {int(p0)} → {int(p1)} ({change:+.0%})")

    for key, label in _SPECIES.items():
        n0, n1 = _number(previous, key), _number(current, key)
        if n0 is None or n1 is None or min(n0, n1) < 0:
            continue
        if n0 > 0 and n1 == 0:
            add("Виды", "Критично", f"Исчезновение: {label}",
                f"Последнее измерение: {int(n0)} особей; теперь 0")
        elif n0 == 0 and n1 > 0:
            add("Виды", "Информация", f"Появление: {label}",
                f"В измерении обнаружено {int(n1)} особей")

    deaths = _number(current, "deaths")
    previous_deaths = _number(previous, "deaths")
    if (deaths is not None and p0 is not None and p0 >= 20
            and deaths >= max(10, p0 * 0.25)
            and (previous_deaths is None or previous_deaths < max(10, p0 * 0.25))):
        add("Популяция", "Внимание", "Высокая смертность",
            f"За шаг записано {int(deaths)} смертей")

    for key, title, limit, unit in (("moisture", "Низкая влажность", 20, "%"),
                                     ("soil", "Истощение минералов", 15, " ед.")):
        old, new = _number(previous, key), _number(current, key)
        if old is not None and new is not None and old >= limit > new:
            add("Ресурсы", "Внимание", title, f"{old:.1f} → {new:.1f}{unit}")

    f0, f1 = _number(previous, "families"), _number(current, "families")
    if f0 is not None and f1 is not None and f0 >= 5 and f1 >= f0 * 1.5:
        add("Родословные", "Информация", "Расширение семей",
            f"Активных линий: {int(f0)} → {int(f1)}")

    d0, d1 = _number(previous, "occupancy"), _number(current, "occupancy")
    if d0 is not None and d1 is not None and d0 < 75 <= d1:
        add("Популяция", "Внимание", "Высокая плотность",
            f"Заполнение поля достигло {d1:.1f}%")
    return result


def scan_timeline(records: Sequence[Mapping[str, object]],
                  sensitivity: str = "normal") -> list[TimelineEvent]:
    """Rebuild a bounded event view from up to 2,000 research rows."""
    if sensitivity not in _THRESHOLDS:
        raise ValueError("Неизвестная чувствительность")
    if len(records) > 2000:
        raise ValueError("Слишком длинный журнал")
    result: list[TimelineEvent] = []
    for previous, current in zip(records, records[1:]):
        result.extend(events_between(previous, current, sensitivity))
    return result


def export_timeline_csv(path: str | Path, events: Sequence[TimelineEvent]) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(("generation", "category", "severity", "title", "description"))
        for event in events:
            writer.writerow((event.generation, event.category, event.severity,
                             event.title, event.description))
