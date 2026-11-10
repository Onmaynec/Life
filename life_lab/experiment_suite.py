"""Reproducible parameter sweeps for the Life research workbench.

This module deliberately has no UI dependencies. Each case is an independent
world and every variant in a repetition receives the same initial seed, so
comparisons do not accidentally mix parameter changes and initial conditions.
"""
from __future__ import annotations

import copy
import csv
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import statistics
from typing import Any

import numpy as np

from .ecology import EcologySettings
from .setup import WorldOptions, create_world
from .world import CellularAutomaton, Ecosystem


@dataclass(frozen=True)
class Parameter:
    key: str
    title: str
    mode: str
    minimum: float
    maximum: float
    unit: str = ""
    integer: bool = False
    setting: bool = False


PARAMETERS: dict[str, Parameter] = {
    "density": Parameter("density", "Плотность клеток", "classic", 0.0, 1.0, "%"),
    "plants": Parameter("plants", "Доля растений", "ecosystem", 0.0, 1.0, "%"),
    "herbivores": Parameter("herbivores", "Доля травоядных", "ecosystem", 0.0, 1.0, "%"),
    "predators": Parameter("predators", "Доля хищников", "ecosystem", 0.0, 1.0, "%"),
    "rain": Parameter("rain", "Количество осадков", "ecosystem", 0.0, 1.0, "", setting=True),
    "heat": Parameter("heat", "Средняя температура", "ecosystem", 0.0, 1.0, "", setting=True),
    "moisture_flow": Parameter("moisture_flow", "Перенос влаги", "ecosystem", 0.0, 0.50, "", setting=True),
    "mineral_flow": Parameter("mineral_flow", "Перенос минералов", "ecosystem", 0.0, 0.30, "", setting=True),
    "season_length": Parameter("season_length", "Длина сезона", "ecosystem", 30, 720, "поколений", integer=True, setting=True),
}

SUMMARY_COLUMNS = (
    "variant", "parameter", "value", "repetition", "seed", "start_generation",
    "steps", "initial_population", "final_population", "peak_population",
    "mean_population", "final_plants", "final_herbivores", "final_predators",
    "total_births", "total_deaths", "herbivore_extinction", "predator_extinction",
)
SERIES_COLUMNS = (
    "variant", "value", "repetition", "seed", "step", "generation",
    "population", "plants", "herbivores", "predators", "births", "deaths",
    "soil", "moisture", "mean_energy",
)


@dataclass(frozen=True)
class ExperimentPlan:
    mode: str = "classic"
    parameter: str = "density"
    minimum: float = 0.12
    maximum: float = 0.36
    variants: int = 4
    repetitions: int = 2
    steps: int = 120
    sample_every: int = 10
    seed: int = 42
    start_mode: str = "fresh"
    width: int = 80
    height: int = 60
    wrap: bool = False
    rule: str = "B3/S23"
    density: float = 0.22
    plants: float = 0.17
    herbivores: float = 0.018
    predators: float = 0.003
    ecology: dict[str, Any] = field(default_factory=dict)
    stop_on_extinction: bool = False

    @property
    def parameter_info(self) -> Parameter:
        return PARAMETERS[self.parameter]

    def values(self) -> tuple[float | int, ...]:
        spec = self.parameter_info
        values = np.linspace(self.minimum, self.maximum, self.variants)
        if spec.integer:
            return tuple(int(round(float(value))) for value in values)
        return tuple(float(round(value, 9)) for value in values)

    def validate(self, source: CellularAutomaton | Ecosystem | None = None) -> None:
        if self.mode not in ("classic", "ecosystem"):
            raise ValueError("Неизвестный режим опыта")
        if self.parameter not in PARAMETERS or PARAMETERS[self.parameter].mode != self.mode:
            raise ValueError("Этот параметр недоступен для выбранного режима")
        for key, low, high in (("variants", 2, 12), ("repetitions", 1, 12),
                               ("steps", 1, 1500), ("sample_every", 1, 500)):
            value = getattr(self, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"Неверное значение {key}: допустимо {low}–{high}")
        if self.variants * self.repetitions * self.steps > 16000:
            raise ValueError("Объём слишком велик: не более 16 000 шагов на серию")
        if type(self.seed) is not int or not 0 <= self.seed <= (2**64 - self.repetitions):
            raise ValueError("Seed должен быть неотрицательным 64-битным числом")
        spec = self.parameter_info
        for value in (self.minimum, self.maximum):
            if (type(value) not in (int, float) or not math.isfinite(value)
                    or not spec.minimum <= value <= spec.maximum):
                raise ValueError(f"{spec.title}: допустимо {spec.minimum}–{spec.maximum}")
        if self.minimum >= self.maximum or len(set(self.values())) != self.variants:
            raise ValueError("Начало диапазона должно быть меньше конца, значения не должны повторяться")
        if self.start_mode not in ("fresh", "snapshot"):
            raise ValueError("Неизвестный источник исходного мира")
        if type(self.stop_on_extinction) is not bool:
            raise ValueError("Неверный параметр остановки")
        if self.start_mode == "snapshot":
            if source is None or source.mode != self.mode:
                raise ValueError("Для опыта из текущего мира нужен мир того же режима")
            if not spec.setting:
                raise ValueError("В текущем мире можно менять только параметры среды")
        else:
            WorldOptions(mode=self.mode, width=self.width, height=self.height,
                         wrap=self.wrap, rule=self.rule, density=self.density,
                         plants=self.plants, herbivores=self.herbivores,
                         predators=self.predators, seed=self.seed).validate()
            if not (20 <= self.width <= 250 and 20 <= self.height <= 180):
                raise ValueError("Для серии новых миров размер ограничен 250×180")
            if not spec.setting and self.mode == "ecosystem":
                for value in self.values():
                    mix = {"plants": self.plants, "herbivores": self.herbivores,
                           "predators": self.predators}
                    mix[self.parameter] = value
                    if sum(mix.values()) > 1 + 1e-12:
                        raise ValueError("При выбранной плотности сумма видов превышает 100%")
        if self.mode == "ecosystem":
            EcologySettings.from_dict(self.ecology)
        elif self.ecology:
            raise ValueError("Настройки экосистемы нельзя применять в классическом режиме")


class ExperimentSession:
    """State machine: one step per tick, safe to drive with Tk.after()."""

    def __init__(self, plan: ExperimentPlan,
                 source: CellularAutomaton | Ecosystem | None = None):
        plan.validate(source)
        self.plan = plan
        self._source = copy.deepcopy(source) if plan.start_mode == "snapshot" else None
        self._values = plan.values()
        self.variant_index = 0
        self.repetition_index = 0
        self.world: CellularAutomaton | Ecosystem | None = None
        self.local_step = 0
        self.steps_done = 0
        self.completed_units = 0
        self.summary: list[dict[str, Any]] = []
        self.series: list[dict[str, Any]] = []
        self.cancelled = False
        self.complete = False
        self._start_population = 0
        self._births = 0
        self._deaths = 0
        self._peak = 0
        self._population_sum = 0
        self._interval_births = 0
        self._interval_deaths = 0
        self._first_herbivore_loss: int | None = None
        self._first_predator_loss: int | None = None
        self._initial_herbivores = 0
        self._initial_predators = 0
        self._begin_case()

    @property
    def total_steps(self) -> int:
        return self.plan.variants * self.plan.repetitions * self.plan.steps

    @property
    def progress(self) -> float:
        return self.completed_units / self.total_steps

    def _begin_case(self) -> None:
        p = self.plan
        value = self._values[self.variant_index]
        replica_seed = p.seed + self.repetition_index
        if self._source is not None:
            world = copy.deepcopy(self._source)
            if isinstance(world, Ecosystem):
                world._rng = np.random.default_rng(replica_seed)
        else:
            values = dict(mode=p.mode, width=p.width, height=p.height,
                          wrap=p.wrap, rule=p.rule, seed=replica_seed,
                          density=p.density, plants=p.plants,
                          herbivores=p.herbivores, predators=p.predators)
            if p.parameter in ("density", "plants", "herbivores", "predators"):
                values[p.parameter] = value
            world = create_world(WorldOptions(**values))
            if isinstance(world, Ecosystem):
                world.ecology = EcologySettings.from_dict(p.ecology)
        if isinstance(world, Ecosystem) and p.parameter_info.setting:
            world.ecology.enabled = True
            setattr(world.ecology, p.parameter, value)
        self.world = world
        self.local_step = 0
        self._births = 0
        self._deaths = 0
        self._interval_births = 0
        self._interval_deaths = 0
        self._peak = int(world.counts()["population"])
        self._start_population = self._peak
        self._population_sum = self._peak
        counts = world.counts()
        self._initial_herbivores = int(counts.get("herbivores", 0))
        self._initial_predators = int(counts.get("predators", 0))
        self._first_herbivore_loss = None
        self._first_predator_loss = None
        self._sample({"births": 0, "deaths": 0})

    def _sample(self, events: dict[str, Any]) -> None:
        assert self.world is not None
        world = self.world
        counts = world.counts()
        ecology = isinstance(world, Ecosystem)
        animals = world.cells >= 2 if ecology else None
        row = dict(variant=self.variant_index + 1, value=self._values[self.variant_index],
                   repetition=self.repetition_index + 1,
                   seed=self.plan.seed + self.repetition_index,
                   step=self.local_step, generation=world.generation,
                   population=int(counts["population"]),
                   plants=int(counts.get("plants", 0)),
                   herbivores=int(counts.get("herbivores", 0)),
                   predators=int(counts.get("predators", 0)),
                   births=int(events.get("births", 0)), deaths=int(events.get("deaths", 0)),
                   soil=float(world.nutrients.mean()) if ecology else None,
                   moisture=float(world.moisture.mean()) if ecology else None,
                   mean_energy=(float(world.energy[animals].mean()) if animals.any() else None)
                   if ecology else None)
        self.series.append(row)

    def advance(self) -> bool:
        """Advance one generation; return False when the session is finished."""
        if self.complete or self.cancelled:
            return False
        assert self.world is not None
        events = self.world.step()
        self.local_step += 1
        self.steps_done += 1
        self.completed_units += 1
        counts = self.world.counts()
        pop = int(counts["population"])
        self._peak = max(self._peak, pop)
        self._population_sum += pop
        births = int(events.get("births", 0))
        deaths = int(events.get("deaths", 0))
        self._births += births
        self._deaths += deaths
        self._interval_births += births
        self._interval_deaths += deaths
        if self._initial_herbivores and counts.get("herbivores", 0) == 0 and self._first_herbivore_loss is None:
            self._first_herbivore_loss = self.local_step
        if self._initial_predators and counts.get("predators", 0) == 0 and self._first_predator_loss is None:
            self._first_predator_loss = self.local_step
        lost_species = (self._first_herbivore_loss is not None or self._first_predator_loss is not None)
        finished = self.local_step >= self.plan.steps or (self.plan.stop_on_extinction and lost_species)
        if self.local_step % self.plan.sample_every == 0 or finished:
            self._sample({"births": self._interval_births, "deaths": self._interval_deaths})
            self._interval_births = 0
            self._interval_deaths = 0
        if finished:
            self.completed_units += self.plan.steps - self.local_step
            self._finish_case(counts)
        return not self.complete

    def _finish_case(self, counts: dict[str, Any]) -> None:
        assert self.world is not None
        self.summary.append({
            "variant": self.variant_index + 1,
            "parameter": self.plan.parameter,
            "value": self._values[self.variant_index],
            "repetition": self.repetition_index + 1,
            "seed": self.plan.seed + self.repetition_index,
            "start_generation": self.world.generation - self.local_step,
            "steps": self.local_step,
            "initial_population": self._start_population,
            "final_population": int(counts["population"]),
            "peak_population": self._peak,
            "mean_population": round(self._population_sum / (self.local_step + 1), 3),
            "final_plants": int(counts.get("plants", 0)),
            "final_herbivores": int(counts.get("herbivores", 0)),
            "final_predators": int(counts.get("predators", 0)),
            "total_births": self._births,
            "total_deaths": self._deaths,
            "herbivore_extinction": self._first_herbivore_loss,
            "predator_extinction": self._first_predator_loss,
        })
        self.repetition_index += 1
        if self.repetition_index >= self.plan.repetitions:
            self.variant_index += 1
            self.repetition_index = 0
        if self.variant_index >= self.plan.variants:
            self.complete = True
            self.world = None
        else:
            self._begin_case()

    def cancel(self) -> None:
        self.cancelled = True
        self.world = None

    def comparison(self, metric: str = "final_population") -> list[dict[str, float | int | None]]:
        if metric not in SUMMARY_COLUMNS or metric not in (
                "final_population", "mean_population", "peak_population", "total_births", "total_deaths",
                "final_plants", "final_herbivores", "final_predators"):
            raise ValueError("Показатель недоступен для сравнения")
        by_rep = {(r["variant"], r["repetition"]): r for r in self.summary}
        output = []
        for variant, value in enumerate(self._values, 1):
            records = [r for r in self.summary if r["variant"] == variant]
            if not records:
                continue
            values = [float(r[metric]) for r in records]
            paired = [float(r[metric] - by_rep[(1,r["repetition"])][metric])
                      for r in records if (1,r["repetition"]) in by_rep]
            output.append({"variant": variant, "value": value, "n": len(values),
                           "mean": statistics.mean(values),
                           "std": statistics.stdev(values) if len(values) >= 2 else None,
                           "paired_delta": statistics.mean(paired) if paired else None})
        return output

    def export_summary_csv(self, filename: str | Path) -> None:
        self._export_csv(filename, SUMMARY_COLUMNS, self.summary)

    def export_series_csv(self, filename: str | Path) -> None:
        self._export_csv(filename, SERIES_COLUMNS, self.series)

    @staticmethod
    def _export_csv(filename: str | Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
        with open(filename, "w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def export_report(self, filename: str | Path) -> None:
        """A human-readable record of the setup and final population statistics."""
        p = self.plan
        lines = [
            "# Life — серия экспериментов", "",
            f"Режим: `{p.mode}`  ", f"Параметр: {p.parameter_info.title} (`{p.parameter}`)  ",
            f"Исходные условия: {p.start_mode}  ",
            f"Диапазон: {p.minimum:g} — {p.maximum:g} ({p.variants} варианта)  ",
            f"Повторы: {p.repetitions}; seed: {p.seed} + номер повтора − 1  ",
            f"Длина: {p.steps} поколений; запись каждые {p.sample_every}  ",
            f"Статус: {'завершено' if self.complete else 'остановлено'}", "",
            "## Конечная численность", "",
            "Вариант | Значение | Повторов | Среднее | σ | Разница к первому варианту",
            "--- | ---: | ---: | ---: | ---: | ---:",
        ]
        for row in self.comparison():
            std = f"{row['std']:.2f}" if row['std'] is not None else "—"
            delta = f"{row['paired_delta']:+.2f}" if row['paired_delta'] is not None else "—"
            lines.append(f"{row['variant']} | {row['value']:g} | {row['n']} | {row['mean']:.2f} | {std} | {delta}")
        lines.extend(["", "## Методика", "",
            "Каждый вариант использует одинаковую последовательность базовых seed для повторов. "
            "Сравнение выполняется для одной изменяемой переменной, остальные настройки сохраняются. "
            "При выборе копии текущего мира сохраняется исходное состояние клеток, но RNG инициализируется "
            "seed конкретного повтора.", "",
            "σ — выборочное стандартное отклонение; при одном повторе не вычисляется. "
            "Разница — средняя попарная разность к первому варианту с тем же номером повтора. "
            "Малое число повторов не даёт надёжного статистического вывода. "
            "Life использует упрощённую экспериментальную модель, а не реальный биологический прогноз.", ""])
        Path(filename).write_text("\n".join(lines), encoding="utf-8")

    def export_json(self, filename: str | Path) -> None:
        document = {"format": "life.experiment_suite", "version": 1,
                    "plan": asdict(self.plan), "completed": self.complete,
                    "cancelled": self.cancelled,
                    "summary": self.summary, "series": self.series}
        with open(filename, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2, allow_nan=False)


def run_experiment(plan: ExperimentPlan,
                   source: CellularAutomaton | Ecosystem | None = None) -> ExperimentSession:
    """Synchronous runner for tests, scripts and reproducibility checks."""
    session = ExperimentSession(plan, source)
    while not session.complete:
        session.advance()
    return session
