"""Controlled ecosystem interventions, with a paired untouched reference run.

Every scenario starts from two deep copies of the same world, including its RNG
state and genealogy. The live Life document is never stepped or modified.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import csv
import json
import math
import os
from pathlib import Path
import hashlib
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED
from typing import Any

import numpy as np

from .world import Ecosystem


INTERVENTION_KINDS = {
    "moisture": "Изменение влажности, ед.",
    "nutrients": "Изменение минералов, ед.",
    "toxins": "Изменение токсинов, ед.",
    "rain": "Установить интенсивность осадков, 0–1",
    "heat": "Установить температуру среды, 0–1",
    "disease": "Болезни: 0 — выключить, 1 — включить",
    "wildfire": "Пожары: 0 — выключить, 1 — включить",
    "water_flow": "Установить перенос влаги, 0–1",
}

CLIMATE_KEYS = frozenset({"rain", "heat", "disease", "wildfire", "water_flow"})


@dataclass(frozen=True)
class Intervention:
    at: int
    kind: str
    value: float

    def validate(self, steps: int, *, ecology_enabled: bool) -> None:
        if type(self.at) is not int or not 0 <= self.at < steps:
            raise ValueError("Поколение вмешательства выходит за пределы опыта")
        if self.kind not in INTERVENTION_KINDS:
            raise ValueError("Неизвестное вмешательство")
        if type(self.value) not in (int, float) or not math.isfinite(self.value):
            raise ValueError("Значение воздействия должно быть конечным числом")
        if self.kind in CLIMATE_KEYS and not ecology_enabled:
            raise ValueError("Для климатического воздействия включите модель среды")
        if self.kind in CLIMATE_KEYS and not 0 <= self.value <= 1:
            raise ValueError("Для параметра среды нужен диапазон от 0 до 1")
        if self.kind in ("disease", "wildfire") and self.value not in (0, 1):
            raise ValueError("Переключатель должен быть равен 0 или 1")
        if self.kind not in CLIMATE_KEYS and not -100 <= self.value <= 100:
            raise ValueError("Разовое изменение ресурса: от -100 до 100")


@dataclass(frozen=True)
class ScenarioPlan:
    steps: int = 120
    sample_every: int = 5
    interventions: tuple[Intervention, ...] = ()
    label: str = "Сценарий"

    def validate(self, world: Ecosystem) -> None:
        if not isinstance(world, Ecosystem):
            raise ValueError("Сценарии доступны только для режима экосистемы")
        if type(self.steps) is not int or not 1 <= self.steps <= 1000:
            raise ValueError("Продолжительность: 1–1000 поколений")
        if type(self.sample_every) is not int or not 1 <= self.sample_every <= self.steps:
            raise ValueError("Интервал замеров должен быть от 1 до длительности")
        if not isinstance(self.label, str) or not self.label.strip() or len(self.label) > 100:
            raise ValueError("Название сценария: 1–100 символов")
        if not isinstance(self.interventions, tuple) or not 1 <= len(self.interventions) <= 32:
            raise ValueError("Выберите от 1 до 32 воздействий")
        for event in self.interventions:
            if not isinstance(event, Intervention):
                raise ValueError("Неверный тип воздействия")
            event.validate(self.steps, ecology_enabled=world.ecology.enabled)


PRESETS: dict[str, tuple[Intervention, ...]] = {
    "Засуха": (Intervention(0, "moisture", -35), Intervention(30, "moisture", -20)),
    "Ливни": (Intervention(0, "moisture", 30), Intervention(40, "moisture", 20)),
    "Истощение почвы": (Intervention(0, "nutrients", -30), Intervention(45, "nutrients", -20)),
    "Восстановление почвы": (Intervention(0, "nutrients", 25), Intervention(45, "nutrients", 15)),
    "Загрязнение": (Intervention(0, "toxins", 35), Intervention(35, "toxins", 20)),
    "Очистка": (Intervention(0, "toxins", -30), Intervention(35, "toxins", -20)),
}


def _apply(world: Ecosystem, intervention: Intervention) -> None:
    if intervention.kind in ("moisture", "nutrients", "toxins"):
        arr = getattr(world, intervention.kind)
        # No new random draws: the intervention only changes the treatment branch.
        np.clip(arr + np.float32(intervention.value), 0, 100, out=arr)
    elif intervention.kind == "rain":
        world.ecology.rain = float(intervention.value)
    elif intervention.kind == "heat":
        world.ecology.heat = float(intervention.value)
    elif intervention.kind == "disease":
        world.ecology.disease = bool(intervention.value)
    elif intervention.kind == "wildfire":
        world.ecology.wildfire = bool(intervention.value)
    elif intervention.kind == "water_flow":
        world.ecology.moisture_flow = float(intervention.value)
    else:
        raise ValueError("Неизвестное воздействие")


def _observe(world: Ecosystem, step: int, branch: str) -> dict[str, int | float | str | None]:
    counts = np.bincount(world.cells.ravel(), minlength=4)
    animals = world.cells >= 2
    return {
        "branch": branch,
        "step": step,
        "generation": world.generation,
        "population": int(counts[1:].sum()),
        "plants": int(counts[1]),
        "herbivores": int(counts[2]),
        "predators": int(counts[3]),
        "soil": float(world.nutrients.mean()),
        "moisture": float(world.moisture.mean()),
        "toxins": float(world.toxins.mean()),
        "animal_energy": float(world.energy[animals].mean()) if np.any(animals) else None,
        "births": int(world.total_births),
        "deaths": int(world.total_deaths),
        "mutations": int(world.mutations),
    }


METRICS = {
    "Численность": "population", "Растения": "plants",
    "Травоядные": "herbivores", "Хищники": "predators",
    "Минералы": "soil", "Влажность": "moisture",
    "Токсины": "toxins", "Энергия животных": "animal_energy",
}


class ScenarioSession:
    """Incremental paired study; advance() processes one generation for each copy."""

    def __init__(self, world: Ecosystem, plan: ScenarioPlan):
        plan.validate(world)
        self.plan = plan
        self.initial = copy.deepcopy(world)
        self.reference = copy.deepcopy(world)
        self.treatment = copy.deepcopy(world)
        self.start_generation = world.generation
        self.completed = 0
        self.samples: list[dict[str, int | float | str | None]] = [
            _observe(self.reference, 0, "control"),
            _observe(self.treatment, 0, "scenario"),
        ]
        self.fired: list[Intervention] = []

    @property
    def done(self) -> bool:
        return self.completed >= self.plan.steps

    def advance(self) -> bool:
        if self.done:
            return False
        # One event can be accompanied by others at the same relative generation.
        for change in self.plan.interventions:
            if change.at == self.completed:
                _apply(self.treatment, change)
                self.fired.append(change)
        self.reference.step()
        self.treatment.step()
        self.completed += 1
        if self.completed % self.plan.sample_every == 0 or self.done:
            self.samples.extend((_observe(self.reference, self.completed, "control"),
                                 _observe(self.treatment, self.completed, "scenario")))
        return True

    def finish(self) -> "ScenarioSession":
        while self.advance():
            pass
        return self

    def curves(self, metric: str) -> dict[str, list[tuple[int, float]]]:
        if metric not in METRICS.values():
            raise ValueError("Неизвестный показатель")
        series: dict[str, list[tuple[int, float]]] = {"control": [], "scenario": []}
        for row in self.samples:
            value = row[metric]
            if value is not None:
                series[str(row["branch"])].append((int(row["step"]), float(value)))
        return series

    def summary(self) -> dict[str, Any]:
        final = {str(row["branch"]): row for row in self.samples[-2:]}
        if len(final) != 2:
            raise ValueError("В опыте отсутствует контрольная группа")
        control = final["control"]
        treatment = final["scenario"]
        return {
            "completed": self.completed,
            "control_population": control["population"],
            "scenario_population": treatment["population"],
            "population_difference": int(treatment["population"]) - int(control["population"]),
            "control_soil": control["soil"],
            "scenario_soil": treatment["soil"],
            "applied_events": len(self.fired),
        }

    def export_csv(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.samples[0]))
            writer.writeheader()
            writer.writerows(self.samples)

    def export_json(self, path: str | Path) -> None:
        document = {
            "format": "life-scenario-1", "label": self.plan.label,
            "start_generation": self.start_generation,
            "plan": {"steps": self.plan.steps, "sample_every": self.plan.sample_every,
                     "interventions": [asdict(e) for e in self.plan.interventions]},
            "completed": self.completed, "applied_events": [asdict(e) for e in self.fired],
            "summary": self.summary(), "samples": self.samples,
        }
        Path(path).write_text(json.dumps(document, ensure_ascii=False, indent=2,
                                        allow_nan=False) + "\n", encoding="utf-8")

    def export_bundle(self, path: str | Path) -> None:
        """Portable experiment: source/branches, samples and checksums in one ZIP.

        Files can be extracted and the .lifelab documents opened independently in
        Life. The running application is never modified by exporting.
        """
        from .storage import save_world
        target = Path(path)
        with tempfile.TemporaryDirectory(prefix=".life-scenario-", dir=target.parent) as folder:
            root = Path(folder)
            for name, world in (("initial", self.initial), ("control", self.reference),
                                ("scenario", self.treatment)):
                save_world(world, root / f"{name}.lifelab")
            self.export_csv(root / "measurements.csv")
            self.export_json(root / "report.json")
            manifest = {
                "format": "life-scenario-bundle-1",
                "version": 1,
                "files": {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                          for f in sorted(root.iterdir()) if f.is_file()},
            }
            (root / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
            staged = root / "study.tmp"
            with ZipFile(staged, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
                for file in sorted(root.iterdir()):
                    if file.is_file() and file != staged:
                        archive.write(file, file.name)
            os.replace(staged, target)


BUNDLE_FILES = frozenset({"initial.lifelab", "control.lifelab", "scenario.lifelab",
                          "measurements.csv", "report.json", "manifest.json"})


def verify_bundle(path: str | Path) -> dict[str, Any]:
    """Validate metadata and SHA-256 of every member without extracting paths."""
    from zipfile import BadZipFile
    try:
        with ZipFile(path, "r") as archive:
            items = archive.infolist()
            if {i.filename for i in items} != BUNDLE_FILES or len(items) != len(BUNDLE_FILES):
                raise ValueError("Архив сценария содержит неполный или неожиданный набор файлов")
            if sum(i.file_size for i in items) > 512*1024*1024 or any(
                i.file_size > 256*1024*1024 for i in items):
                raise ValueError("Архив сценария превышает допустимый размер")
            manifest = json.loads(archive.read("manifest.json"))
            if (not isinstance(manifest,dict) or
                manifest.get("format") != "life-scenario-bundle-1" or
                manifest.get("version") != 1 or
                not isinstance(manifest.get("files"),dict) or
                set(manifest["files"]) != BUNDLE_FILES - {"manifest.json"}):
                raise ValueError("Неверный манифест исследования")
            for name, expected in manifest["files"].items():
                if (not isinstance(expected,str) or len(expected)!=64 or
                    any(ch not in "0123456789abcdef" for ch in expected)):
                    raise ValueError("Некорректная контрольная сумма")
                digest=hashlib.sha256()
                with archive.open(name) as file:
                    for chunk in iter(lambda:file.read(1024*1024),b""):
                        digest.update(chunk)
                if digest.hexdigest() != expected:
                    raise ValueError(f"Файл {name} не прошёл проверку SHA-256")
            report = json.loads(archive.read("report.json"))
            if not isinstance(report,dict) or report.get("format") != "life-scenario-1":
                raise ValueError("Неподдерживаемый отчёт сценария")
            return {"valid": True, "label": report.get("label"),
                    "completed": report.get("completed"),
                    "files": len(BUNDLE_FILES), "format": manifest["format"]}
    except (BadZipFile, KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Невозможно проверить архив сценария") from error


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Проверка ZIP-архива Life Scenario Studio")
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_bundle(args.archive), ensure_ascii=False, indent=2))
