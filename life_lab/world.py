"""Simulation engines for LIFE//LAB.

Two explicit modes:
- CellularAutomaton: configurable B/S Moore-neighbourhood rules.
- Ecosystem: an experimental agent-based model with finite nutrients,
  producers, herbivores, predators and inherited, mutable traits.

The ecosystem is *not* Conway's original Game of Life or a realistic
biological simulator; it is a deliberately simplified sandbox.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .genetics import evolve_genome, select_food_step
from .lineage import Pedigree
from .ecology import EcologySettings, environmental_step


RULE_PRESETS = {
    "Жизнь Конвея": "B3/S23",
    "HighLife": "B36/S23",
    "Seeds": "B2/S",
    "Day & Night": "B3678/S34678",
    "Maze": "B3/S12345",
    "Replicator": "B1357/S1357",
}


def parse_rule(rule: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return birth and survival digits for a conventional B*/S* rule."""
    rule = rule.strip().upper().replace(" ", "")
    if "/" not in rule or not rule.startswith("B"):
        raise ValueError("Правило должно иметь вид B3/S23")
    birth, survive = rule.split("/", maxsplit=1)
    if not survive.startswith("S"):
        raise ValueError("Правило должно иметь вид B3/S23")
    b, s = birth[1:], survive[1:]
    if any(ch not in "012345678" for ch in b + s):
        raise ValueError("Разрешены только числа соседей от 0 до 8")
    if len(set(b)) != len(b) or len(set(s)) != len(s):
        raise ValueError("Числа в правиле не должны повторяться")
    return tuple(sorted(int(ch) for ch in b)), tuple(sorted(int(ch) for ch in s))


def neighbour_counts(mask: np.ndarray, wrap: bool = False) -> np.ndarray:
    """Count the eight Moore neighbours of every cell."""
    if mask.ndim != 2:
        raise ValueError("Expected 2D grid")
    values = mask.astype(np.uint8, copy=False)
    if wrap:
        total = np.zeros_like(values, dtype=np.uint8)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    total += np.roll(np.roll(values, dy, axis=0), dx, axis=1)
        return total
    padded = np.pad(values, 1, mode="constant")
    h, w = values.shape
    total = np.zeros_like(values, dtype=np.uint8)
    for dy in range(3):
        for dx in range(3):
            if dy != 1 or dx != 1:
                total += padded[dy:dy + h, dx:dx + w]
    return total


class CellularAutomaton:
    mode = "classic"

    def __init__(self, width: int = 160, height: int = 100, rule: str = "B3/S23", wrap: bool = False):
        self._validate_dimensions(width, height)
        self.cells = np.zeros((height, width), dtype=np.uint8)
        self.generation = 0
        self.rule = ""
        self.birth = (3,)
        self.survival = (2, 3)
        self.set_rule(rule)
        self.wrap = wrap
        self.total_births = 0
        self.total_deaths = 0

    @staticmethod
    def _validate_dimensions(w: int, h: int) -> None:
        if not (20 <= w <= 700 and 20 <= h <= 500):
            raise ValueError("Размер мира: от 20×20 до 700×500")

    @property
    def width(self) -> int:
        return self.cells.shape[1]

    @property
    def height(self) -> int:
        return self.cells.shape[0]

    def set_rule(self, rule: str) -> None:
        self.birth, self.survival = parse_rule(rule)
        self.rule = f"B{''.join(map(str, self.birth))}/S{''.join(map(str, self.survival))}"

    def step(self) -> dict[str, int]:
        neighbours = neighbour_counts(self.cells, wrap=self.wrap)
        alive = self.cells.astype(bool)
        born = ~alive & np.isin(neighbours, self.birth)
        survive = alive & np.isin(neighbours, self.survival)
        deaths = int(np.count_nonzero(alive & ~survive))
        births = int(np.count_nonzero(born))
        self.cells = (born | survive).astype(np.uint8)
        self.generation += 1
        self.total_births += births
        self.total_deaths += deaths
        return {"births": births, "deaths": deaths}

    def clear(self) -> None:
        self.cells.fill(0)
        self.generation = 0
        self.total_births = 0
        self.total_deaths = 0

    def randomize(self, density: float = 0.22, seed: int | None = None) -> None:
        if not 0 <= density <= 1:
            raise ValueError("Density must be between 0 and 1")
        self.cells[:] = (np.random.default_rng(seed).random(self.cells.shape) < density).astype(np.uint8)
        self.generation = 0
        self.total_births = 0
        self.total_deaths = 0

    def paint(self, x: int, y: int, kind: int = 1) -> bool:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.cells[y, x] = np.uint8(1 if kind else 0)
            return True
        return False

    def counts(self) -> dict[str, int]:
        return {"cells": int(np.count_nonzero(self.cells)), "population": int(np.count_nonzero(self.cells))}


@dataclass(frozen=True)
class Species:
    name: str
    cost: float
    food_gain: float
    reproduce_at: float
    initial_energy: float


SPECIES = {
    1: Species("Растения", 0.0, 0.0, 0.0, 0.0),
    2: Species("Травоядные", 1.1, 16.0, 39.0, 24.0),
    3: Species("Хищники", 2.0, 23.0, 61.0, 34.0),
}


class Ecosystem:
    """Local predator–prey sandbox with three inheritable animal traits.

    Metabolism, sight and fertility affect the cost of living, foraging
    and reproduction. All random choices use the world RNG.
    """
    mode = "ecosystem"

    def __init__(self, width: int = 160, height: int = 100, seed: int | None = None, wrap: bool = False):
        CellularAutomaton._validate_dimensions(width, height)
        self.cells = np.zeros((height, width), dtype=np.uint8)
        self.energy = np.zeros((height, width), dtype=np.float32)
        self.trait = np.ones((height, width), dtype=np.float32)
        self.sight = np.full((height, width), 2.0, dtype=np.float32)
        self.fertility = np.ones((height, width), dtype=np.float32)
        self.nutrients = np.full((height, width), 65.0, dtype=np.float32)
        self.moisture = np.full((height, width), 55.0, dtype=np.float32)
        self.age = np.zeros((height, width), dtype=np.uint16)
        self.seeds = np.zeros((height, width), dtype=np.uint8)
        self.organic = np.zeros((height, width), dtype=np.float32)
        self.toxins = np.zeros((height, width), dtype=np.float32)
        self.altitude = np.full((height, width),0.50,dtype=np.float32)
        self.ecology = EcologySettings()
        self.lineage = Pedigree(self.cells.shape)
        self.generation = 0
        self.wrap = wrap
        self._rng = np.random.default_rng(seed)
        self.last_births = 0
        self.last_deaths = 0
        self.mutations = 0
        self.total_births = 0
        self.total_deaths = 0

    @property
    def width(self) -> int:
        return self.cells.shape[1]

    @property
    def height(self) -> int:
        return self.cells.shape[0]

    def clear(self) -> None:
        self.cells.fill(0)
        self.energy.fill(0)
        self.trait.fill(1)
        self.sight.fill(2)
        self.fertility.fill(1)
        self.nutrients.fill(65)
        self.moisture.fill(55)
        self.age.fill(0)
        self.seeds.fill(0)
        self.organic.fill(0)
        self.toxins.fill(0)
        self.altitude.fill(.50)
        self.lineage.reset()
        self.generation = 0
        self.last_births = self.last_deaths = self.mutations = 0
        self.total_births = self.total_deaths = 0

    def randomize(self, density: float = 0.19, seed: int | None = None,
                  species_densities: tuple[float, float, float] | None = None) -> None:
        if species_densities is None:
            if not np.isfinite(density) or not 0 <= density <= 1:
                raise ValueError("Плотность должна быть от 0 до 1")
            species_densities = (density * 0.89, density * 0.095, density * 0.015)
        if (len(species_densities) != 3 or any(not np.isfinite(v) or v < 0 or v > 1
                for v in species_densities) or sum(species_densities) > 1.0 + 1e-12):
            raise ValueError("Плотность видов должна суммарно составлять не более 100%")
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.clear()
        plant_density, herb_density, pred_density = species_densities
        roll = self._rng.random(self.cells.shape)
        self.cells[roll < plant_density] = 1
        self.cells[(roll >= plant_density) & (roll < plant_density + herb_density)] = 2
        self.cells[(roll >= plant_density + herb_density) &
                   (roll < plant_density + herb_density + pred_density)] = 3
        for kind in (2, 3):
            self.energy[self.cells == kind] = SPECIES[kind].initial_energy
        self.nutrients[:] = self._rng.uniform(45, 95, self.cells.shape).astype(np.float32)
        self.moisture[:] = self._rng.uniform(35, 85, self.cells.shape).astype(np.float32)
        gy,gx=np.indices(self.cells.shape,dtype=np.float32)
        elevation = .5+.18*np.sin(gx*.035)+.16*np.cos(gy*.05)+self._rng.normal(0,.05,self.cells.shape)
        self.altitude[:]=np.clip(elevation,0,1).astype(np.float32)
        # Every initial animal is a founder. Registration consumes no random draws.
        self.rebuild_lineage()

    def rebuild_lineage(self) -> None:
        """Start a new family for each existing animal (legacy save migration)."""
        self.lineage.reset()
        for yy, xx in np.argwhere(self.cells >= 2):
            y, x = int(yy), int(xx)
            self.lineage.founder(y, x, int(self.cells[y, x]), self.generation,
                                 (self.trait[y, x], self.sight[y, x], self.fertility[y, x]))

    def paint(self, x: int, y: int, kind: int = 1) -> bool:
        if not (0 <= x < self.width and 0 <= y < self.height and 0 <= kind <= 3):
            return False
        self.lineage.die(y, x, self.generation, "редактор")
        self.cells[y, x] = kind
        self.energy[y, x] = SPECIES[kind].initial_energy if kind in (2, 3) else 0.0
        self.trait[y, x] = 1.0
        self.sight[y, x] = 2.0
        self.fertility[y, x] = 1.0
        self.age[y, x] = 0
        self.seeds[y, x] = 0
        if kind >= 2:
            self.lineage.founder(y, x, kind, self.generation, (1.0, 2.0, 1.0))
        return True

    def _neighbours(self, x: int, y: int) -> list[tuple[int, int]]:
        points = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                xx, yy = x + dx, y + dy
                if self.wrap:
                    xx %= self.width
                    yy %= self.height
                elif not (0 <= xx < self.width and 0 <= yy < self.height):
                    continue
                points.append((xx, yy))
        self._rng.shuffle(points)
        return points

    def step(self) -> dict[str, int]:
        rng = self._rng
        cells = self.cells
        births = deaths = new_mutations = 0
        # Soil refills slowly everywhere; occupied plant cells consume it.
        np.minimum(self.nutrients + 0.38, 100.0, out=self.nutrients)
        plant = cells == 1
        self.nutrients[plant] -= 2.2
        starving = plant & (self.nutrients < 4.0)
        death_count = int(np.count_nonzero(starving))
        cells[starving] = 0
        deaths += death_count
        # Plants reproduce only on suitable, adjacent soil.
        n = neighbour_counts(cells == 1, self.wrap)
        available = (cells == 0) & (self.nutrients >= 23.0)
        fertile = available & (n >= 1) & (n <= 5)
        if self.ecology.enabled and self.ecology.pollination:
            fertile &= n >= 2
        sprout = (fertile & (rng.random(cells.shape) < 0.17)) | (available & (self.nutrients > 55) & (rng.random(cells.shape) < 0.00018))
        births += int(np.count_nonzero(sprout))
        cells[sprout] = 1
        self.nutrients[sprout] -= 7.0

        # Predators move first; herbivores then take their turns. Animal
        # positions are shuffled to avoid permanent scan-direction bias.
        if self.ecology.enabled:
            from .ecology import climate_values
            climate = climate_values(self.generation,self.ecology)
        for kind in (3, 2):
            actors = np.argwhere(cells == kind)
            rng.shuffle(actors)
            prey_kind = kind - 1
            for yy, xx in actors:
                y, x = int(yy), int(xx)
                if cells[y, x] != kind:
                    continue  # Already eaten/moved.
                spec = SPECIES[kind]
                metabolic = float(self.trait[y, x])
                sight = float(self.sight[y, x])
                fertility = float(self.fertility[y, x])
                # Better sight and faster reproduction have an ongoing cost.
                life_cost = spec.cost * metabolic * (1 + 0.09 * (sight - 1) + 0.12 * max(0, fertility - 1))
                if self.ecology.enabled:
                    if self.ecology.winter_sleep and kind==2 and climate["season"]=="Зима":
                        life_cost *= .64
                    if self.ecology.nocturnal_predators and kind==3 and climate["sunlight"]<.55:
                        life_cost *= .83
                    if self.ecology.territoriality and sum(cells[ny,nx]==kind for nx,ny in self._neighbours(x,y)) >= 5:
                        life_cost *= 1.30
                self.energy[y,x] -= life_cost
                if self.energy[y, x] <= 0:
                    self.lineage.die(y, x, self.generation + 1, "голод")
                    cells[y, x] = 0
                    self.energy[y, x] = 0
                    self.trait[y, x] = 1
                    self.sight[y, x] = 2
                    self.fertility[y, x] = 1
                    self.age[y, x] = 0
                    deaths += 1
                    continue
                nearby = self._neighbours(x, y)
                food = [(nx, ny) for nx, ny in nearby if cells[ny, nx] == prey_kind]
                empty = [(nx, ny) for nx, ny in nearby if cells[ny, nx] == 0]
                destination: tuple[int, int] | None = None
                if self.ecology.enabled and self.ecology.herd_defense and kind==3:
                    food=[(nx,ny) for nx,ny in food if sum(cells[py,px]==2 for px,py in self._neighbours(nx,ny))<4]
                if self.ecology.enabled and self.ecology.herbivore_flee and kind==2 and empty and any(cells[ny,nx]==3 for nx,ny in nearby):
                    danger=[(nx,ny) for nx,ny in nearby if cells[ny,nx]==3]
                    destination=max(empty,key=lambda dest: min((dest[0]-px)**2+(dest[1]-py)**2 for px,py in danger))
                elif food:
                    destination = food[0]
                    deaths += 1  # Prey eaten; never counted as surviving.
                    if prey_kind >= 2:
                        self.lineage.die(destination[1], destination[0], self.generation + 1, "хищник")
                    bonus=1.0
                    if self.ecology.enabled and self.ecology.pack_hunt and kind==3:
                        bonus += 0.12*sum(cells[ny,nx]==3 for nx,ny in nearby)
                    self.energy[y, x] = min(100, self.energy[y, x] + spec.food_gain * bonus)
                elif empty and rng.random() < 0.78:
                    if self.ecology.enabled and self.ecology.animal_thirst and self.moisture[y,x]<12:
                        destination=max(empty,key=lambda pos:float(self.moisture[pos[1],pos[0]]))
                    else:
                        destination = select_food_step(cells, x, y, prey_kind, sight, empty, self.wrap, rng)
                if destination is not None:
                    nx, ny = destination
                    self.lineage.move(y, x, ny, nx)
                    cells[ny, nx] = kind
                    self.energy[ny, nx] = self.energy[y, x]
                    self.trait[ny, nx] = self.trait[y, x]
                    self.sight[ny, nx] = self.sight[y, x]
                    self.fertility[ny, nx] = self.fertility[y, x]
                    self.age[ny, nx] = self.age[y, x]
                    cells[y, x] = 0
                    self.energy[y, x] = 0
                    self.trait[y, x] = 1
                    self.sight[y, x] = 2
                    self.fertility[y, x] = 1
                    self.age[y, x] = 0
                    x, y = nx, ny

                reproduction=(0.09 if kind == 2 else 0.055) * float(self.fertility[y,x])
                if self.ecology.enabled:
                    if self.ecology.seasonal_breeding:
                        reproduction *= (1.22 if climate["season"] == "Весна" else .82 if climate["season"] == "Зима" else 1.0)
                    if self.ecology.juvenile_stage and self.age[y,x] < (11 if kind==2 else 19):
                        reproduction=0
                if self.energy[y, x] > spec.reproduce_at and rng.random() < reproduction:
                    free = [(nx, ny) for nx, ny in self._neighbours(x, y) if cells[ny, nx] == 0]
                    if free:
                        nx, ny = free[0]
                        child_energy = float(self.energy[y, x]) * 0.38
                        self.energy[y, x] -= child_energy
                        cells[ny, nx] = kind
                        self.energy[ny, nx] = child_energy
                        genome, changed = evolve_genome((self.trait[y, x], self.sight[y, x],
                                                          self.fertility[y, x]), rng)
                        self.trait[ny, nx], self.sight[ny, nx], self.fertility[ny, nx] = genome
                        self.age[ny, nx] = 0
                        self.lineage.child(ny, nx, kind, self.generation + 1, genome,
                                           int(self.lineage.ids[y, x]), changed)
                        new_mutations += int(changed)
                        births += 1

        self.generation += 1
        self.last_births, self.last_deaths = births, deaths
        self.total_births += births
        self.total_deaths += deaths
        self.mutations += new_mutations
        environmental = environmental_step(self)
        return {"births": self.last_births, "deaths": self.last_deaths,
                "mutations": new_mutations, **environmental}

    def pedigree_stats(self) -> dict[str, float | int]:
        return self.lineage.summary()

    def genome_stats(self) -> dict[str, float]:
        """Population means over living animals only; empty worlds use zero."""
        mask = self.cells >= 2
        if not np.any(mask):
            return {"metabolism": 0.0, "sight": 0.0, "fertility": 0.0}
        return {"metabolism": float(self.trait[mask].mean()),
                "sight": float(self.sight[mask].mean()),
                "fertility": float(self.fertility[mask].mean())}

    def counts(self) -> dict[str, int]:
        plants = int(np.count_nonzero(self.cells == 1))
        herbivores = int(np.count_nonzero(self.cells == 2))
        predators = int(np.count_nonzero(self.cells == 3))
        return {"plants": plants, "herbivores": herbivores,
                "predators": predators, "population": plants + herbivores + predators,
                "mutations": self.mutations}
