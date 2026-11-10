"""Parameters and validation for creating reproducible Life worlds."""
from __future__ import annotations

from dataclasses import dataclass
import math

from .world import CellularAutomaton, Ecosystem, parse_rule


@dataclass(frozen=True)
class WorldOptions:
    mode: str = "classic"
    width: int = 160
    height: int = 100
    wrap: bool = False
    rule: str = "B3/S23"
    density: float = 0.22
    plants: float = 0.17
    herbivores: float = 0.018
    predators: float = 0.003
    seed: int | None = None

    def validate(self) -> None:
        if self.mode not in ("classic", "ecosystem"):
            raise ValueError("Неизвестный режим симуляции")
        if type(self.width) is not int or type(self.height) is not int:
            raise ValueError("Размеры мира должны быть целыми числами")
        CellularAutomaton._validate_dimensions(self.width, self.height)
        if not isinstance(self.wrap, bool):
            raise ValueError("Параметр границ должен быть логическим")
        if self.seed is not None and (type(self.seed) is not int or not 0 <= self.seed < 2**64):
            raise ValueError("Seed должен быть целым числом от 0 до 2^64−1")
        values = (self.density, self.plants, self.herbivores, self.predators)
        if any(not isinstance(v, (float, int)) or not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("Плотности должны находиться в пределах от 0 до 100%")
        if self.plants + self.herbivores + self.predators > 1.0 + 1e-12:
            raise ValueError("Суммарная плотность трёх видов не может превышать 100%")
        if self.mode == "classic":
            parse_rule(self.rule)


def create_world(options: WorldOptions) -> CellularAutomaton | Ecosystem:
    """Build a world with deterministic initial conditions when seed is set."""
    options.validate()
    if options.mode == "classic":
        world = CellularAutomaton(options.width, options.height, options.rule, options.wrap)
        world.randomize(options.density, options.seed)
        return world
    world = Ecosystem(options.width, options.height, options.seed, options.wrap)
    world.randomize(species_densities=(options.plants, options.herbivores, options.predators))
    return world
