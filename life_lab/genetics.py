"""Small, deterministic genetics and foraging helpers for Life ecosystems."""
from __future__ import annotations

import numpy as np
from functools import lru_cache

# Mutations stay small. Restrained ranges prevent runaway numerical values.
LIMITS = ((0.65, 1.45), (1.0, 3.0), (0.70, 1.35))
SIGMAS = (0.04, 0.18, 0.045)


def evolve_genome(parent: tuple[float, float, float], rng: np.random.Generator) -> tuple[tuple[float, float, float], bool]:
    """Create a child with mutated metabolism, sight and fertility."""
    genes = []
    mutated = False
    for value, bounds, sigma in zip(parent, LIMITS, SIGMAS):
        variation = float(rng.normal(0, sigma))
        genes.append(min(bounds[1], max(bounds[0], float(value) + variation)))
        mutated |= abs(variation) > sigma * 0.625
    return (genes[0], genes[1], genes[2]), mutated


@lru_cache(maxsize=3)
def _vision_offsets(radius: int) -> tuple[tuple[int, int, int], ...]:
    """Immutable scan order; caching avoids rebuilding the same square per animal."""
    return tuple((dx,dy,max(abs(dx),abs(dy)))
                 for dy in range(-radius,radius+1)
                 for dx in range(-radius,radius+1) if dx or dy)


def select_food_step(cells: np.ndarray, x: int, y: int, food: int,
                     sight: float, options: list[tuple[int, int]],
                     wrap: bool, rng: np.random.Generator) -> tuple[int, int]:
    """Take one unoccupied step towards the closest visible food, otherwise wander.

    Vision uses Chebyshev distance. Ties are broken using the provided RNG,
    and the periodic distance is used on toroidal worlds.
    """
    height, width = cells.shape
    radius = max(1, min(3, round(float(sight))))
    target: tuple[int, int] | None = None
    distance = radius + 1
    for dx, dy, d in _vision_offsets(radius):
        nx, ny = x + dx, y + dy
        if wrap:
            nx %= width
            ny %= height
        elif not (0 <= nx < width and 0 <= ny < height):
            continue
        if cells[ny, nx] != food:
            continue
        if d < distance or (d == distance and rng.random() < 0.5):
            target, distance = (nx, ny), d
    if target is None:
        return options[0]

    def d_to_food(option: tuple[int, int]) -> int:
        dx, dy = abs(target[0] - option[0]), abs(target[1] - option[1])
        if wrap:
            dx, dy = min(dx, width - dx), min(dy, height - dy)
        return max(dx, dy)

    best = min(d_to_food(option) for option in options)
    candidates = [option for option in options if d_to_food(option) == best]
    return candidates[int(rng.integers(len(candidates)))]
