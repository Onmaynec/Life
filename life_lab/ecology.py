"""Optional experimental climate and life-cycle model for the ecosystem.

This model is deliberately educational, not a validated ecological forecast.
All stochastic processes use the world's own random generator so a saved run
can be resumed without introducing hidden randomness.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


@dataclass
class EcologySettings:
    enabled: bool = False
    season_length: int = 180
    rain: float = 0.60
    heat: float = 0.50
    moisture_flow: float = 0.10
    mineral_flow: float = 0.03
    solar_cycle: bool = True
    season_cycle: bool = True
    storms: bool = True
    drought: bool = True
    photosynthesis: bool = True
    erosion: bool = True
    decomposition: bool = True
    aging: bool = True
    competition: bool = True
    dispersal: bool = True
    temperature_stress: bool = True
    disease: bool = False
    wildfire: bool = False
    wildfire_chance: float = 0.002
    seed_bank: bool = True
    organic_cycle: bool = True
    toxins: bool = False
    toxin_diffusion: bool = True
    water_springs: bool = True
    terrain_runoff: bool = True
    microhabitats: bool = True
    plant_shade: bool = True
    pollination: bool = True
    grazing_fertilizer: bool = True
    herbivore_flee: bool = True
    herd_defense: bool = True
    pack_hunt: bool = True
    territoriality: bool = True
    nocturnal_predators: bool = True
    winter_sleep: bool = True
    juvenile_stage: bool = True
    animal_thirst: bool = True
    seasonal_breeding: bool = True
    migrations: bool = True

    @classmethod
    def from_dict(cls, data: Any) -> "EcologySettings":
        if not isinstance(data, dict):
            raise ValueError("Параметры экосистемы должны быть объектом")
        values = {}
        for key, field in cls.__dataclass_fields__.items():
            value = data.get(key, field.default)
            if isinstance(field.default, bool):
                if type(value) is not bool:
                    raise ValueError(f"Неверное значение {key}")
            elif isinstance(field.default, int):
                if type(value) is not int or not 10 <= value <= 5000:
                    raise ValueError(f"Неверное значение {key}")
            else:
                if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Неверное значение {key}")
            values[key] = value
        return cls(**values)

    def as_dict(self) -> dict[str, float | bool | int]:
        return asdict(self)


def _diffuse(arr: np.ndarray, factor: float, wrap: bool) -> None:
    """Conservative 4-neighbour diffusion, without flow through map borders."""
    if factor <= 0:
        return
    if wrap:
        neighbours = (np.roll(arr, 1, 0) + np.roll(arr, -1, 0) +
                      np.roll(arr, 1, 1) + np.roll(arr, -1, 1))
        arr += factor * (neighbours * 0.25 - arr)
    else:
        transfer = np.zeros_like(arr)
        amount = (arr[:, 1:] - arr[:, :-1]) * (factor * 0.25)
        transfer[:, 1:] -= amount
        transfer[:, :-1] += amount
        amount = (arr[1:, :] - arr[:-1, :]) * (factor * 0.25)
        transfer[1:, :] -= amount
        transfer[:-1, :] += amount
        arr += transfer


def climate_values(generation: int, settings: EcologySettings) -> dict[str, float | str]:
    season = 2 * math.pi * generation / settings.season_length if settings.season_cycle else 0
    sun_phase = 2 * math.pi * (generation % 24) / 24 if settings.solar_cycle else 0
    sunlight = 0.65 + 0.30 * math.cos(season) + (0.20 * math.cos(sun_phase) if settings.solar_cycle else 0)
    temperature = (settings.heat - 0.5) * 1.3 + (0.36 * math.sin(season) if settings.season_cycle else 0)
    return {"sunlight": max(0.1, min(1.3, sunlight)), "temperature": temperature,
            "season": ("Весна", "Лето", "Осень", "Зима")[int(generation / settings.season_length * 4) % 4]}


def environmental_step(world: Any) -> dict[str, int]:
    """Advance optional water/climate and life-cycle processes after normal rules."""
    settings: EcologySettings = world.ecology
    if not settings.enabled:
        return {}
    births_before = world.last_births
    rng = world._rng
    cells = world.cells
    wet = world.moisture
    minerals = world.nutrients
    age = world.age
    weather = climate_values(world.generation, settings)
    sun = float(weather["sunlight"])
    temperature = float(weather["temperature"])
    # Local terrain altitude and latitude produce stable spatial microhabitats.
    terrain = world.altitude
    if settings.microhabitats:
        rain_mask = 0.72 + 0.54 * terrain
    else:
        rain_mask = 1.0
    # 1-4: rainfall, spatial storm bursts, evaporation, soil water diffusion.
    rainfall = 1.15 * settings.rain
    if settings.storms and world.generation % 37 < 4:
        rainfall *= 2.5
    wet += rainfall * rain_mask
    evaporation = (0.35 + settings.heat * 1.15 + max(0.0, temperature) * 0.5)
    wet -= evaporation
    np.clip(wet, 0, 100, out=wet)
    _diffuse(wet, settings.moisture_flow, world.wrap)
    # Springs are stationary, finite sources; downhill runoff redistributes water.
    if settings.water_springs:
        for cy,cx in ((world.height//4,world.width//4),(3*world.height//4,3*world.width//4)):
            wet[max(0,cy-2):cy+3,max(0,cx-2):cx+3] += 2.0
    if settings.terrain_runoff:
        # One small downhill flux from higher cells to their southern neighbours.
        drop = np.maximum(0.0, terrain[:-1,:]-terrain[1:,:])
        flow = np.minimum(wet[:-1,:],drop * 0.065)
        wet[:-1,:] -= flow
        wet[1:,:] += flow
    # 5-7: mineral diffusion, slow washout on wet ground, mineral recycling.
    _diffuse(minerals, settings.mineral_flow, world.wrap)
    if settings.erosion:
        minerals -= (wet > 92).astype(np.float32) * 0.06
    plants = cells == 1
    animals = cells >= 2
    # 8-10: photosynthesis, plant uptake and water shortage.
    plant_losses = np.zeros_like(plants)
    if settings.photosynthesis:
        wet[plants] -= 0.85
        available_light = sun
        if settings.plant_shade:
            from .world import neighbour_counts
            shade = neighbour_counts(plants,world.wrap)
            minerals[plants] -= 0.08 * shade[plants]
        minerals[plants] += 0.22 * (available_light - 0.65)
    if settings.drought:
        plant_losses |= plants & (wet < 8) & (rng.random(cells.shape) < 0.15)
    # 11: crowding accelerates local resource depletion.
    if settings.competition:
        from .world import neighbour_counts
        crowded = plants & (neighbour_counts(plants, world.wrap) >= 5)
        minerals[crowded] -= 0.65
    # Gut microbes recycle waste near herbivores, increasing local minerals.
    if settings.grazing_fertilizer:
        minerals[cells == 2] += 0.12
    # Organic detritus mineralises over time; toxicity slows its recycling.
    if settings.organic_cycle:
        compost = np.minimum(world.organic,0.18)
        minerals += compost * np.maximum(0.25,1-world.toxins/120)
        world.organic -= compost
    if settings.toxins:
        world.toxins[cells >= 2] += 0.09
        world.toxins *= 0.997
        if settings.toxin_diffusion:
            _diffuse(world.toxins,0.055,world.wrap)
    # 12: dispersal occurs on nutrient-rich, wet empty soil.
    if settings.dispersal:
        fertile = (cells == 0) & (wet > 40) & (minerals > 50)
        sprout = fertile & (rng.random(cells.shape) < 0.00022 * sun)
        cells[sprout] = 1
        world.total_births += int(np.count_nonzero(sprout))
        world.last_births += int(np.count_nonzero(sprout))
    # 13: heat/cold stress drains animal energy, reversible with climate.
    if settings.temperature_stress:
        world.energy[animals] -= abs(temperature) * 0.65
    if settings.animal_thirst:
        world.energy[animals & (wet < 9)] -= 0.6
    if settings.toxins:
        world.energy[animals & (world.toxins > 15)] -= 0.55
    # 14-15: age, senescence, age-related mortality.
    if settings.aging:
        age[cells != 0] = np.minimum(age[cells != 0].astype(np.uint32) + 1, 65535).astype(np.uint16)
        plant_losses |= (cells == 1) & (age > 120 + int(50 * sun)) & (rng.random(cells.shape) < 0.015)
        lifespan = np.where(cells == 3, 285, 220)
        plant_losses |= animals & (age > lifespan) & (rng.random(cells.shape) < 0.16)
    # 16: rare disease wave increases mortality in densely occupied patches.
    if settings.disease and world.generation % 53 < 5:
        from .world import neighbour_counts
        exposed = animals & (neighbour_counts(animals, world.wrap) >= 4)
        plant_losses |= exposed & (rng.random(cells.shape) < 0.018)
    # Dormant seeds survive on empty soil and germinate during wet periods.
    if settings.seed_bank:
        world.seeds[plants] = np.minimum(255,world.seeds[plants].astype(np.uint16)+1).astype(np.uint8)
        world.seeds[cells == 2] = 0
        ready = (cells == 0) & (world.seeds > 0) & (wet > 45) & (minerals > 26)
        germ = ready & (rng.random(cells.shape) < .014 * sun)
        cells[germ] = 1
        world.seeds[germ] = 0
        new = int(np.count_nonzero(germ))
        world.total_births += new
        world.last_births += new
    # Simple migration avoids extremely dry adjacent cells.
    if settings.migrations:
        dy,dx = rng.integers(-1,2,size=2)
        if dy or dx:
            dry = (cells == 2) & (wet < 16)
            moved = np.argwhere(dry)
            for y,x in moved[:min(len(moved),30)]:
                ny,nx=int(y+dy),int(x+dx)
                if 0 <= ny < world.height and 0 <= nx < world.width and cells[ny,nx] == 0 and wet[ny,nx] > wet[y,x]:
                    cells[ny,nx] = 2;cells[y,x]=0
                    for field,initial in ((world.energy,0),(world.age,0),(world.trait,1),(world.sight,2),(world.fertility,1)):
                        field[ny,nx]=field[y,x];field[y,x]=initial
    # 17-18: localized wildfire and mineral-rich ash.
    fires = 0
    if settings.wildfire and wet.mean() < 28 and temperature > 0.1:
        from .world import neighbour_counts
        risky = (cells == 1) & (wet < 14)
        if risky.any():
            ignite = risky & (rng.random(cells.shape) < settings.wildfire_chance)
            spread = risky & (neighbour_counts(ignite, world.wrap) > 0)
            burn = ignite | spread
            fires = int(np.count_nonzero(burn))
            plant_losses |= burn
            minerals[burn] += 9
    dead = plant_losses & (cells != 0)
    lost = int(np.count_nonzero(dead))
    # 19: dead organisms return nutrients to soil.
    if settings.decomposition:
        minerals[dead] += np.where(cells[dead] == 1, 4.5, 11.0)
        if settings.organic_cycle:
            world.organic[dead] += np.where(cells[dead] == 1, 3.0, 6.0)
    cells[dead] = 0
    world.energy[dead] = 0
    world.trait[dead] = 1
    world.sight[dead] = 2
    world.fertility[dead] = 1
    age[dead] = 0
    world.seeds[dead & (wet > 10)] = np.minimum(255,world.seeds[dead & (wet > 10)].astype(np.uint16)+2).astype(np.uint8)
    np.clip(wet, 0, 100, out=wet)
    np.clip(minerals, 0, 100, out=minerals)
    np.clip(world.toxins, 0, 100, out=world.toxins)
    np.clip(world.organic, 0, 100, out=world.organic)
    world.last_deaths += lost
    world.total_deaths += lost
    return {"climate_deaths": lost, "fires": fires,
            "environment_births": int(world.last_births - births_before),
            "temperature": temperature, "sunlight": sun}
