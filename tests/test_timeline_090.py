"""v0.9.0: deterministic timeline events and fast spatial sampling."""
from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from life_lab.instruments import ResearchLog, spatial_field
from life_lab.timeline import events_between, export_timeline_csv, scan_timeline
from life_lab.world import CellularAutomaton, Ecosystem


class TimelineTests(unittest.TestCase):
    def test_population_collapse_and_severity(self):
        events = events_between({"generation": 0, "population": 100},
                                {"generation": 1, "population": 40})
        self.assertEqual(events[0].title, "Резкое сокращение")
        self.assertEqual(events[0].severity, "Критично")

    def test_population_growth(self):
        self.assertIn("Быстрый рост", [x.title for x in events_between(
            {"generation": 2, "population": 100},
            {"generation": 3, "population": 150})])

    def test_sensitivity(self):
        p = {"generation": 0, "population": 100}
        n = {"generation": 1, "population": 75}
        self.assertFalse(events_between(p, n, "normal"))
        self.assertTrue(events_between(p, n, "high"))
        self.assertFalse(events_between(p, n, "low"))

    def test_extinction_and_reappearance_are_not_fabricated(self):
        a = {"generation": 5, "herbivores": 4, "predators": 0}
        b = {"generation": 8, "herbivores": 0, "predators": 2}
        events = events_between(a, b)
        self.assertEqual([x.title for x in events],
                         ["Исчезновение: Травоядные", "Появление: Хищники"])
        self.assertTrue(all(x.generation == 8 for x in events))

    def test_missing_value_not_zero(self):
        events = events_between({"generation": 0, "herbivores": 5},
                                {"generation": 5, "herbivores": None})
        self.assertEqual(events, [])

    def test_resource_threshold_only_on_crossing(self):
        events = events_between({"generation": 0, "moisture": 21, "soil": 16},
                                {"generation": 1, "moisture": 19, "soil": 14})
        self.assertEqual({x.title for x in events},
                         {"Низкая влажность", "Истощение минералов"})
        self.assertFalse(events_between({"generation": 1, "moisture": 19},
                                        {"generation": 2, "moisture": 18}))

    def test_genealogy_and_crowding(self):
        events = events_between({"generation": 4, "families": 8, "occupancy": 74},
                                {"generation": 5, "families": 12, "occupancy": 76})
        self.assertEqual(len(events), 2)

    def test_bad_generations_and_sensitivity(self):
        for p, q in ((0, 0), (1, 0), (3, -1)):
            with self.assertRaises(ValueError):
                events_between({"generation": p}, {"generation": q})
        with self.assertRaises(ValueError):
            scan_timeline([], "invalid")

    def test_no_mutation_and_csv(self):
        rows = [{"generation": 0, "population": 100},
                {"generation": 3, "population": 45},
                {"generation": 10, "population": 25}]
        before = [dict(x) for x in rows]
        events = scan_timeline(rows)
        self.assertEqual(rows, before)
        self.assertTrue(events)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "events.csv"
            export_timeline_csv(path, events)
            with path.open(encoding="utf-8-sig", newline="") as source:
                data = list(csv.DictReader(source))
            self.assertEqual(len(data), len(events))
            self.assertEqual(int(data[0]["generation"]), 3)

    def test_real_world_journal_does_not_change_rng(self):
        world = Ecosystem(20, 20, seed=9)
        world.randomize(.17, seed=9)
        log = ResearchLog()
        log.capture(world)
        for _ in range(3):
            actions = world.step()
            log.capture(world, actions)
        rng_before = repr(world._rng.bit_generator.state)
        cells = world.cells.copy()
        scan_timeline(log.records)
        np.testing.assert_array_equal(cells, world.cells)
        self.assertEqual(rng_before, repr(world._rng.bit_generator.state))


class SpatialSamplingTests(unittest.TestCase):
    def _reference(self, arr, max_side, aggregation):
        import math
        h, w = arr.shape
        sy = max(1, math.ceil(h / max_side))
        sx = max(1, math.ceil(w / max_side))
        chunks = []
        for y in range(0, h, sy):
            row = []
            for x in range(0, w, sx):
                part = arr[y:y+sy, x:x+sx]
                row.append({"mean": np.mean, "max": np.max, "sum": np.sum}[aggregation](part))
            chunks.append(row)
        return np.asarray(chunks, dtype=np.float32)

    def test_nondivisible_edges_and_modes(self):
        world = Ecosystem(23, 21, seed=2)
        values = np.arange(23*21, dtype=np.float32).reshape(21, 23)
        world.moisture[:] = values
        for max_side in (2, 7, 50):
            for aggregation in ("mean", "max", "sum"):
                observed = spatial_field(world, "Влага", max_side=max_side,
                                         aggregation=aggregation)
                expected = self._reference(values, max_side, aggregation)
                np.testing.assert_allclose(observed, expected, rtol=1e-6, atol=1e-4)

    def test_all_ecology_fields_and_classic(self):
        world = Ecosystem(25, 20, seed=4)
        world.randomize(.12, seed=4)
        field_arrays = {
            "Плотность": (world.cells > 0).astype(np.float32),
            "Типы клеток": world.cells,
            "Энергия": world.energy,
            "Минералы": world.nutrients,
            "Влага": world.moisture,
            "Возраст": world.age,
            "Метаболизм": np.where(world.cells >= 2, world.trait, 0),
            "Зрение": np.where(world.cells >= 2, world.sight, 0),
            "Плодовитость": np.where(world.cells >= 2, world.fertility, 0),
            "Семена": world.seeds,
            "Органика": world.organic,
            "Токсины": world.toxins,
            "Рельеф": world.altitude,
        }
        for field, values in field_arrays.items():
            for mode in ("mean", "max", "sum"):
                with self.subTest(field=field, mode=mode):
                    np.testing.assert_allclose(
                        spatial_field(world, field, max_side=12, aggregation=mode),
                        self._reference(values, 12, mode), rtol=1e-5, atol=1e-4)
        classic = CellularAutomaton(23, 20)
        classic.randomize(.3, seed=23)
        np.testing.assert_allclose(spatial_field(classic, "Плотность", max_side=6),
            self._reference(classic.cells > 0, 6, "mean"), atol=1e-6)

    def test_lazy_does_not_compute_unselected_neighbours(self):
        world = Ecosystem(20, 20, seed=1)
        with patch("life_lab.instruments.neighbour_counts", side_effect=AssertionError):
            self.assertEqual(spatial_field(world, "Влага").shape, (20, 20))

    def test_unknown_field_is_rejected(self):
        with self.assertRaises(ValueError):
            spatial_field(Ecosystem(20,20,seed=3), "unknown")

    def test_no_world_mutation(self):
        world = Ecosystem(23, 20, seed=5)
        world.randomize(.2, seed=5)
        cells, nutrients = world.cells.copy(), world.nutrients.copy()
        for _ in range(2):
            spatial_field(world, "Соседи", max_side=10)
            spatial_field(world, "Глубина родословной", max_side=10)
        np.testing.assert_array_equal(cells, world.cells)
        np.testing.assert_array_equal(nutrients, world.nutrients)


if __name__ == "__main__":
    unittest.main()
