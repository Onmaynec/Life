"""Regression tests for Life v0.2.0 genetics and migration."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from life_lab.genetics import evolve_genome, select_food_step
from life_lab.storage import load_world, save_world
from life_lab.world import Ecosystem


class GeneticsTests(unittest.TestCase):
    def test_genes_mutate_inside_limits_and_are_reproducible(self):
        left = np.random.default_rng(42)
        right = np.random.default_rng(42)
        for _ in range(100):
            genes, changed = evolve_genome((1.0, 2.0, 1.0), left)
            self.assertEqual((genes, changed), evolve_genome((1.0, 2.0, 1.0), right))
            self.assertTrue(0.65 <= genes[0] <= 1.45)
            self.assertTrue(1.0 <= genes[1] <= 3.0)
            self.assertTrue(0.70 <= genes[2] <= 1.35)

    def test_food_finding_uses_visible_target(self):
        cells = np.zeros((20, 20), dtype=np.uint8)
        cells[10, 12] = 1
        options = [(9, 10), (10, 9), (11, 10), (10, 11)]
        step = select_food_step(cells, 10, 10, 1, 2, options, False, np.random.default_rng(5))
        self.assertEqual(step, (11, 10))

    def test_vision_wraps_across_border(self):
        cells = np.zeros((20, 20), dtype=np.uint8)
        cells[10, 19] = 1
        options = [(1, 10), (0, 9), (0, 11)]
        step = select_food_step(cells, 0, 10, 1, 2, options, True, np.random.default_rng(3))
        self.assertIn(step, [(0, 9), (0, 11)])

    def test_genome_follows_animal_on_move(self):
        world = Ecosystem(30, 20, seed=17)
        world.paint(10, 10, 2)
        world.trait[10, 10] = 0.79
        world.sight[10, 10] = 2.6
        world.fertility[10, 10] = 1.2
        world.cells[10, 11] = 1
        world.step()
        moved = np.argwhere(world.cells == 2)
        self.assertEqual(moved.shape[0], 1)
        y, x = moved[0]
        self.assertAlmostEqual(float(world.trait[y, x]), 0.79, places=4)
        self.assertAlmostEqual(float(world.sight[y, x]), 2.6, places=4)
        self.assertAlmostEqual(float(world.fertility[y, x]), 1.2, places=4)
        self.assertEqual(world.genome_stats()["sight"], float(world.sight[y, x]))

    def test_save_load_genome_and_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            world = Ecosystem(35, 25, seed=20)
            world.randomize(0.24)
            world.sight[world.cells >= 2] = 2.9
            world.fertility[world.cells >= 2] = 1.19
            path = Path(directory) / "world.lifelab"
            save_world(world, path)
            loaded = load_world(path)
            np.testing.assert_array_equal(world.sight, loaded.sight)
            np.testing.assert_array_equal(world.fertility, loaded.fertility)
            for _ in range(7):
                world.step()
                loaded.step()
            for name in ("cells", "energy", "sight", "fertility", "trait", "nutrients"):
                np.testing.assert_array_equal(getattr(world, name), getattr(loaded, name))

    def test_old_save_defaults_new_genes(self):
        world = Ecosystem(20, 20, seed=4)
        world.randomize(0.18)
        old_meta = {"version": 1, "mode": "ecosystem", "generation": 8,
                    "width": 20, "height": 20, "wrap": False,
                    "rng_state": world._rng.bit_generator.state}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.lifelab"
            with path.open("wb") as dest:
                np.savez_compressed(dest, metadata=np.array(json.dumps(old_meta)),
                                    cells=world.cells, energy=world.energy,
                                    trait=world.trait, nutrients=world.nutrients)
            loaded = load_world(path)
            self.assertEqual(loaded.generation, 8)
            self.assertTrue(np.all(loaded.sight == 2))
            self.assertTrue(np.all(loaded.fertility == 1))

    def test_empty_population_averages_are_zero(self):
        world = Ecosystem(20, 20)
        self.assertEqual(world.genome_stats(), {"metabolism": 0.0, "sight": 0.0, "fertility": 0.0})


if __name__ == "__main__":
    unittest.main()
