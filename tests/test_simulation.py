import tempfile
import unittest
from pathlib import Path

import numpy as np

from life_lab.world import CellularAutomaton, Ecosystem, neighbour_counts, parse_rule, _grid_distance_squared
from life_lab.patterns import PATTERNS, pattern_array, place_pattern, encode_rle, decode_rle
from life_lab.storage import save_world, load_world


class GameOfLifeTests(unittest.TestCase):
    def test_rule_parser(self):
        self.assertEqual(parse_rule("b36/s23"), ((3, 6), (2, 3)))
        for s in ("B9/S23", "bad", "B33/S23", "B3/23", "B3/S2345X"):
            with self.assertRaises(ValueError):
                parse_rule(s)

    def test_blinker(self):
        w = CellularAutomaton(20, 20)
        w.cells[5, 5:8] = 1
        w.step()
        self.assertEqual(np.argwhere(w.cells).tolist(), [[4, 6], [5, 6], [6, 6]])
        w.step()
        self.assertEqual(np.argwhere(w.cells).tolist(), [[5, 5], [5, 6], [5, 7]])
        self.assertEqual(w.generation, 2)

    def test_block_still_life(self):
        w = CellularAutomaton(30, 30)
        w.cells[10:12, 10:12] = 1
        for _ in range(20):
            w.step()
        self.assertEqual(int(w.cells.sum()), 4)

    def test_glider_moves_one_diagonal_after_four_generations(self):
        w = CellularAutomaton(30, 30)
        p = pattern_array("Глайдер")
        place_pattern(w.cells, p, 5, 5)
        for _ in range(4):
            w.step()
        np.testing.assert_array_equal(w.cells[6:9, 6:9], p)
        self.assertEqual(int(w.cells.sum()), 5)

    def test_wrap_boundary(self):
        grid = np.zeros((4, 4), dtype=np.uint8)
        grid[0, 0] = 1
        self.assertEqual(neighbour_counts(grid, False)[3, 3], 0)
        self.assertEqual(neighbour_counts(grid, True)[3, 3], 1)

    def test_all_patterns_within_legal_symbols(self):
        for name in PATTERNS:
            arr = pattern_array(name)
            self.assertTrue(np.isin(arr, [0, 1]).all())
            self.assertGreater(arr.sum(), 0)

    def test_rle_roundtrip(self):
        w = CellularAutomaton()
        place_pattern(w.cells, pattern_array("Пушка Госпера"), 22, 25)
        text = encode_rle(w.cells)
        decoded, rule = decode_rle(text)
        self.assertEqual(rule, "B3/S23")
        self.assertEqual(int(decoded.sum()), int(w.cells.sum()))
        self.assertEqual(decoded.shape[1], 36)

    def test_eco_paint_and_step(self):
        w = Ecosystem(35, 25, seed=9)
        w.paint(5, 5, 2)
        self.assertEqual(w.cells[5, 5], 2)
        self.assertGreater(w.energy[5, 5], 0)
        w.step()
        self.assertEqual(w.generation, 1)
        self.assertTrue(np.isfinite(w.energy).all())
        self.assertTrue(np.isfinite(w.nutrients).all())

    def test_ecosystem_deterministic(self):
        a = Ecosystem(50, 35, seed=4)
        b = Ecosystem(50, 35, seed=4)
        a.randomize(0.21)
        b.randomize(0.21)
        for _ in range(8):
            a.step()
            b.step()
        np.testing.assert_array_equal(a.cells, b.cells)
        np.testing.assert_array_equal(a.energy, b.energy)

    def test_save_load_both_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            for world in (CellularAutomaton(33, 25, "B36/S23", wrap=True), Ecosystem(33, 25, seed=10)):
                world.randomize(0.2, seed=8)
                for _ in range(2):
                    world.step()
                path = Path(tmp) / "test.lifelab"
                save_world(world, path)
                loaded = load_world(path)
                self.assertEqual(world.mode, loaded.mode)
                self.assertEqual(world.generation, loaded.generation)
                self.assertEqual(world.total_births, loaded.total_births)
                np.testing.assert_array_equal(world.cells, loaded.cells)
                if isinstance(world, Ecosystem):
                    np.testing.assert_array_equal(world.nutrients, loaded.nutrients)
                    np.testing.assert_array_equal(world.trait, loaded.trait)
                    world.step()
                    loaded.step()
                    np.testing.assert_array_equal(world.cells, loaded.cells)
                    np.testing.assert_array_equal(world.energy, loaded.energy)
                else:
                    self.assertEqual(world.rule, loaded.rule)


class RLEValidationTests(unittest.TestCase):
    def test_invalid_rule_is_rejected_during_decode(self):
        with self.assertRaises(ValueError):
            decode_rle("x = 1, y = 1, rule = B9/S23\no!")

    def test_toroidal_distance_uses_shortest_path(self):
        self.assertEqual(_grid_distance_squared((0, 5), (19, 5), 20, 20, True), 1)
        self.assertEqual(_grid_distance_squared((0, 5), (19, 5), 20, 20, False), 361)


if __name__ == "__main__":
    unittest.main()
