"""World setup, measurements, CSV export and save compatibility."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from life_lab.history import COLUMNS, ExperimentHistory
from life_lab.setup import WorldOptions, create_world
from life_lab.storage import load_document, load_world, save_world
from life_lab.world import CellularAutomaton, Ecosystem


class WorldSetupTests(unittest.TestCase):
    def test_classic_seed_matches_and_changes(self):
        a = WorldOptions(mode="classic", width=38, height=28, density=0.4, seed=155, wrap=True)
        left = create_world(a)
        right = create_world(a)
        np.testing.assert_array_equal(left.cells, right.cells)
        self.assertTrue(left.wrap)
        self.assertTrue((left.cells != create_world(WorldOptions(**{**vars(a), "seed": 156})).cells).any())

    def test_ecosystem_species_distribution_and_seed(self):
        opts = WorldOptions(mode="ecosystem", width=50, height=40, plants=0.18,
                            herbivores=0.10, predators=0.05, seed=17, wrap=True)
        one, two = create_world(opts), create_world(opts)
        for name in ("cells", "energy", "nutrients", "trait", "sight", "fertility"):
            np.testing.assert_array_equal(getattr(one, name), getattr(two, name))
        counts = one.counts()
        self.assertGreater(counts["plants"], counts["herbivores"])
        self.assertGreater(counts["herbivores"], counts["predators"])
        one.step()
        two.step()
        np.testing.assert_array_equal(one.cells, two.cells)

    def test_invalid_parameters_are_rejected(self):
        invalid = (
            {"width": 19}, {"height": 501}, {"width": 25.5}, {"mode": "unknown"},
            {"density": -0.1}, {"density": float("nan")}, {"seed": -1},
            {"seed": 2**64}, {"plants": 0.9, "predators": 0.5},
            {"mode": "classic", "rule": "B999/S2"},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    create_world(WorldOptions(**changes))

    def test_exact_zero_density_is_empty(self):
        empty = create_world(WorldOptions(mode="ecosystem", plants=0, herbivores=0,
                                          predators=0, seed=5))
        self.assertEqual(empty.counts()["population"], 0)


class ExperimentTests(unittest.TestCase):
    def test_gene_averages_are_split_by_species(self):
        w = Ecosystem(20, 20)
        w.paint(1, 1, 2)
        w.paint(2, 2, 3)
        w.trait[1, 1] = 0.75
        w.trait[2, 2] = 1.35
        history = ExperimentHistory.start(w)
        row = history.records[0]
        self.assertAlmostEqual(row["herbivores_metabolism"], 0.75)
        self.assertAlmostEqual(row["predators_metabolism"], 1.35)
        self.assertEqual(row["generation"], 0)

    def test_empty_species_averages_are_blank_in_csv(self):
        w = Ecosystem(20, 20, seed=1)
        w.paint(2, 2, 2)
        history = ExperimentHistory.start(w)
        w.step()
        history.record(w, {"births": w.last_births, "deaths": w.last_deaths})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "history.csv"
            before = w.cells.copy()
            history.export_csv(path)
            np.testing.assert_array_equal(before, w.cells)
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                self.assertEqual(tuple(reader.fieldnames), COLUMNS)
                rows = list(reader)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["predators_sight"], "")
            self.assertEqual(rows[0]["generation"], "0")

    def test_history_in_save_and_rng_restored(self):
        w = create_world(WorldOptions(mode="ecosystem", width=35, height=25,
                                      seed=9, plants=0.18, herbivores=0.04, predators=0.01))
        history = ExperimentHistory.start(w)
        for _ in range(3):
            history.record(w, w.step())
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "demo.lifelab"
            save_world(w, path, history)
            loaded, restored = load_document(path)
            self.assertEqual(history.records, restored.records)
            self.assertEqual(w.generation, loaded.generation)
            for _ in range(4):
                np.testing.assert_array_equal(w.cells, loaded.cells)
                w.step()
                loaded.step()
            np.testing.assert_array_equal(w.cells, loaded.cells)
            self.assertEqual(load_world(path).generation, 3)

    def test_legacy_v2_save_starts_history_at_current_generation(self):
        w = Ecosystem(20, 20, seed=3)
        w.randomize(0.1)
        w.step()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "v2.lifelab"
            save_world(w, path)
            with np.load(path, allow_pickle=False) as old:
                meta = json.loads(str(old["metadata"].item()))
                meta["version"] = 2
                arrays = {k: old[k] for k in old.files if k != "metadata"}
            with path.open("wb") as file:
                np.savez_compressed(file, metadata=np.array(json.dumps(meta)), **arrays)
            restored, history = load_document(path)
            self.assertEqual(len(history.records), 1)
            self.assertEqual(history.records[0]["generation"], restored.generation)

    def test_reject_bad_history(self):
        with self.assertRaises(ValueError):
            ExperimentHistory.from_records([{"generation": 0}])
        w = CellularAutomaton(20, 20)
        h = ExperimentHistory.start(w)
        duplicated = [dict(h.records[0]), dict(h.records[0])]
        with self.assertRaises(ValueError):
            ExperimentHistory.from_records(duplicated)

    def test_history_is_bounded(self):
        w = CellularAutomaton(20, 20)
        hist = ExperimentHistory.start(w)
        for _ in range(10003):
            w.generation += 1
            hist.record(w)
        self.assertEqual(len(hist.records), 10000)
        self.assertEqual(hist.records[0]["generation"], 4)


if __name__ == "__main__":
    unittest.main()
