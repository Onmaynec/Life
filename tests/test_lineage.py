"""Life v0.7.0: identity, genealogy, migration, migration from old files."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from life_lab.lineage import Pedigree
from life_lab.storage import load_research, load_world, save_world
from life_lab.instruments import MEASURES, ResearchLog, measure, spatial_field
from life_lab.world import Ecosystem, CellularAutomaton


class PedigreeTests(unittest.TestCase):
    def test_founder_child_and_ancestors(self):
        p = Pedigree((20, 20))
        founder = p.founder(3, 4, 2, 0, (1, 2, 1))
        child = p.child(3, 5, 2, 4, (.98, 2.14, 1), founder, True)
        grandchild = p.child(3, 6, 2, 7, (.96, 2.10, .98), child, False)
        self.assertEqual([r.id for r in p.ancestors(grandchild)], [grandchild, child, founder])
        self.assertEqual(p.get(3, 6).founder, founder)
        self.assertEqual(p.get(3, 6).depth, 2)
        self.assertEqual(p.records[founder].offspring, 1)
        self.assertEqual(p.records[child].offspring, 1)
        self.assertEqual(p.summary()["families"], 1)
        self.assertEqual(p.summary()["max_depth"], 2)

    def test_archive_prunes_dead_history_but_keeps_all_living_ids(self):
        with mock.patch("life_lab.lineage.MAX_ARCHIVE", 4), \
             mock.patch("life_lab.lineage.PRUNE_BATCH", 2):
            p = Pedigree((20, 20))
            positions = [(1, x) for x in range(8)]
            for y, x in positions:
                p.founder(y, x, 2, 0, (1, 2, 1))
            for y, x in positions[:4]:
                p.die(y, x, 1, "тест")
            for x in range(8, 11):
                p.founder(1, x, 2, 2, (1, 2, 1))
            live = int(np.count_nonzero(p.ids))
            self.assertEqual(live, 7)
            self.assertEqual(len(p.records), live)
            self.assertEqual(p.lost_records, 4)
            for y, x in np.argwhere(p.ids):
                self.assertIsNotNone(p.get(int(y), int(x)))

    def test_movement_death_and_record_kept(self):
        p = Pedigree((20, 20))
        uid = p.founder(3, 4, 3, 1, (1, 2, 1))
        p.move(3, 4, 4, 5)
        self.assertEqual(p.get(4, 5).id, uid)
        self.assertIsNone(p.get(3, 4))
        p.die(4, 5, 6, "хищник")
        self.assertEqual(p.records[uid].died, 6)
        self.assertEqual(p.records[uid].cause, "хищник")
        self.assertIsNone(p.get(4, 5))
        self.assertEqual(p.summary()["living"], 0)

    def test_validation_wrong_parent_or_occupied_cell(self):
        p = Pedigree((20, 20))
        p.founder(4, 3, 3, 0, (1, 2, 1))
        with self.assertRaises(ValueError):
            p.founder(4, 3, 3, 0, (1, 2, 1))
        with self.assertRaises(ValueError):
            p.child(4, 4, 3, 1, (1, 2, 1), 999, False)
        with self.assertRaises(ValueError):
            p.child(4, 4, 2, 1, (1, 2, 1), 1, False)

    def test_ecosystem_founders_follow_movement(self):
        w = Ecosystem(30, 20, seed=17)
        w.paint(10, 10, 2)
        uid = int(w.lineage.ids[10, 10])
        w.paint(11, 10, 1)
        w.step()
        self.assertEqual(int(np.count_nonzero(w.lineage.ids == uid)), 1)
        self.assertEqual(w.lineage.records[uid].died, None)
        self.assertEqual(int(w.lineage.ids[w.cells == 2][0]), uid)

    def test_predator_eats_and_prey_has_record(self):
        w = Ecosystem(20, 20, seed=11)
        w.paint(5, 5, 3)
        w.paint(5, 6, 2)
        predator = w.lineage.get(5, 5).id
        prey = w.lineage.get(6, 5).id
        w.step()
        self.assertEqual(w.lineage.records[prey].cause, "хищник")
        self.assertEqual(w.lineage.records[predator].died, None)
        self.assertEqual(int(np.count_nonzero(w.lineage.ids == predator)), 1)

    def test_paint_erase_registers_new_founder(self):
        w = Ecosystem(20, 20)
        w.paint(3, 4, 2)
        uid = w.lineage.get(4, 3).id
        w.paint(3, 4, 0)
        self.assertEqual(w.lineage.records[uid].cause, "редактор")
        self.assertEqual(int(w.lineage.ids[4, 3]), 0)
        w.paint(3, 4, 2)
        self.assertNotEqual(w.lineage.get(4, 3).id, uid)

    def test_randomize_seed_same_family_ids(self):
        a = Ecosystem(40, 30, seed=12)
        b = Ecosystem(40, 30, seed=999)
        a.randomize(0.35, seed=200)
        b.randomize(0.35, seed=200)
        np.testing.assert_array_equal(a.lineage.ids, b.lineage.ids)
        for _ in range(12):
            a.step(); b.step()
        np.testing.assert_array_equal(a.lineage.ids, b.lineage.ids)
        self.assertEqual(a.lineage.to_dict(), b.lineage.to_dict())

    def test_real_reproduction_creates_parent_link(self):
        w = Ecosystem(40, 35, seed=72)
        w.randomize(seed=72, species_densities=(.55, .13, 0))
        # Lower the threshold through a generous energy reserve only, without changing rules.
        w.energy[w.cells == 2] = 90
        for _ in range(14):
            w.step()
        children = [r for r in w.lineage.records.values() if r.parent]
        self.assertGreater(len(children), 0)
        self.assertTrue(all(r.founder in w.lineage.records for r in children))
        self.assertTrue(all(w.lineage.records[r.parent].species == r.species for r in children))

    def test_save_restore_preserves_full_family_and_future(self):
        w = Ecosystem(35, 28, seed=42)
        w.randomize(seed=42, species_densities=(.5, .13, .02))
        w.energy[w.cells >= 2] = 75
        for _ in range(6):
            w.step()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "pedigree.lifelab"
            save_world(w, path)
            restored = load_world(path)
            self.assertEqual(w.lineage.to_dict(), restored.lineage.to_dict())
            np.testing.assert_array_equal(w.lineage.ids, restored.lineage.ids)
            for _ in range(7):
                w.step(); restored.step()
            self.assertEqual(w.lineage.to_dict(), restored.lineage.to_dict())
            np.testing.assert_array_equal(w.cells, restored.cells)

    def test_v4_file_migration_builds_founders(self):
        w = Ecosystem(20, 20, seed=5)
        w.randomize(seed=5, species_densities=(.25, .05, .02))
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "legacy.lifelab"
            old = {"version": 4, "mode": "ecosystem", "width": w.width,
                   "height": w.height, "generation": 24, "wrap": False,
                   "rng_state": w._rng.bit_generator.state,
                   "ecology": w.ecology.as_dict()}
            with path.open("wb") as f:
                np.savez_compressed(f, metadata=np.array(json.dumps(old)),
                    **{key: getattr(w, key) for key in
                       ("cells", "energy", "trait", "sight", "fertility",
                        "nutrients", "moisture", "age", "seeds",
                        "organic", "toxins", "altitude")})
            restored = load_world(path)
            self.assertEqual(restored.generation, 24)
            self.assertEqual(restored.lineage.summary()["living"],
                             int(np.count_nonzero(restored.cells >= 2)))
            self.assertEqual(restored.lineage.summary()["max_depth"], 0)
            restored.step()

    def test_bad_ids_rejected_before_use(self):
        cells = np.zeros((20, 20), dtype=np.uint8)
        cells[0, 0] = cells[0, 1] = 2
        ids = np.zeros_like(cells, dtype=np.uint64)
        ids[0, 0] = ids[0, 1] = 1
        p = Pedigree((20, 20))
        p.founder(0, 0, 2, 0, (1, 2, 1))
        with self.assertRaises(ValueError):
            Pedigree.from_dict(ids, p.to_dict(), cells)
        ids[0, 1] = 99
        with self.assertRaises(ValueError):
            Pedigree.from_dict(ids, p.to_dict(), cells)

    def test_export_csv_and_research_history(self):
        w = Ecosystem(20, 20)
        w.paint(3, 3, 2)
        row = measure(w)
        self.assertEqual(row["families"], 1)
        self.assertEqual(row["lineage_depth"], 0)
        self.assertEqual(row["pedigree_records"], 1)
        history = ResearchLog()
        history.capture(w)
        old = dict(history.records[0])
        for key in ("families", "lineage_depth", "mean_lineage_depth", "pedigree_records"):
            old.pop(key)
        recovered = ResearchLog.from_records([old])
        self.assertEqual(set(recovered.records[0]), set(MEASURES))
        self.assertIsNone(recovered.records[0]["families"])
        self.assertEqual(float(spatial_field(w,"Глубина родословной").max()), 0.0)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "genealogy.csv"
            w.lineage.export_csv(path)
            with path.open("r", encoding="utf-8-sig") as f:
                result = list(csv.DictReader(f))
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]["founder"], "1")
            self.assertEqual(result[0]["species"], "2")

    def test_classic_remains_unchanged(self):
        world = CellularAutomaton(30, 30)
        world.cells[10:13, 10] = 1
        world.step()
        self.assertEqual(int(world.cells.sum()), 3)
        self.assertIsNone(measure(world)["families"])


if __name__ == "__main__":
    unittest.main()
