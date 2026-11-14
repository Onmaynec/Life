"""Regression tests for atomic world writes and long-running session recovery."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from life_lab.history import ExperimentHistory
from life_lab.instruments import ResearchLog
from life_lab.preferences import Preferences, load_preferences, save_preferences
from life_lab.recovery import RecoveryStore
from life_lab.storage import load_document, load_research, save_world
from life_lab.world import CellularAutomaton, Ecosystem


class RecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = RecoveryStore(self.root / "state")

    @staticmethod
    def world_data(ecosystem: bool = False):
        if ecosystem:
            world = Ecosystem(32, 24, seed=431)
            world.randomize(0.23, seed=431)
        else:
            world = CellularAutomaton(32, 24)
            world.randomize(0.24, seed=431)
        history = ExperimentHistory.start(world)
        research = ResearchLog()
        research.capture(world)
        for _ in range(4):
            events = world.step()
            history.record(world, events)
            research.capture(world, events)
        return world, history, research

    def test_classic_resume_keeps_history_and_cells(self):
        world, history, research = self.world_data()
        self.store.save(world, history, research)
        loaded, loaded_history, loaded_research = self.store.restore()
        self.assertEqual(loaded.generation, 4)
        self.assertTrue(np.array_equal(loaded.cells, world.cells))
        self.assertEqual(loaded_history.records, history.records)
        self.assertEqual(loaded_research.records, research.records)

    def test_ecosystem_resume_replays_rng_and_lineage(self):
        world, history, research = self.world_data(ecosystem=True)
        self.store.save(world, history, research)
        loaded, _, _ = self.store.restore()
        self.assertTrue(np.array_equal(loaded.lineage.ids, world.lineage.ids))
        self.assertEqual(loaded._rng.bit_generator.state, world._rng.bit_generator.state)
        for _ in range(5):
            world.step()
            loaded.step()
        self.assertTrue(np.array_equal(loaded.cells, world.cells))
        self.assertTrue(np.array_equal(loaded.energy, world.energy))
        self.assertTrue(np.array_equal(loaded.lineage.ids, world.lineage.ids))

    def test_manual_slot_is_independent_and_persistent(self):
        world, history, research = self.world_data()
        self.store.save(world, history, research, manual=True)
        self.store.save(world, history, research)
        self.store.discard()
        self.assertFalse(self.store.available())
        self.assertTrue(self.store.available(manual=True))
        self.assertEqual(self.store.restore(manual=True)[0].generation, world.generation)

    def test_manual_slot_overwrites_own_previous_snapshot(self):
        world, history, research = self.world_data()
        self.store.save(world, history, research, manual=True)
        world.step()
        self.store.save(world, history, research, manual=True)
        self.assertEqual(self.store.restore(manual=True)[0].generation, 5)

    def test_missing_slot_is_not_created_by_check(self):
        self.assertFalse(self.store.available())
        with self.assertRaises(FileNotFoundError):
            self.store.restore()
        self.assertFalse(self.store.directory.exists())

    def test_corrupted_checkpoint_rejected_without_removal(self):
        self.store.directory.mkdir(parents=True)
        self.store.file().write_bytes(b"not a world")
        with self.assertRaises((OSError, ValueError)):
            self.store.restore()
        self.assertEqual(self.store.file().read_bytes(), b"not a world")

    def test_transactional_save_keeps_prior_document_on_failure(self):
        world, history, research = self.world_data()
        target = self.root / "saved.lifelab"
        save_world(world, target, history, research)
        before = target.read_bytes()
        world.step()
        with patch("life_lab.storage.np.savez_compressed", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                save_world(world, target, history, research)
        self.assertEqual(target.read_bytes(), before)
        self.assertFalse(list(self.root.glob("*.tmp")))
        self.assertEqual(load_document(target)[0].generation, 4)

    def test_replace_success_cleans_temp_files(self):
        world, history, research = self.world_data()
        target = self.root / "saved.lifelab"
        save_world(world, target)
        world.step()
        save_world(world, target, history, research)
        self.assertEqual(load_document(target)[0].generation, 5)
        self.assertEqual(load_research(target).records, research.records)
        self.assertFalse(list(self.root.glob("*.tmp")))

    def test_failed_checkpoint_write_leaves_old_checkpoint(self):
        world, history, research = self.world_data(ecosystem=True)
        self.store.save(world, history, research)
        with patch("life_lab.storage.np.savez_compressed", side_effect=OSError("denied")):
            with self.assertRaises(OSError):
                self.store.save(world, history, research)
        self.assertTrue(self.store.available())
        self.assertEqual(self.store.restore()[0].generation, 4)

    def test_preferences_enable_recovery_by_default(self):
        self.assertTrue(Preferences.from_dict({"palette": "Графит"}).autosave)
        self.assertTrue(Preferences.from_dict({"autosave": "false"}).autosave)
        self.assertFalse(Preferences.from_dict({"autosave": False}).autosave)

    def test_recovery_preference_survives_restart(self):
        path = self.root / "prefs.json"
        save_preferences(Preferences(autosave=False), path)
        self.assertFalse(load_preferences(path).autosave)

    def test_saved_document_remains_valid_npz(self):
        world, history, research = self.world_data(ecosystem=True)
        target = self.root / "world.lifelab"
        save_world(world, target, history, research)
        with np.load(target, allow_pickle=False) as saved:
            self.assertEqual(json.loads(str(saved["metadata"].item()))["version"], 5)
            self.assertIn("lineage", json.loads(str(saved["metadata"].item())))
            self.assertIn("research", saved)


if __name__ == "__main__":
    unittest.main()
