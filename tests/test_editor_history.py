"""Editor history snapshots restore every editable layer without touching RNG."""
from __future__ import annotations

import unittest

from life_lab.editor_history import EditorHistory
from life_lab.world import CellularAutomaton, Ecosystem


class EditorHistoryTests(unittest.TestCase):
    def test_classic_undo_redo_restores_rule_and_cells(self):
        world = CellularAutomaton(20, 20, "B3/S23")
        world.cells[4, 5] = 1
        history = EditorHistory()
        history.push(world)
        world.cells[4, 5] = 0
        world.set_rule("B36/S23")
        self.assertTrue(history.undo(world))
        self.assertEqual(int(world.cells[4, 5]), 1)
        self.assertEqual(world.rule, "B3/S23")
        self.assertTrue(history.redo(world))
        self.assertEqual(int(world.cells[4, 5]), 0)
        self.assertEqual(world.rule, "B36/S23")

    def test_ecosystem_undo_restores_layers_and_lineage(self):
        world = Ecosystem(20, 20, seed=4)
        world.paint(5, 6, 2)
        uid = int(world.lineage.ids[6, 5])
        history = EditorHistory()
        history.push(world)
        world.paint(5, 6, 0)
        world.moisture[6, 5] = 2
        self.assertTrue(history.undo(world))
        self.assertEqual(int(world.cells[6, 5]), 2)
        self.assertEqual(int(world.lineage.ids[6, 5]), uid)
        self.assertEqual(float(world.moisture[6, 5]), 55.0)


if __name__ == "__main__":
    unittest.main()
