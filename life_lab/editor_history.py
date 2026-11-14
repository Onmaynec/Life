"""Undo/redo snapshots for editor operations, independent of Tk widgets."""
from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any

import numpy as np

from .world import CellularAutomaton, Ecosystem


@dataclass
class EditorSnapshot:
    mode: str
    arrays: tuple[np.ndarray, ...]
    lineage: Any = None
    rule: str | None = None


class EditorHistory:
    """Bounded editor-only history; simulation generations and RNG are untouched."""

    _ECOSYSTEM_ARRAYS = (
        "cells", "energy", "trait", "sight", "fertility", "nutrients",
        "moisture", "age", "seeds", "organic", "toxins", "altitude",
    )

    def __init__(self, limit: int = 25):
        if type(limit) is not int or limit < 1:
            raise ValueError("Лимит истории должен быть положительным")
        self.limit = limit
        self.undo_stack: list[EditorSnapshot] = []
        self.redo_stack: list[EditorSnapshot] = []

    def clear(self) -> None:
        self.undo_stack.clear()
        self.redo_stack.clear()

    @classmethod
    def snapshot(cls, world: CellularAutomaton | Ecosystem) -> EditorSnapshot:
        if isinstance(world, Ecosystem):
            arrays = tuple(getattr(world, key).copy() for key in cls._ECOSYSTEM_ARRAYS)
            return EditorSnapshot(world.mode, arrays, copy.deepcopy(world.lineage))
        return EditorSnapshot(world.mode, (world.cells.copy(),), rule=world.rule)

    @classmethod
    def restore(cls, world: CellularAutomaton | Ecosystem, snapshot: EditorSnapshot) -> bool:
        if snapshot.mode != world.mode or not snapshot.arrays:
            return False
        if world.cells.shape != snapshot.arrays[0].shape:
            return False
        if isinstance(world, Ecosystem):
            if len(snapshot.arrays) != len(cls._ECOSYSTEM_ARRAYS) or snapshot.lineage is None:
                return False
            for key, data in zip(cls._ECOSYSTEM_ARRAYS, snapshot.arrays):
                getattr(world, key)[:] = data
            world.lineage = copy.deepcopy(snapshot.lineage)
        else:
            world.cells[:] = snapshot.arrays[0]
            if snapshot.rule is not None:
                world.set_rule(snapshot.rule)
        return True

    def push(self, world: CellularAutomaton | Ecosystem) -> None:
        self.undo_stack.append(self.snapshot(world))
        if len(self.undo_stack) > self.limit:
            del self.undo_stack[:-self.limit]
        self.redo_stack.clear()

    def undo(self, world: CellularAutomaton | Ecosystem) -> bool:
        if not self.undo_stack:
            return False
        previous = self.undo_stack.pop()
        current = self.snapshot(world)
        if not self.restore(world, previous):
            self.undo_stack.append(previous)
            return False
        self.redo_stack.append(current)
        if len(self.redo_stack) > self.limit:
            del self.redo_stack[:-self.limit]
        return True

    def redo(self, world: CellularAutomaton | Ecosystem) -> bool:
        if not self.redo_stack:
            return False
        following = self.redo_stack.pop()
        current = self.snapshot(world)
        if not self.restore(world, following):
            self.redo_stack.append(following)
            return False
        self.undo_stack.append(current)
        if len(self.undo_stack) > self.limit:
            del self.undo_stack[:-self.limit]
        return True
