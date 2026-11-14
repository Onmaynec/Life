"""Identity and ancestry bookkeeping for the ecosystem's animal populations.

This is a *pedigree*, not a phylogenetic inference: animals reproduce asexually
and the parent link is an explicit simulation event. Plants are not tracked.
The record archive is capped to bound long-running experiments. Old dead
records may be pruned, while all living identities are retained.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

MAX_ARCHIVE = 50_000
PRUNE_BATCH = 2_500


@dataclass
class Individual:
    id: int
    species: int
    parent: int
    founder: int
    depth: int
    born: int
    genome: tuple[float, float, float]
    mutated: bool = False
    died: int | None = None
    cause: str = ""
    offspring: int = 0


class Pedigree:
    """Stable animal IDs in a 2D grid, with an indexed genealogy."""

    def __init__(self, shape: tuple[int, int]):
        self.ids = np.zeros(shape, dtype=np.uint64)
        self.records: dict[int, Individual] = {}
        self.next_id = 1
        self.lost_records = 0

    def reset(self) -> None:
        self.ids.fill(0)
        self.records.clear()
        self.next_id = 1
        self.lost_records = 0

    def _spawn(self, y: int, x: int, species: int, generation: int,
               genome: tuple[float, float, float], parent: int = 0,
               mutated: bool = False) -> int:
        if species not in (2, 3) or self.ids[y, x] != 0:
            raise ValueError("Место для нового животного должно быть свободно")
        ancestor = self.records.get(parent)
        if parent and (ancestor is None or ancestor.species != species):
            raise ValueError("Родитель не найден или относится к другому виду")
        uid = self.next_id
        self.next_id += 1
        record = Individual(id=uid, species=species, parent=parent,
                            founder=ancestor.founder if ancestor else uid,
                            depth=ancestor.depth + 1 if ancestor else 0,
                            born=int(generation),
                            genome=tuple(float(v) for v in genome), mutated=bool(mutated))
        self.records[uid] = record
        self.ids[y, x] = uid
        if ancestor:
            ancestor.offspring += 1
        if len(self.records) > MAX_ARCHIVE + PRUNE_BATCH:
            self._prune()
        return uid

    def founder(self, y: int, x: int, species: int, generation: int,
                genome: tuple[float, float, float]) -> int:
        return self._spawn(y, x, species, generation, genome)

    def child(self, y: int, x: int, species: int, generation: int,
              genome: tuple[float, float, float], parent: int,
              mutated: bool) -> int:
        if parent <= 0:
            raise ValueError("У потомка должен быть родитель")
        return self._spawn(y, x, species, generation, genome, parent, mutated)

    def die(self, y: int, x: int, generation: int, cause: str) -> None:
        uid = int(self.ids[y, x])
        if not uid:
            return
        record = self.records[uid]
        record.died = int(generation)
        record.cause = cause
        self.ids[y, x] = 0

    def move(self, y: int, x: int, new_y: int, new_x: int) -> None:
        if self.ids[new_y, new_x] != 0:
            raise ValueError("Место перемещения уже занято другим животным")
        self.ids[new_y, new_x] = self.ids[y, x]
        self.ids[y, x] = 0

    def get(self, y: int, x: int) -> Individual | None:
        return self.records.get(int(self.ids[y, x]))

    def ancestors(self, uid: int, limit: int = 30) -> list[Individual]:
        result = []
        seen = set()
        while uid and len(result) < limit and uid not in seen:
            seen.add(uid)
            record = self.records.get(uid)
            if record is None:
                break  # An ancestor may have been pruned from the bounded archive.
            result.append(record)
            uid = record.parent
        return result

    def summary(self) -> dict[str, int | float]:
        living = self.ids[self.ids > 0]
        records = [self.records[int(uid)] for uid in living]
        founders = {record.founder for record in records}
        return {
            "living": len(records),
            "families": len(founders),
            "max_depth": max((rec.depth for rec in records), default=0),
            "mean_depth": sum(rec.depth for rec in records) / len(records) if records else 0.0,
            "recorded": len(self.records),
            "pruned": self.lost_records,
        }

    def dominant_families(self, limit: int = 8) -> list[tuple[int, int, int]]:
        """(founder ID, living descendants, maximum living depth)."""
        groups: dict[int, list[int]] = {}
        for uid in self.ids.flat:
            if uid:
                rec = self.records[int(uid)]
                group = groups.setdefault(rec.founder, [0, 0])
                group[0] += 1
                group[1] = max(group[1], rec.depth)
        return sorted(((f, n, depth) for f, (n, depth) in groups.items()),
                      key=lambda row: (-row[1], row[0]))[:limit]

    def _prune(self) -> None:
        alive = set(map(int, self.ids[self.ids != 0]))
        # Retain the ancestors of every living animal to keep the visible trees intact.
        protected = set(alive)
        for uid in alive:
            seen = set()
            while uid and uid not in seen:
                seen.add(uid)
                record = self.records.get(uid)
                if record is None or not record.parent:
                    break
                uid = record.parent
                protected.add(uid)
        candidates = [uid for uid, rec in self.records.items()
                      if rec.died is not None and uid not in protected]
        for uid in sorted(candidates)[:max(PRUNE_BATCH, len(self.records) - MAX_ARCHIVE)]:
            del self.records[uid]
            self.lost_records += 1

    def to_dict(self) -> dict[str, Any]:
        return {"next_id": self.next_id, "lost_records": self.lost_records,
                "records": [asdict(r) for r in self.records.values()]}

    @classmethod
    def from_dict(cls, ids: np.ndarray, data: dict[str, Any], cells: np.ndarray) -> "Pedigree":
        if ids.shape != cells.shape or ids.dtype.kind not in "iu":
            raise ValueError("Повреждённый массив идентификаторов")
        if np.any((cells < 2) & (ids != 0)) or np.any((cells >= 2) & (ids == 0)):
            raise ValueError("Идентификаторы не соответствуют расположению животных")
        if not isinstance(data, dict) or not isinstance(data.get("records"), list):
            raise ValueError("Повреждённая родословная")
        ledger = cls(cells.shape)
        ledger.ids[:] = ids
        for raw in data["records"]:
            if not isinstance(raw, dict):
                raise ValueError("Повреждённая запись о животном")
            try:
                r = Individual(**raw)
                r.genome = tuple(r.genome)
            except (TypeError, ValueError) as exc:
                raise ValueError("Повреждённая запись о животном") from exc
            if (type(r.id) is not int or r.id <= 0 or r.species not in (2, 3)
                    or r.id in ledger.records or r.depth < 0 or r.born < 0
                    or r.parent < 0 or r.founder < 1 or len(r.genome) != 3
                    or not all(np.isfinite(v) for v in r.genome)):
                raise ValueError("Некорректный идентификатор или геном")
            ledger.records[r.id] = r
        try:
            ledger.next_id = int(data["next_id"])
            ledger.lost_records = int(data.get("lost_records", 0))
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError("Повреждённые счётчики родословной") from exc
        if (ledger.next_id <= max(ledger.records.keys(), default=0)
                or ledger.lost_records < 0):
            raise ValueError("Некорректный следующий идентификатор")
        live_ids = {int(v) for v in ledger.ids[ledger.ids != 0]}
        if len(live_ids) != int(np.count_nonzero(ledger.ids)):
            raise ValueError("Повторяющиеся идентификаторы живых животных")
        for y, x in np.argwhere(cells >= 2):
            record = ledger.records.get(int(ids[y, x]))
            if record is None or record.died is not None or record.species != int(cells[y, x]):
                raise ValueError("Отсутствующая или неправильная запись живого животного")
        return ledger

    def export_csv(self, path: str | Path) -> None:
        columns = ("id", "species", "parent", "founder", "depth", "born", "died",
                   "cause", "offspring", "mutated", "metabolism", "sight", "fertility")
        with open(path, "w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for uid in sorted(self.records):
                r = self.records[uid]
                writer.writerow({"id": r.id, "species": r.species, "parent": r.parent,
                                 "founder": r.founder, "depth": r.depth, "born": r.born,
                                 "died": "" if r.died is None else r.died, "cause": r.cause,
                                 "offspring": r.offspring, "mutated": int(r.mutated),
                                 "metabolism": r.genome[0], "sight": r.genome[1],
                                 "fertility": r.genome[2]})
