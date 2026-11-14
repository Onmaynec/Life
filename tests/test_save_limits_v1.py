"""Release 1.0: untrusted saves must be bounded before NumPy unpacks them."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np

from life_lab.storage import _check_archive_size, load_document, load_research, save_world
from life_lab.world import CellularAutomaton, Ecosystem


class ArchiveSafetyTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "test.lifelab"

    def test_valid_old_and_new_modes(self):
        for mode in (CellularAutomaton(22, 20), Ecosystem(22, 20, seed=7)):
            save_world(mode, self.path)
            _check_archive_size(self.path)
            self.assertEqual(load_document(self.path)[0].mode, mode.mode)
            self.assertEqual(len(load_research(self.path).records), 0)

    def test_rejects_corrupt_archive(self):
        self.path.write_bytes(b"this is not a zip")
        with self.assertRaises(ValueError):
            load_document(self.path)

    def test_rejects_excessively_many_entries(self):
        with ZipFile(self.path, "w") as archive:
            for i in range(50):
                archive.writestr(f"{i}.npy", b"dummy")
        with self.assertRaises(ValueError):
            load_document(self.path)

    def test_rejects_duplicate_member_names(self):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with ZipFile(self.path, "w") as archive:
                archive.writestr("cells.npy", b"dummy")
                archive.writestr("cells.npy", b"dummy")
        with self.assertRaises(ValueError):
            load_document(self.path)

    def test_rejects_member_before_decompression(self):
        with ZipFile(self.path, "w", ZIP_DEFLATED) as archive:
            archive.writestr("cells.npy", b"x" * 128)
        with mock.patch("life_lab.storage.MAX_MEMBER_BYTES", 100):
            with self.assertRaises(ValueError):
                load_document(self.path)

    def test_rejects_expanded_archive(self):
        with ZipFile(self.path,"w", ZIP_DEFLATED) as archive:
            archive.writestr("cells.npy",b"x"*128)
        with mock.patch("life_lab.storage.MAX_EXPANDED_BYTES",100):
            with self.assertRaises(ValueError):
                load_document(self.path)

    def _rewrite_ecosystem_array(self, key, replacement):
        save_world(Ecosystem(20, 20, seed=3), self.path)
        with np.load(self.path, allow_pickle=False) as archive:
            metadata = archive["metadata"].copy()
            arrays = {name: archive[name].copy() for name in archive.files if name != "metadata"}
        arrays[key] = replacement
        with self.path.open("wb") as stream:
            np.savez_compressed(stream, metadata=metadata, **arrays)

    def test_rejects_numeric_layers_outside_model_ranges(self):
        shape = (20, 20)
        cases = (
            ("nutrients", np.full(shape, -1.0, dtype=np.float64)),
            ("altitude", np.full(shape, 1.5, dtype=np.float64)),
            ("energy", np.full(shape, 1e100, dtype=np.float64)),
            ("age", np.full(shape, 1.5, dtype=np.float64)),
        )
        for key, value in cases:
            with self.subTest(key=key):
                self._rewrite_ecosystem_array(key, value)
                with self.assertRaises(ValueError):
                    load_document(self.path)


if __name__ == "__main__":
    unittest.main()
