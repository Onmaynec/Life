"""Regressions from GitHub issues #4-#10, version 0.8.1."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from life_lab.chrome import _rgb, dark_titlebar
from life_lab.experiment_archive import ImportedStudy, StudyCollection, import_study
from life_lab.experiment_suite import ExperimentPlan, run_experiment
from life_lab.instruments import spatial_field
from life_lab.layout import permitted_panels, dialog_dimensions, window_dimensions
from life_lab.lineage_analysis import compare_families, living_families, trait_map, history, export_family_history
from life_lab.smoke import run_smoke
from life_lab.storage import save_world, load_world
from life_lab.world import CellularAutomaton, Ecosystem
from life_lab.preferences import Preferences


class ExperimentArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.p = Path(self.temp.name)/"archive.json"
        self.plan = ExperimentPlan(variants=2,repetitions=1,steps=5,sample_every=2,
                                   width=20,height=20,minimum=.1,maximum=.2)
        self.session = run_experiment(self.plan)
        self.session.export_json(self.p)

    def test_round_trip_and_missing_metrics(self):
        study = import_study(self.p)
        self.assertEqual(len(study.summary),2)
        self.assertEqual(len(study.series),8)
        self.assertTrue(study.complete)
        self.assertTrue(study.curve("population"))
        self.assertEqual(study.curve("soil"),[])

    def test_curves_average_repetitions_without_mixing_variants(self):
        study = ImportedStudy("synthetic", self.plan, True, False, (), (
            {"variant": 1, "step": 0, "population": 10},
            {"variant": 1, "step": 0, "population": 14},
            {"variant": 2, "step": 0, "population": 100},
        ))
        self.assertEqual(study.curve("population", 1), [(0, 12.0)])
        self.assertEqual(study.curve("population", 2), [(0, 100.0)])
        with self.assertRaises(ValueError):
            study.curve("population", 3)

    def test_atomic_import_and_malformed_json(self):
        coll = StudyCollection()
        coll.add(self.p)
        before = list(coll.studies)
        self.p.write_text("{broken",encoding="utf-8")
        with self.assertRaises(ValueError): coll.add(self.p)
        self.assertEqual(coll.studies,before)

    def test_wrong_version_rejected(self):
        doc = json.loads(self.p.read_text(encoding="utf-8"))
        doc["version"] = 99
        self.p.write_text(json.dumps(doc))
        with self.assertRaises(ValueError): import_study(self.p)

    def test_missing_columns_rejected(self):
        doc=json.loads(self.p.read_text())
        doc["series"][0].pop("population")
        self.p.write_text(json.dumps(doc))
        with self.assertRaises(ValueError): import_study(self.p)

    def test_duplicate_sample_rejected(self):
        doc=json.loads(self.p.read_text())
        doc["series"].append(dict(doc["series"][0]))
        self.p.write_text(json.dumps(doc))
        with self.assertRaises(ValueError): import_study(self.p)

    def test_invalid_number_rejected(self):
        doc=json.loads(self.p.read_text())
        doc["summary"][0]["final_population"]="NaN"
        self.p.write_text(json.dumps(doc))
        with self.assertRaises(ValueError): import_study(self.p)

    def test_partial_runs(self):
        doc=json.loads(self.p.read_text())
        doc["completed"]=False
        doc["cancelled"]=True
        doc["summary"].pop()
        self.p.write_text(json.dumps(doc))
        self.assertEqual(import_study(self.p).completed_runs,1)

    def test_incompatible_settings_flagged(self):
        original=import_study(self.p)
        doc=json.loads(self.p.read_text())
        doc["plan"]["sample_every"]=5
        self.p.write_text(json.dumps(doc))
        another=import_study(self.p)
        self.assertFalse(original.compatible_with(another)[0])

    def test_combined_csv_records_origin(self):
        c=StudyCollection();c.add(self.p);c.add(self.p)
        out=Path(self.temp.name)/"all.csv";c.export_csv(out,"population")
        text=out.read_text(encoding="utf-8-sig")
        self.assertIn("source",text.splitlines()[0])
        self.assertIn("archive.json",text)
        self.assertEqual(len(text.splitlines()),len(self.session.series)*2+1)


class LineageAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.world=Ecosystem(20,20,seed=4)
        self.world.paint(2,2,2)
        self.world.paint(8,8,3)
        self.a=int(self.world.lineage.ids[2,2]); self.b=int(self.world.lineage.ids[8,8])
        self.world.trait[2,2] = 1.3
        self.world.sight[2,2] = 2.5

    def test_living_genetic_means(self):
        report=compare_families(self.world,self.a,self.b)
        self.assertEqual(report[0]["living"],1)
        self.assertAlmostEqual(report[0]["metabolism"],1.3,places=4)
        self.assertEqual(sum(report[0]["metabolism_distribution"]),1)

    def test_missing_family_has_null_means(self):
        self.world.paint(2,2,0)
        rep=compare_families(self.world,self.a,self.b)
        self.assertIsNone(rep[0]["sight"])
        self.assertEqual(rep[0]["living"],0)

    def test_maps_read_only_and_masked(self):
        before=self.world.trait.copy()
        arr=trait_map(self.world,"Метаболизм",self.a)
        self.assertTrue(np.isnan(arr[8,8]))
        self.assertAlmostEqual(float(arr[2,2]),1.3,places=4)
        np.testing.assert_array_equal(self.world.trait,before)

    def test_family_history_export(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"families.csv"
            export_family_history(p,self.world,self.a,self.b)
            self.assertIn("founder",p.read_text(encoding="utf-8-sig"))
            self.assertTrue(history(self.world,(self.a,self.b)))

    def test_save_load_genetics(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"test.lifelab"
            save_world(self.world,p)
            restored=load_world(p)
            self.assertEqual(living_families(self.world).keys(),living_families(restored).keys())


class SpatialAndResponsiveTests(unittest.TestCase):
    def test_invalid_saved_palette_uses_neutral_default(self):
        self.assertEqual(Preferences.from_dict({"palette": "missing"}).palette, "Графит")
    def test_spatial_aggregation(self):
        world=CellularAutomaton(20,20)
        world.cells[:3,:3] = 1
        mean=spatial_field(world,"Плотность",max_side=10,aggregation="mean")
        maxv=spatial_field(world,"Плотность",max_side=10,aggregation="max")
        sumv=spatial_field(world,"Плотность",max_side=10,aggregation="sum")
        self.assertEqual(mean.shape,(10,10))
        self.assertAlmostEqual(float(mean[0,0]),1)
        self.assertEqual(float(maxv[1,1]),1)
        self.assertEqual(float(sumv[0,0]),4)
        self.assertEqual(int(world.cells.sum()),9)

    def test_invalid_aggregation(self):
        w=CellularAutomaton(20,20)
        with self.assertRaises(ValueError): spatial_field(w,"Плотность",aggregation="banana")
        with self.assertRaises(ValueError): spatial_field(w,"Плотность",max_side=0)

    def test_responsive_breakpoints(self):
        self.assertEqual(permitted_panels(900),0)
        self.assertEqual(permitted_panels(1152),1)
        self.assertEqual(permitted_panels(1920),2)
        self.assertEqual(window_dimensions(950,640),(900,555))
        self.assertEqual(dialog_dimensions(900,500,470,650),(470,410))

    def test_windows_colour_encoding(self):
        self.assertEqual(_rgb("#112233"),0x332211)
        with self.assertRaises(ValueError):_rgb("red")
        self.assertFalse(dark_titlebar(object()))  # Linux: explicit safe no-op

    def test_packaged_smoke_checks(self):
        run_smoke()


if __name__ == '__main__':
    unittest.main()
