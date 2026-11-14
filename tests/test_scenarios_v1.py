"""Release 1.0: paired intervention experiments and DPI-safe layout."""
from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import tempfile
import unittest

import numpy as np

from life_lab.layout import logical_width, permitted_panels
from life_lab.scenarios import (INTERVENTION_KINDS, PRESETS, METRICS,
                                Intervention, ScenarioPlan, ScenarioSession, verify_bundle)
from life_lab.world import CellularAutomaton, Ecosystem


class ScenarioTests(unittest.TestCase):
    def setUp(self):
        self.world = Ecosystem(25, 21, seed=7)
        self.world.randomize(seed=103)
        self.world.ecology.enabled = True

    @staticmethod
    def plan(events=(Intervention(0,"moisture",-20),), *, steps=12, sample=4):
        return ScenarioPlan(steps=steps, sample_every=sample, interventions=tuple(events))

    def test_initial_branches_are_identical(self):
        study = ScenarioSession(self.world, self.plan())
        np.testing.assert_array_equal(study.reference.cells, study.treatment.cells)
        np.testing.assert_array_equal(study.reference.nutrients, study.treatment.nutrients)
        self.assertEqual(study.samples[0]["population"],study.samples[1]["population"])
        self.assertIsNot(study.reference, self.world)
        self.assertIsNot(study.reference, study.treatment)

    def test_original_untouched_after_full_study(self):
        start_generation = self.world.generation
        original_cells = self.world.cells.copy()
        original_moisture = self.world.moisture.copy()
        rng = json.dumps(self.world._rng.bit_generator.state,sort_keys=True)
        family = self.world.lineage.ids.copy()
        study = ScenarioSession(self.world,self.plan()).finish()
        self.assertTrue(study.done)
        self.assertEqual(self.world.generation,start_generation)
        np.testing.assert_array_equal(self.world.cells,original_cells)
        np.testing.assert_array_equal(self.world.moisture,original_moisture)
        np.testing.assert_array_equal(self.world.lineage.ids,family)
        self.assertEqual(json.dumps(self.world._rng.bit_generator.state,sort_keys=True),rng)

    def test_no_op_produces_identical_trajectories(self):
        study=ScenarioSession(self.world,self.plan([Intervention(0,"moisture",0)])).finish()
        for a,b in zip(study.samples[::2],study.samples[1::2]):
            aa=dict(a);bb=dict(b);aa.pop("branch");bb.pop("branch")
            self.assertEqual(aa,bb)
        np.testing.assert_array_equal(study.reference.cells,study.treatment.cells)
        np.testing.assert_array_equal(study.reference.lineage.ids,study.treatment.lineage.ids)
        self.assertEqual(json.dumps(study.reference._rng.bit_generator.state,sort_keys=True),
                         json.dumps(study.treatment._rng.bit_generator.state,sort_keys=True))

    def test_repeatable_with_same_source(self):
        plan=self.plan(events=[Intervention(1,"nutrients",-30),Intervention(5,"moisture",15)])
        first=ScenarioSession(self.world,plan).finish()
        second=ScenarioSession(self.world,plan).finish()
        self.assertEqual(first.samples,second.samples)

    def test_intervention_only_treatment(self):
        study=ScenarioSession(self.world,self.plan([Intervention(0,"nutrients",40)],steps=1,sample=1))
        reference=study.reference.nutrients.copy()
        study.advance()
        self.assertEqual(len(study.fired),1)
        self.assertGreater(study.treatment.nutrients.mean(),study.reference.nutrients.mean())
        self.assertFalse(np.shares_memory(study.reference.nutrients,study.treatment.nutrients))
        self.assertEqual(study.completed,1)
        self.assertFalse(study.advance())
        self.assertTrue(np.any(study.reference.nutrients != reference))

    def test_staggered_events_and_final_sample(self):
        events=[Intervention(0,"moisture",-10),Intervention(3,"nutrients",-15),
                Intervention(3,"moisture",10)]
        s=ScenarioSession(self.world,self.plan(events,steps=7,sample=4)).finish()
        self.assertEqual(len(s.fired),3)
        self.assertEqual([x["step"] for x in s.samples[::2]],[0,4,7])
        self.assertEqual(s.summary()["completed"],7)

    def test_replay_bundle_contains_worlds_and_checksums(self):
        from life_lab.storage import load_world
        s=ScenarioSession(self.world,self.plan(steps=7,sample=3)).finish()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"study.zip"
            s.export_bundle(p)
            verified=verify_bundle(p)
            self.assertTrue(verified["valid"])
            self.assertEqual(verified["completed"],7)
            with ZipFile(p) as archive:
                self.assertEqual(len(archive.namelist()),6)
                archive.extract("initial.lifelab",d)
                archive.extract("control.lifelab",d)
                archive.extract("scenario.lifelab",d)
            self.assertEqual(load_world(Path(d)/"initial.lifelab").generation,0)
            self.assertEqual(load_world(Path(d)/"control.lifelab").generation,7)
            self.assertEqual(load_world(Path(d)/"scenario.lifelab").generation,7)

    def test_bundle_tampering_is_detected(self):
        s=ScenarioSession(self.world,self.plan(steps=2,sample=1)).finish()
        with tempfile.TemporaryDirectory() as d:
            original=Path(d)/"original.zip";bad=Path(d)/"bad.zip"
            s.export_bundle(original)
            with ZipFile(original) as archive, ZipFile(bad,"w",ZIP_DEFLATED) as altered:
                for name in archive.namelist():
                    contents=archive.read(name)
                    if name == "measurements.csv":contents+=b"tampered"
                    altered.writestr(name,contents)
            with self.assertRaises(ValueError):verify_bundle(bad)

    def test_bundle_unknown_file_rejected(self):
        s=ScenarioSession(self.world,self.plan(steps=1,sample=1)).finish()
        with tempfile.TemporaryDirectory() as d:
            original=Path(d)/"original.zip";bad=Path(d)/"bad.zip"
            s.export_bundle(original)
            with ZipFile(original) as archive, ZipFile(bad,"w",ZIP_DEFLATED) as altered:
                for name in archive.namelist():altered.writestr(name,archive.read(name))
                altered.writestr("../evil.txt", "no")
            with self.assertRaises(ValueError):verify_bundle(bad)

    def test_partial_export_csv_json(self):
        s=ScenarioSession(self.world,self.plan(steps=4,sample=2))
        s.advance()
        with tempfile.TemporaryDirectory() as directory:
            csv_path=Path(directory)/"report.csv"
            json_path=Path(directory)/"report.json"
            s.export_csv(csv_path);s.export_json(json_path)
            text=csv_path.read_text(encoding="utf-8-sig")
            self.assertIn("branch,step,generation",text.splitlines()[0])
            doc=json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(doc["format"],"life-scenario-1")
            self.assertEqual(doc["completed"],1)
            self.assertEqual(len(doc["samples"]),2)
            self.assertEqual(len(doc["applied_events"]),1)

    def test_all_metrics_available(self):
        s=ScenarioSession(self.world,self.plan()).finish()
        for key in METRICS.values():
            data=s.curves(key)
            self.assertEqual(sorted(data),["control","scenario"])
            self.assertEqual(len(data["control"]),4)
            self.assertEqual(len(data["scenario"]),4)
        with self.assertRaises(ValueError):s.curves("unknown")

    def test_preset_library_is_valid(self):
        for name,events in PRESETS.items():
            with self.subTest(name=name):
                self.plan(events=events,steps=120).validate(self.world)
                self.assertGreaterEqual(len(events),1)
                self.assertLessEqual(len(events),32)

    def test_classic_not_supported(self):
        with self.assertRaises(ValueError):ScenarioSession(CellularAutomaton(20,20),self.plan())

    def test_out_of_range_generation(self):
        for bad in (-1,3,1.5,True):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                ScenarioSession(self.world,self.plan([Intervention(bad,"moisture",-10)],steps=3))

    def test_invalid_numeric_values(self):
        for bad in (float("nan"),float("inf"),float("-inf"),"10"):
            with self.subTest(bad=str(bad)),self.assertRaises(ValueError):
                ScenarioSession(self.world,self.plan([Intervention(0,"moisture",bad)]))

    def test_invalid_event_names(self):
        with self.assertRaises(ValueError):ScenarioSession(self.world,self.plan([Intervention(0,"banana",1)]))
        self.assertEqual(len(INTERVENTION_KINDS),8)

    def test_ecology_required_for_climate_events(self):
        self.world.ecology.enabled=False
        for kind in ("rain","heat","disease","wildfire","water_flow"):
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                ScenarioSession(self.world,self.plan([Intervention(0,kind,1)]))

    def test_climate_switches_and_setters(self):
        s=ScenarioSession(self.world,self.plan([
            Intervention(0,"rain",0.9),Intervention(0,"heat",0.8),
            Intervention(0,"disease",1),Intervention(0,"wildfire",1),
            Intervention(0,"water_flow",0.1)],steps=1,sample=1))
        s.advance()
        self.assertEqual(s.treatment.ecology.rain,.9)
        self.assertEqual(s.treatment.ecology.heat,.8)
        self.assertTrue(s.treatment.ecology.disease)
        self.assertTrue(s.treatment.ecology.wildfire)
        self.assertEqual(s.reference.ecology.rain,.6)

    def test_bad_climate_ranges(self):
        for kind,val in (("rain",-1),("heat",2),("disease",.5),("wildfire",-1)):
            with self.subTest(kind=kind),self.assertRaises(ValueError):
                ScenarioSession(self.world,self.plan([Intervention(0,kind,val)]))

    def test_bad_plan_limits(self):
        for steps,sample,events in ((0,1,(Intervention(0,"moisture",1),)),
                                     (1001,1,(Intervention(0,"moisture",1),)),
                                     (20,0,(Intervention(0,"moisture",1),)),
                                     (20,21,(Intervention(0,"moisture",1),)),
                                     (20,1,()),
                                     (20,1,tuple(Intervention(0,"moisture",1) for _ in range(33)))):
            with self.subTest(steps=steps,sample=sample,events=len(events)),self.assertRaises(ValueError):
                ScenarioSession(self.world,ScenarioPlan(steps,sample,events))


class ResponsiveScaleTests(unittest.TestCase):
    def test_logical_width(self):
        self.assertEqual(logical_width(1440,144),960)
        self.assertEqual(logical_width(1200,96),1200)
        self.assertEqual(logical_width(1000,0),1000)
        self.assertEqual(logical_width(1000,float("nan")),1000)

    def test_layout_at_125_and_150_percent(self):
        self.assertEqual(permitted_panels(1366,120),1)
        self.assertEqual(permitted_panels(1366,144),0)
        self.assertEqual(permitted_panels(1920,144),1)
        self.assertEqual(permitted_panels(1920,96),2)


if __name__=="__main__":
    unittest.main()
