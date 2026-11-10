"""Reproducibility and file-format checks for parameter sweeps."""
import csv
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from life_lab.ecology import EcologySettings
from life_lab.experiment_suite import ExperimentPlan, ExperimentSession, run_experiment
from life_lab.experiment_ui import series_for_chart
from life_lab.world import CellularAutomaton, Ecosystem


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.plan = ExperimentPlan(width=30,height=22,steps=9,variants=3,repetitions=2,sample_every=4)

    def test_plan_valid(self):
        self.plan.validate()
        self.assertEqual(self.plan.values(),(0.12,0.24,0.36))

    def test_invalid_range_and_duplicate_integer_values(self):
        for case in (replace(self.plan,minimum=0.7,maximum=0.3),
                     replace(self.plan,minimum=float('nan')),
                     replace(self.plan,mode='ecosystem',parameter='season_length',minimum=60,
                             maximum=61,variants=6)):
            with self.subTest(case=case):
                with self.assertRaises(ValueError):
                    case.validate()

    def test_workload_limit(self):
        with self.assertRaisesRegex(ValueError,"16 000"):
            replace(self.plan,steps=1500,variants=12,repetitions=3).validate()

    def test_invalid_ecology_mix(self):
        with self.assertRaisesRegex(ValueError,"100%"):
            replace(self.plan,mode='ecosystem',parameter='plants',minimum=0.8,
                    maximum=0.99).validate()

    def test_snapshot_requires_compatible_world_and_setting(self):
        case=replace(self.plan,mode='ecosystem',parameter='rain',start_mode='snapshot',
                     minimum=0.2,maximum=0.8)
        with self.assertRaises(ValueError):
            case.validate()
        with self.assertRaises(ValueError):
            case.validate(CellularAutomaton(30,22))
        case.validate(Ecosystem(30,22))
        with self.assertRaises(ValueError):
            replace(case,parameter='plants').validate(Ecosystem(30,22))

    def test_population_history_includes_initial_and_last(self):
        result=run_experiment(self.plan)
        self.assertEqual(len(result.summary),6)
        self.assertEqual(result.steps_done,54)
        self.assertEqual(len(result.series),6*4)  # 0, 4, 8, 9
        self.assertEqual(result.progress,1)
        for case in result.summary:
            self.assertEqual(case['steps'],9)
            samples=[row['step'] for row in result.series if row['variant']==case['variant']
                     and row['repetition']==case['repetition']]
            self.assertEqual(samples,[0,4,8,9])
            self.assertGreaterEqual(case['mean_population'],0)
            values=[row for row in result.series if row["variant"]==case["variant"] and row["repetition"]==case["repetition"]]
            self.assertEqual(sum(r["births"] for r in values),case["total_births"])
            self.assertEqual(sum(r["deaths"] for r in values),case["total_deaths"])

    def test_same_plan_and_seed_identical(self):
        a=run_experiment(self.plan)
        b=run_experiment(self.plan)
        self.assertEqual(a.summary,b.summary)
        self.assertEqual(a.series,b.series)

    def test_replicates_paired_to_same_seed(self):
        result=run_experiment(self.plan)
        by_rep={}
        for row in result.series:
            if row['step']==0:
                by_rep.setdefault(row['repetition'],set()).add(row['seed'])
        self.assertTrue(all(len(seeds)==1 for seeds in by_rep.values()))
        self.assertEqual(len(by_rep),2)
        baseline=result.comparison()[0]
        self.assertEqual(baseline['paired_delta'],0)
        self.assertIsNotNone(baseline['std'])
        self.assertEqual(result.comparison()[2]['n'],2)

    def test_ecosystem_fresh_reproducible(self):
        case=replace(self.plan,mode='ecosystem',parameter='rain',minimum=0.1,maximum=0.9,
                     variants=2,repetitions=2,steps=5)
        a,b=run_experiment(case),run_experiment(case)
        self.assertEqual(a.summary,b.summary)
        self.assertEqual(a.series,b.series)
        self.assertTrue(a.summary)

    def test_ecosystem_snapshot_is_not_modified(self):
        initial=Ecosystem(30,22,seed=15)
        initial.randomize(seed=15)
        initial.ecology=EcologySettings(enabled=True,rain=0.5)
        before=initial.cells.copy(); energy=initial.energy.copy()
        rng=initial._rng.bit_generator.state
        case=replace(self.plan,mode='ecosystem',parameter='rain',start_mode='snapshot',
                     minimum=0.1,maximum=0.9,variants=2,repetitions=2,steps=3)
        a=run_experiment(case,initial)
        b=run_experiment(case,initial)
        self.assertEqual(a.summary,b.summary)
        np.testing.assert_array_equal(initial.cells,before)
        np.testing.assert_array_equal(initial.energy,energy)
        self.assertEqual(initial._rng.bit_generator.state,rng)
        self.assertEqual(initial.generation,0)

    def test_cancel_stops_generations(self):
        session=ExperimentSession(self.plan)
        session.advance()
        self.assertEqual(session.steps_done,1)
        session.cancel()
        self.assertFalse(session.advance())
        self.assertTrue(session.cancelled)
        self.assertFalse(session.complete)

    def test_early_stop_and_real_step_counter(self):
        initial=Ecosystem(20,20,seed=19)
        initial.cells[2,2]=2
        initial.energy[2,2]=0.01
        initial.cells[4,4]=3
        initial.energy[4,4]=0.01
        case=replace(self.plan,mode='ecosystem',parameter='rain',start_mode='snapshot',
                     minimum=0.2,maximum=0.8,variants=2,repetitions=1,steps=20,
                     stop_on_extinction=True)
        session=run_experiment(case,initial)
        self.assertTrue(session.complete)
        self.assertEqual(session.progress,1)
        self.assertLess(session.steps_done,session.total_steps)
        self.assertTrue(all(row['steps']<20 for row in session.summary))

    def test_exports_have_stable_headers_and_nulls(self):
        result=run_experiment(self.plan)
        with TemporaryDirectory() as folder:
            summary=Path(folder)/'summary.csv'
            series=Path(folder)/'series.csv'
            raw=Path(folder)/'results.json'
            result.export_summary_csv(summary)
            result.export_series_csv(series)
            result.export_json(raw)
            result.export_report(Path(folder)/"study.md")
            with summary.open(encoding='utf-8-sig',newline='') as stream:
                summary_rows=list(csv.DictReader(stream))
            with series.open(encoding='utf-8-sig',newline='') as stream:
                series_rows=list(csv.DictReader(stream))
            document=json.loads(raw.read_text(encoding='utf-8'))
            self.assertEqual(len(summary_rows),6)
            self.assertEqual(len(series_rows),24)
            self.assertEqual(document['version'],1)
            self.assertEqual(len(document['summary']),6)
            self.assertIsNone(document['series'][0]['soil'])
            self.assertEqual(summary_rows[0]['parameter'],'density')
            self.assertIn("попарная",(Path(folder)/"study.md").read_text(encoding="utf-8"))

    def test_averaged_curves_use_generation_steps(self):
        result=run_experiment(self.plan)
        chart=series_for_chart(result,'population')
        self.assertEqual(sorted(chart),[1,2,3])
        self.assertEqual([step for step,_ in chart[1]],[0,4,8,9])
        with self.assertRaises(ValueError):
            series_for_chart(result,'not_a_metric')


if __name__=='__main__':
    unittest.main()
