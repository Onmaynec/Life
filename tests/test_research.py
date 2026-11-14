"""Regression checks for v0.5 quantitative tools, climate and editor transforms."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from life_lab.ecology import EcologySettings, _diffuse, climate_values
from life_lab.instruments import MEASURES, ResearchLog, _clusters, measure, spatial_field
from life_lab.pattern_tools import transform, shift, fill_region
from life_lab.patterns import PATTERNS, pattern_array
from life_lab.storage import load_document, load_research, load_world, save_world
from life_lab.visuals import PALETTES, HEATMAP_FIELDS, heatmap_view
from life_lab.world import CellularAutomaton, Ecosystem


class EditorResearchTests(unittest.TestCase):
    def test_library_contains_at_least_25_patterns(self):
        self.assertGreaterEqual(len(PATTERNS),25)
        for name in PATTERNS:
            with self.subTest(name=name):
                pattern=pattern_array(name)
                self.assertGreater(int(pattern.sum()),0)
                self.assertTrue(np.isin(pattern,[0,1]).all())

    def test_rotation_mirrors_are_lossless(self):
        pat=pattern_array('Глайдер')
        shifted=pat.copy()
        for _ in range(4):
            shifted=transform(shifted,'clockwise')
        np.testing.assert_array_equal(shifted,pat)
        np.testing.assert_array_equal(transform(transform(pat,'flip_vertical'),'flip_vertical'),pat)
        np.testing.assert_array_equal(transform(transform(pat,'flip_horizontal'),'flip_horizontal'),pat)
        np.testing.assert_array_equal(transform(transform(pat,'clockwise'),'counterclockwise'),pat)
        with self.assertRaises(ValueError):
            transform(pat,'invalid')

    def test_shift_clips_and_wraps(self):
        cells=np.zeros((3,3),dtype=np.uint8)
        cells[1,2]=1
        self.assertEqual(int(shift(cells,1,0,wrap=False).sum()),0)
        self.assertEqual(int(shift(cells,1,0,wrap=True)[1,0]),1)
        self.assertEqual(int(shift(cells,0,1,wrap=False)[2,2]),1)

    def test_fill_rectangle_clips_and_sorts(self):
        cells=np.zeros((6,6),dtype=np.uint8)
        fill_region(cells,4,4,2,2,1)
        self.assertEqual(int(cells.sum()),9)
        fill_region(cells,-100,-100,3,3,0)
        self.assertEqual(int(cells.sum()),5)

    def test_clusters_are_cardinal_not_diagonal(self):
        grid=np.array([[1,0,0],[0,1,1],[0,0,1]],dtype=bool)
        self.assertEqual(_clusters(grid),(2,3))

    def test_clusters_follow_toroidal_edges(self):
        grid = np.zeros((5, 5), dtype=bool)
        grid[2, 0] = grid[2, 4] = True
        self.assertEqual(_clusters(grid, wrap=False), (2, 1))
        self.assertEqual(_clusters(grid, wrap=True), (1, 2))

    def test_readonly_measurement_and_empty_world(self):
        world=CellularAutomaton(20,20)
        world.paint(3,3,1);world.paint(4,3,1)
        before=world.cells.copy()
        m=measure(world)
        np.testing.assert_array_equal(before,world.cells)
        self.assertEqual(m['clusters'],1)
        self.assertEqual(m['population'],2)
        self.assertEqual(m['diversity'],1)
        self.assertAlmostEqual(m['occupancy'],0.5)
        self.assertIsNone(m['energy'])
        world.clear()
        self.assertEqual(measure(world)['entropy'],0)

    def test_spatial_fields_are_downsampled_and_nonmutating(self):
        world=Ecosystem(30,20,seed=7)
        world.randomize(seed=7)
        copy=world.nutrients.copy()
        for name in ('Плотность','Энергия','Минералы','Влага','Возраст','Органика','Токсины','Рельеф'):
            field=spatial_field(world,name,max_side=7)
            self.assertLessEqual(field.shape[0],7)
            self.assertLessEqual(field.shape[1],7)
            self.assertTrue(np.isfinite(field).all())
        np.testing.assert_array_equal(copy,world.nutrients)
        with self.assertRaises(ValueError):
            spatial_field(world,'Несуществующий')

    def test_cluster_measurements_are_not_repeated_as_fresh_data(self):
        world = CellularAutomaton(20, 20)
        world.cells[2, 2] = 1
        log = ResearchLog()
        log.capture(world)
        world.generation = 1
        world.cells[2, 2] = 0
        world.cells[2, 3] = 1
        log.capture(world)
        self.assertIsNone(log.records[-1]["clusters"])
        self.assertIsNone(log.records[-1]["largest_cluster"])
        self.assertIsNone(log.records[-1]["cluster_generation"])
        world.generation = 5
        log.capture(world)
        self.assertEqual(log.records[-1]["cluster_generation"], 5)
        self.assertIsNotNone(log.records[-1]["clusters"])

    def test_legacy_unsampled_cluster_values_are_invalidated(self):
        world = CellularAutomaton(20, 20)
        row = measure(world)
        row["generation"] = 1
        row["clusters"] = 7
        row["largest_cluster"] = 9
        row.pop("cluster_generation")
        restored = ResearchLog.from_records([row])
        self.assertIsNone(restored.records[0]["clusters"])
        self.assertIsNone(restored.records[0]["largest_cluster"])

    def test_research_log_export_and_validation(self):
        world=CellularAutomaton(20,20)
        log=ResearchLog()
        log.capture(world)
        world.step()
        log.capture(world)
        self.assertEqual(len(log.records),2)
        self.assertEqual(set(log.records[0]),set(MEASURES))
        self.assertEqual(ResearchLog.from_records(log.records).records,log.records)
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'metrics.csv'
            log.export_csv(file)
            self.assertIn('occupancy',file.read_text(encoding='utf-8-sig').splitlines()[0])
        with self.assertRaises(ValueError):
            ResearchLog.from_records([{'generation':-1}])

    def test_heatmap_has_fixed_intensity_and_is_readonly(self):
        values=np.array([[0,.5,1]],dtype=np.float32)
        cells=np.zeros((1,3),dtype=np.uint8)
        for name in ('Графит','Оникс','Титан','Аметист'):
            picture=heatmap_view(values,cells,zoom=9,palette_name=name,grid=True)
            self.assertEqual(picture.size,(27,9))
            self.assertNotEqual(picture.getpixel((4,4)),picture.getpixel((22,4)))
        np.testing.assert_array_equal(values,np.array([[0,.5,1]],dtype=np.float32))
        self.assertIn('Организмы',HEATMAP_FIELDS)


class ClimateTests(unittest.TestCase):
    def test_settings_validate_and_roundtrip(self):
        original=EcologySettings(enabled=True,heat=.22,season_length=250,toxins=True)
        self.assertEqual(EcologySettings.from_dict(original.as_dict()),original)
        for bad in ({'rain':float('nan')},{'enabled':1},{'season_length':1},{'wildfire_chance':2}):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                EcologySettings.from_dict(bad)

    def test_seasons_are_periodic(self):
        settings=EcologySettings(enabled=True,season_length=80)
        one=climate_values(17,settings)
        repeated=climate_values(97,settings)
        self.assertEqual(one['season'],repeated['season'])
        self.assertAlmostEqual(float(one['temperature']),float(repeated['temperature']))

    def test_disabled_season_cycle_has_no_seasonal_modifiers(self):
        settings = EcologySettings(enabled=True, season_cycle=False, solar_cycle=False)
        first, later = climate_values(0, settings), climate_values(1200, settings)
        self.assertEqual(first["season"], "Без сезона")
        self.assertEqual(later["season"], "Без сезона")
        self.assertEqual(first["temperature"], later["temperature"])
        self.assertEqual(first["sunlight"], later["sunlight"])

    def test_diffusion_conserves_total_without_wrap(self):
        for wrap in (True,False):
            field=np.zeros((5,5),dtype=np.float32)
            field[2,2]=100
            _diffuse(field,.4,wrap)
            self.assertAlmostEqual(float(field.sum()),100,places=4)
            self.assertLess(field[2,2],100)

    def test_disabled_extensions_leave_moisture_unchanged(self):
        world=Ecosystem(20,20,seed=5)
        world.randomize(seed=5)
        before=world.moisture.copy()
        world.step()
        np.testing.assert_array_equal(world.moisture,before)

    def test_enabled_ecology_is_seeded_and_finite(self):
        a=Ecosystem(30,25,seed=18)
        b=Ecosystem(30,25,seed=18)
        for w in (a,b):
            w.randomize(seed=117)
            w.ecology=EcologySettings(enabled=True,toxins=True,wildfire=True)
        for _ in range(9):
            ea,eb=a.step(),b.step()
            self.assertEqual(ea,eb)
        for key in ('cells','energy','trait','sight','fertility','age','moisture','seeds','organic','toxins'):
            np.testing.assert_array_equal(getattr(a,key),getattr(b,key))
        self.assertTrue(np.isfinite(a.nutrients).all())
        self.assertGreaterEqual(a.moisture.min(),0)
        self.assertLessEqual(a.moisture.max(),100)

    def test_research_save_preserves_rng_and_all_arrays(self):
        world=Ecosystem(30,25,seed=6)
        world.randomize(seed=12)
        world.ecology=EcologySettings(enabled=True,disease=True)
        logs=ResearchLog()
        for _ in range(4):
            logs.capture(world,world.step())
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'study.lifelab'
            save_world(world,path,research=logs)
            copy=load_world(path)
            recovered=load_research(path)
            self.assertEqual(recovered.records,logs.records)
            for _ in range(4):
                a=world.step();b=copy.step()
                self.assertEqual(a,b)
                for attr in ('cells','energy','age','moisture','seeds','organic','altitude'):
                    np.testing.assert_array_equal(getattr(world,attr),getattr(copy,attr))

    def test_legacy_v3_save_still_opens(self):
        original=Ecosystem(20,20,seed=8)
        original.randomize(seed=9)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'legacy.lifelab'
            save_world(original,path)
            with np.load(path,allow_pickle=False) as data:
                meta=json.loads(str(data['metadata'].item()))
                meta['version']=3
                old={key:data[key] for key in data.files if key not in
                     ('metadata','moisture','age','seeds','organic','toxins','altitude')}
            with path.open('wb') as stream:
                np.savez_compressed(stream,metadata=np.array(json.dumps(meta)),**old)
            restored=load_world(path)
            self.assertFalse(restored.ecology.enabled)
            np.testing.assert_array_equal(restored.cells,original.cells)
            self.assertEqual(load_research(path).records,[])


if __name__=='__main__':
    unittest.main()
