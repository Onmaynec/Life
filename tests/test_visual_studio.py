"""Life v0.8 visualisation regressions; no display server required."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from life_lab.plotting import contiguous_segments, hex_blend, nearest_sample_index
from life_lab.preferences import Preferences, save_preferences, load_preferences
from life_lab.skins import SKINS, skin
from life_lab.visuals import PALETTES, STYLES, draw_view, heatmap_view, minimap


class StudioColorTests(unittest.TestCase):
    def test_added_themes_use_valid_colors(self):
        self.assertGreaterEqual(len(SKINS), 8)
        for name, theme in SKINS.items():
            with self.subTest(theme=name):
                for color in vars(theme).values():
                    self.assertRegex(color, r"^#[0-9A-Fa-f]{6}$")

    def test_scientific_palette_renders_all_styles(self):
        cells = np.array([[0, 1, 0, 2], [3, 0, 1, 0]], dtype=np.uint8)
        before = cells.copy()
        self.assertGreaterEqual(len(PALETTES), 10)
        self.assertGreaterEqual(len(STYLES), 6)
        for palette in PALETTES:
            with self.subTest(palette=palette):
                for style in STYLES:
                    image = draw_view(cells, mode="ecosystem", zoom=12,
                                      palette_name=palette, style=style)
                    self.assertEqual(image.size, (48, 24))
                    self.assertEqual(image.mode, "RGB")
        np.testing.assert_array_equal(before, cells)

    def test_neon_and_contours_differ_from_existing_styles(self):
        cells = np.array([[1, 0, 2], [0, 3, 0]], dtype=np.uint8)
        images = [draw_view(cells, mode="ecosystem", zoom=16,
                            palette_name="Карбон", style=style).tobytes()
                  for style in ("Неон", "Контуры", "Созвездие", "Мозаика")]
        self.assertEqual(len(images), len(set(images)))

    def test_heatmaps_have_consistent_shape(self):
        data = np.array([[0.0, .5], [1.0, 2.0]], dtype=float)
        cells = np.zeros((2, 2), dtype=np.uint8)
        heat = heatmap_view(data, cells, zoom=8, palette_name="Сигнал")
        self.assertEqual(heat.size, (16, 16))
        self.assertEqual(minimap(cells, palette_name="Сигнал", width=32,
                                 height=12).size, (32, 12))


class PlottingTests(unittest.TestCase):
    def test_hex_blend_ends_and_midpoint(self):
        self.assertEqual(hex_blend("#000000", "#ffffff", 0), "#000000")
        self.assertEqual(hex_blend("#000000", "#ffffff", 1), "#ffffff")
        self.assertEqual(hex_blend("#000000", "#ffffff", .5), "#808080")

    def test_hex_blend_rejects_invalid_values(self):
        for a,b,f in (("blue","#ffffff", .5),("#abc001","#ffffff", 1.2),
                      ("#qqqqqq","#ffffff", .5)):
            with self.subTest(a=a,b=b,f=f), self.assertRaises(ValueError):
                hex_blend(a,b,f)

    def test_missing_values_break_segments_without_changing_time_axis(self):
        rows = [{"x":4},{"x":5},{"x":None},{"x":9},{"x":10},{"x":float("nan")},{"x":0}]
        self.assertEqual(contiguous_segments(rows,"x"),
                         [[(0,4.0),(1,5.0)],[(3,9.0),(4,10.0)],[(6,0.0)]])
        self.assertEqual(rows[2]["x"], None)

    def test_cursor_clamps_to_valid_generation(self):
        self.assertEqual(nearest_sample_index(-10,16,180,10),0)
        self.assertEqual(nearest_sample_index(500,16,180,10),9)
        self.assertEqual(nearest_sample_index(98,16,180,10),4)
        self.assertEqual(nearest_sample_index(98,16,16,0),0)


class StudioPreferencesTests(unittest.TestCase):
    def test_v080_preferences_roundtrip(self):
        prefs=Preferences(skin="Тёмная медь",palette="Северное сияние",
                          style="Созвездие",overlay=False,chart_fill=False)
        with TemporaryDirectory() as directory:
            path=Path(directory)/"preferences.json"
            save_preferences(prefs,path)
            self.assertEqual(load_preferences(path),prefs)

    def test_old_preferences_keep_new_flags_enabled(self):
        prefs=Preferences.from_dict({"skin":"Оникс / лайм","palette":"Оникс"})
        self.assertTrue(prefs.overlay)
        self.assertTrue(prefs.chart_fill)
        self.assertEqual(prefs.skin,"Оникс / лайм")

    def test_invalid_flags_are_not_accepted(self):
        prefs=Preferences.from_dict({"overlay":"false","chart_fill":0,"style":"Неон"})
        self.assertTrue(prefs.overlay)
        self.assertTrue(prefs.chart_fill)
        self.assertEqual(prefs.style,"Неон")


if __name__ == "__main__":
    unittest.main()
