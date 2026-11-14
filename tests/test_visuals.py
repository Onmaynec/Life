"""Rendering and local preferences: these tests do not need a display server."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from life_lab.preferences import Preferences, load_preferences, save_preferences
from life_lab.visuals import PALETTES, STYLES, colorize, draw_view, minimap


class RenderingTests(unittest.TestCase):
    def setUp(self):
        self.cells = np.array([[0, 1, 0, 0], [1, 0, 2, 0], [0, 3, 0, 1]], dtype=np.uint8)

    def test_all_palettes_and_styles_render(self):
        for palette in PALETTES:
            for style in STYLES:
                with self.subTest(palette=palette, style=style):
                    rendered = draw_view(self.cells, mode="ecosystem", zoom=11,
                                         palette_name=palette, style=style)
                    self.assertEqual(rendered.size, (44, 33))
                    self.assertEqual(rendered.mode, "RGB")

    def test_styles_and_palettes_change_pixels(self):
        original = self.cells.copy()
        frames = [draw_view(self.cells, mode="ecosystem", zoom=10,
                            palette_name="Аврора", style=style).tobytes() for style in STYLES]
        self.assertEqual(len(set(frames)), len(STYLES))
        another = draw_view(self.cells, mode="ecosystem", zoom=10,
                            palette_name="Биосфера", style="Свечение")
        self.assertNotEqual(frames[0], another.tobytes())
        np.testing.assert_array_equal(original, self.cells)

    def test_trails_show_only_with_input(self):
        cells = np.zeros((3, 4), dtype=np.uint8)
        trails = np.zeros_like(cells)
        trails[1, 2] = 7
        base = colorize(cells, "classic", PALETTES["Аврора"])
        after = colorize(cells, "classic", PALETTES["Аврора"], trails=trails)
        self.assertGreater(sum(after[1, 2]), sum(base[1, 2]))
        np.testing.assert_array_equal(base[1, 0], after[1, 0])
        self.assertEqual(int(trails[1, 2]), 7)

    def test_ecosystem_soil_is_visible_without_animals(self):
        cells = np.zeros((3, 4), dtype=np.uint8)
        soil = np.full_like(cells, 100, dtype=np.float32)
        result = colorize(cells, "ecosystem", PALETTES["Аврора"], nutrients=soil)
        self.assertNotEqual(tuple(result[0, 0]), PALETTES["Аврора"].soil)
        self.assertEqual(minimap(cells, palette_name="Аврора", width=12, height=6,
                                 mode="ecosystem").size, (12, 6))

    def test_invalid_render_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            draw_view(self.cells, mode="classic", zoom=0, palette_name="Аврора")
        with self.assertRaises(ValueError):
            draw_view(self.cells, mode="classic", zoom=10, palette_name="несуществующая")
        with self.assertRaises(ValueError):
            colorize(self.cells, "classic", PALETTES["Аврора"], trails=np.zeros((1, 2)))

    def test_zoom_2_is_supported_for_large_maps(self):
        image = draw_view(np.zeros((25, 50), dtype=np.uint8), mode="classic",
                          zoom=2, palette_name="Ультрафиолет", style="Свечение")
        self.assertEqual(image.size, (100, 50))


class PreferencesTests(unittest.TestCase):
    def test_round_trip(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "preferences.json"
            prefs = Preferences("Биосфера", "Мозаика", False, False, 17)
            save_preferences(prefs, path)
            self.assertEqual(load_preferences(path), prefs)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["speed"], 17)

    def test_broken_or_missing_file_uses_defaults(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            self.assertEqual(load_preferences(path), Preferences())
            path.write_text("{broken", encoding="utf-8")
            self.assertEqual(load_preferences(path), Preferences())

    def test_wrong_values_are_repaired(self):
        actual = Preferences.from_dict({"palette": "неизвестная", "style": "Мозаика",
                                        "grid": "false", "trails": True, "speed": -5})
        self.assertEqual(actual.palette, "Графит")
        self.assertEqual(actual.style, "Мозаика")
        self.assertTrue(actual.grid)
        self.assertEqual(actual.speed, 8)


if __name__ == "__main__":
    unittest.main()
