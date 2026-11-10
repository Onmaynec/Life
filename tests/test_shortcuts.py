"""Keyboard commands never hijack editable widgets."""
import tkinter as tk
from tkinter import ttk
import unittest

from life_lab.app import is_text_input, LifeLabApp


class ShortcutTests(unittest.TestCase):
    def test_editable_widget_classes_are_recognized_without_display(self):
        for kind in (tk.Entry, tk.Text, tk.Spinbox, ttk.Entry, ttk.Combobox, ttk.Spinbox):
            self.assertTrue(is_text_input(object.__new__(kind)))
        self.assertFalse(is_text_input(object.__new__(tk.Button)))
        self.assertFalse(is_text_input(None))

    def test_shortcut_guard_ignores_text_fields(self):
        class FakeRoot:
            def __init__(self, focused):
                self.focused = focused
            def focus_get(self):
                return self.focused
        class FakeEvent:
            def __init__(self, widget):
                self.widget = widget
        class FakeWidget:
            def __init__(self, top):
                self.top = top
            def winfo_toplevel(self):
                return self.top
        app = object.__new__(LifeLabApp)
        root = FakeRoot(object.__new__(tk.Entry))
        app.root = root
        action_count = []
        self.assertIsNone(app._game_shortcut(FakeEvent(FakeWidget(root)), lambda: action_count.append(1)))
        self.assertEqual(action_count, [])
        root.focused = None
        self.assertEqual(app._game_shortcut(FakeEvent(FakeWidget(root)), lambda: action_count.append(1)), "break")
        self.assertEqual(action_count, [1])
        self.assertIsNone(app._game_shortcut(FakeEvent(FakeWidget(object())), lambda: action_count.append(2)))
        self.assertEqual(action_count, [1])


if __name__ == "__main__":
    unittest.main()
