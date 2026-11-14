"""Tkinter desktop UI; NumPy runs the simulation, Pillow draws the viewport."""
from __future__ import annotations

from collections import deque
import copy
import math
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from . import __version__
from .patterns import PATTERNS, decode_rle, encode_rle, pattern_array, place_pattern
from .storage import load_document, load_research, save_world
from .history import ExperimentHistory
from .instruments import ResearchLog
from .chrome import dark_titlebar
from .skins import SKINS, skin
from .research_ui import ResearchWorkspace
from .experiment_ui import ExperimentStudio
from .setup import WorldOptions, create_world
from .visuals import PALETTES, STYLES, HEATMAP_FIELDS, colorize, draw_view, minimap, heatmap_view
from .preferences import Preferences, load_preferences, save_preferences
from .world import CellularAutomaton, Ecosystem, RULE_PRESETS


BG = "#0B0D10"
PANEL = "#14171C"
PANEL_2 = "#1D2229"
EDGE = "#303640"
TEXT = "#F0F2F5"
MUTED = "#949DAA"
GREEN = "#54D9E6"
YELLOW = "#E3B56C"
PINK = "#EB728B"
CYAN = "#9DAEFF"
FONT = "Segoe UI"


def rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[i:i+2], 16) for i in (1, 3, 5))


def is_text_input(widget: Any) -> bool:
    """True when an editable widget should receive a key instead of a game command."""
    return isinstance(widget, (tk.Entry, tk.Text, tk.Spinbox,
                               ttk.Entry, ttk.Combobox, ttk.Spinbox))


class LifeLabApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        preferences = load_preferences()
        self.skin_var = tk.StringVar(value=preferences.skin)
        self.animation_var = tk.BooleanVar(value=preferences.animation)
        self._set_colors(preferences.skin)
        self.root.title(f"Life  —  v{__version__}")
        self.root.geometry("1570x950")
        self.root.minsize(1080, 710)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(80, lambda: dark_titlebar(self.root, BG))
        self.suite_window: ExperimentStudio | None = None

        self.world: CellularAutomaton | Ecosystem = CellularAutomaton()
        self.world.randomize(0.22, seed=42)
        self.running = False
        self.speed_var = tk.IntVar(value=preferences.speed)
        self.zoom = 9
        self.cam_x = 55.0
        self.cam_y = 42.0
        self.grid_var = tk.BooleanVar(value=preferences.grid)
        self.wrap_var = tk.BooleanVar(value=False)
        self.trail_var = tk.BooleanVar(value=preferences.trails)
        self.palette_var = tk.StringVar(value=preferences.palette)
        self.cell_style_var = tk.StringVar(value=preferences.style)
        self.layer_var = tk.StringVar(value="Организмы")
        self.step_ms = 0.0
        self.minimap_image: ImageTk.PhotoImage | None = None
        self._last_inspected: tuple[int, int] | None = None
        self.density_var = tk.DoubleVar(value=0.22)
        self.density_label_var = tk.StringVar(value="22 %")
        self.density_var.trace_add("write", lambda *_: self.density_label_var.set(f"{self.density_var.get()*100:.0f} %"))
        self.tool = "draw"
        self.species = 1
        self.pattern_name = tk.StringVar(value="Глайдер")
        self.rule_name = tk.StringVar(value="Жизнь Конвея")
        self.rule_entry_var = tk.StringVar(value="B3/S23")
        self.last_t = time.perf_counter()
        self.accumulated = 0.0
        self.dirty = True
        self.dragging = False
        self.panning = False
        self.pan_anchor: tuple[int, int] | None = None
        self.pencil_value = 1
        self.last_paint: tuple[int, int] | None = None
        self.pointer: tuple[int, int] | None = None
        self.experiment = ExperimentHistory.start(self.world)
        self.research_log = ResearchLog()
        self.research_log.capture(self.world)
        self.chart_mode_var = tk.StringVar(value="Численность")
        self.trails = np.zeros_like(self.world.cells, dtype=np.uint8)
        self.undo: list[tuple[Any, ...]] = []
        self.redo: list[tuple[Any, ...]] = []
        self.current_image: ImageTk.PhotoImage | None = None
        self.fps_samples: deque[float] = deque(maxlen=25)

        self._configure_styles()
        self._build_ui()
        self._bind_events()
        self._update_mode_ui()
        self._update_tool_ui()
        self._update_stats()
        self._show_hint("Пробел — запуск  ·  ЛКМ — редактирование  ·  ПКМ / СКМ — перемещение  ·  Колесо — масштаб")
        self._pulse_phase = 0
        self.root.after(220, self._pulse_status)
        self.root.after(35, self._tick)

    def run(self) -> None:
        self.root.mainloop()

    def _on_close(self) -> None:
        try:
            save_preferences(Preferences(palette=self.palette_var.get(),
                                         style=self.cell_style_var.get(),
                                         grid=self.grid_var.get(),
                                         trails=self.trail_var.get(),
                                         speed=self.speed_var.get(),
                                         skin=self.skin_var.get(),
                                         animation=self.animation_var.get()))
        except OSError:
            pass  # Lack of write access must not prevent the application from closing.
        if self.suite_window and self.suite_window.window.winfo_exists():
            self.suite_window.close()
        self.root.destroy()

    def _set_colors(self, name: str) -> None:
        """Apply a neutral dark surface and an accessible colour accent."""
        global BG, PANEL, PANEL_2, EDGE, TEXT, MUTED, GREEN, CYAN, YELLOW, PINK
        selected = skin(name)
        BG, PANEL, PANEL_2, EDGE = selected.base, selected.panel, selected.surface, selected.border
        TEXT, MUTED = selected.text, selected.muted
        GREEN, CYAN, YELLOW, PINK = selected.accent, selected.secondary, selected.warning, selected.danger

    def change_skin(self, _event: Any = None) -> None:
        """Recreate Tk surfaces so every control picks up the same neutral theme."""
        self._set_colors(self.skin_var.get())
        self.root.configure(bg=BG)
        for widget in self.root.winfo_children():
            if not isinstance(widget, tk.Toplevel):
                widget.destroy()
        self._configure_styles()
        self._build_ui()
        self._bind_events()
        self._update_mode_ui()
        self._update_tool_ui()
        self._update_stats()
        self._mark_dirty()
        self.root.after(80, lambda: dark_titlebar(self.root, BG))
        self.suite_window: ExperimentStudio | None = None
        self._show_hint("Оформление переключено — состояние мира не изменилось")

    def _pulse_status(self) -> None:
        """Low-cost heartbeat for the run indicator; disabled when animations are off."""
        if self.root.winfo_exists():
            if hasattr(self,"live_indicator"):
                if self.running:
                    self._pulse_phase = (self._pulse_phase + 1) % 6
                    color = GREEN if self._pulse_phase < 3 or not self.animation_var.get() else MUTED
                    self.live_indicator.config(text="● ИДЁТ", fg=color)
                else:
                    self.live_indicator.config(text="● ПАУЗА",fg=MUTED)
            self.root.after(220,self._pulse_status)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Lab.TCombobox", foreground=TEXT, fieldbackground=PANEL_2,
                        background=PANEL_2, bordercolor=EDGE, arrowcolor=GREEN,
                        selectbackground=PANEL_2, selectforeground=TEXT, padding=6)
        style.map("Lab.TCombobox", fieldbackground=[("readonly", PANEL_2)],
                  foreground=[("readonly", TEXT)], selectforeground=[("readonly", TEXT)])
        style.configure("Lab.Horizontal.TScale", background=PANEL,
                        troughcolor=PANEL_2, bordercolor=PANEL)
        self.root.option_add("*TCombobox*Listbox.background", PANEL_2)
        self.root.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.root.option_add("*TCombobox*Listbox.selectBackground", EDGE)

    def _button(self, parent: tk.Misc, title: str, command: Any,
                accent: bool = False, width: int | None = None) -> tk.Button:
        b = tk.Button(parent, text=title, command=command, bg=GREEN if accent else PANEL_2,
                      fg=BG if accent else TEXT, activebackground=GREEN if accent else EDGE,
                      activeforeground=BG if accent else TEXT, relief="flat", borderwidth=0,
                      font=(FONT, 10, "bold"), cursor="hand2", pady=9, padx=12,
                      highlightthickness=0)
        if width is not None:
            b.config(width=width)
        # Hover is a non-blocking colour transition, no additional threads.
        b._hover_epoch = 0
        b.bind("<Enter>",lambda _e,button=b:self._button_hover(button,True),add="+")
        b.bind("<Leave>",lambda _e,button=b:self._button_hover(button,False),add="+")
        return b

    def _button_hover(self, button: tk.Button, entering: bool) -> None:
        if not button.winfo_exists():
            return
        button._hover_epoch += 1
        epoch=button._hover_epoch
        original=button.cget("bg")
        if entering:
            button._hover_base=original
        end = EDGE if entering else getattr(button,"_hover_base",PANEL_2)
        if original == end:
            return
        if not self.animation_var.get():
            button.config(bg=end)
            return
        first, last = rgb(original), rgb(end)
        def tick(step:int)->None:
            if not button.winfo_exists() or button._hover_epoch != epoch:
                return
            t=step/4
            color="#{:02x}{:02x}{:02x}".format(*(round(a*(1-t)+b*t) for a,b in zip(first,last)))
            button.config(bg=color)
            if step<4:
                button.after(18,lambda:tick(step+1))
        tick(1)

    def _label(self, parent: tk.Misc, text: str, size: int = 10,
               color: str = MUTED, bold: bool = False) -> tk.Label:
        return tk.Label(parent, text=text, fg=color, bg=parent.cget("bg"),
                        anchor="w", font=(FONT, size, "bold" if bold else "normal"))

    def _section(self, parent: tk.Misc, title: str) -> tk.Frame:
        f = tk.Frame(parent, bg=PANEL, pady=2)
        self._label(f, title.upper(), 9, MUTED, True).pack(anchor="w", pady=(8, 10))
        return f

    def _scroll_panel(self, parent: tk.Misc, width: int) -> tuple[tk.Frame, tk.Frame, tk.Canvas]:
        outer = tk.Frame(parent, bg=PANEL, width=width, highlightbackground=EDGE, highlightthickness=1)
        outer.pack_propagate(False)
        scroller = tk.Canvas(outer, bg=PANEL, highlightthickness=0, bd=0)
        scroller.pack(side="left", fill="both", expand=True)
        bar = tk.Scrollbar(outer, orient="vertical", command=scroller.yview,
                           troughcolor=PANEL, bg=PANEL_2, activebackground=EDGE, width=11)
        bar.pack(side="right", fill="y")
        scroller.configure(yscrollcommand=bar.set)
        inner = tk.Frame(scroller, bg=PANEL, padx=17, pady=14)
        window = scroller.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda _e: scroller.configure(scrollregion=scroller.bbox("all")))
        scroller.bind("<Configure>", lambda e: scroller.itemconfigure(window, width=e.width))
        return outer, inner, scroller

    def _metric(self, parent: tk.Misc, title: str, text: str, accent: str) -> tk.Label:
        card = tk.Frame(parent, bg=PANEL, padx=16, pady=9)
        card.pack(side="left", fill="x", expand=True, padx=(0, 8))
        tk.Frame(card, bg=accent, width=3).pack(side="left", fill="y", padx=(0, 11))
        body = tk.Frame(card, bg=PANEL)
        body.pack(side="left", fill="x", expand=True)
        self._label(body, title.upper(), 8, MUTED, True).pack(anchor="w")
        value = self._label(body, text, 18, TEXT, True)
        value.pack(anchor="w", pady=(3, 0))
        return value

    def _build_ui(self) -> None:
        # Workspace header: actions stay visible while the left and right panels scroll.
        header = tk.Frame(self.root, bg=BG, padx=18, pady=11)
        header.pack(fill="x")
        brand = tk.Frame(header, bg=BG)
        brand.pack(side="left")
        tk.Label(brand, text="◉", fg=GREEN, bg=BG, font=(FONT, 27, "bold")).pack(side="left", padx=(0, 10))
        wordmark = tk.Frame(brand, bg=BG)
        wordmark.pack(side="left")
        self._label(wordmark, "life.", 23, TEXT, True).pack(anchor="w")
        self._label(wordmark, "RESEARCH STUDIO  /  ЭКСПЕРИМЕНТАЛЬНАЯ СРЕДА", 8, MUTED, True).pack(anchor="w")
        actions = tk.Frame(header, bg=BG)
        actions.pack(side="right")
        self.live_indicator=self._label(actions, "● ПАУЗА", 9, MUTED, True)
        self.live_indicator.pack(side="left",padx=(0,15))
        self._button(actions, "?  СПРАВКА", self.show_help).pack(side="left", padx=(0, 7))
        self._button(actions, "+  НОВЫЙ МИР", self.new_world_dialog).pack(side="left", padx=(0, 8))
        self._button(actions, "СОХРАНИТЬ", self.save).pack(side="left", padx=(0, 16))
        self._label(actions, "ШАГОВ / СЕК", 9, MUTED, True).pack(side="left", padx=(0, 7))
        self._button(actions, "−", lambda: self._adjust_speed(-1), width=3).pack(side="left")
        tk.Label(actions, textvariable=self.speed_var, bg=PANEL_2, fg=TEXT,
                 width=3, font=(FONT, 11, "bold"), pady=9).pack(side="left", padx=3)
        self._button(actions, "+", lambda: self._adjust_speed(1), width=3).pack(side="left", padx=(0, 16))
        self._button(actions, "ОДИН ШАГ", self.step_once).pack(side="left", padx=(0, 8))
        self.run_button = self._button(actions, "▶  ЗАПУСТИТЬ", self.toggle_run, accent=True)
        self.run_button.pack(side="left")

        divider = tk.Frame(self.root, height=1, bg=EDGE)
        divider.pack(fill="x")
        workspace = tk.Frame(self.root, bg=BG, padx=11, pady=11)
        workspace.pack(fill="both", expand=True)

        # Left: experiment configuration and tools.
        left_shell, self.side, self.side_scroll = self._scroll_panel(workspace, 260)
        self.left_shell=left_shell
        left_shell.pack(side="left", fill="y", padx=(0, 12))
        self._label(self.side, "НАСТРОЙКА", 10, TEXT, True).pack(anchor="w", pady=(0, 4))
        self._label(self.side, "Режим, инструменты и правила", 9, MUTED).pack(anchor="w", pady=(0, 11))
        tk.Frame(self.side, height=1, bg=EDGE).pack(fill="x", pady=(0, 8))

        mode = self._section(self.side, "Симуляция")
        mode.pack(fill="x", pady=(0, 10))
        mode_row = tk.Frame(mode, bg=PANEL)
        mode_row.pack(fill="x")
        self.classic_button = self._button(mode_row, "КОНВЕЙ", lambda: self.change_mode("classic"))
        self.classic_button.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.eco_button = self._button(mode_row, "ЭКОСИСТЕМА", lambda: self.change_mode("ecosystem"))
        self.eco_button.pack(side="left", fill="x", expand=True)

        tools = self._section(self.side, "Рисование")
        tools.pack(fill="x", pady=(0, 10))
        toolbox = tk.Frame(tools, bg=PANEL)
        toolbox.pack(fill="x")
        self.draw_button = self._button(toolbox, "●  КИСТЬ", lambda: self.set_tool("draw"))
        self.draw_button.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.erase_button = self._button(toolbox, "⌫  ЛАСТИК", lambda: self.set_tool("erase"))
        self.erase_button.pack(side="left", fill="x", expand=True)
        self.pattern_controls = tk.Frame(tools, bg=PANEL, pady=6)
        self._label(self.pattern_controls, "Готовые структуры", 9).pack(anchor="w", pady=(3, 5))
        self.pattern_combo = ttk.Combobox(self.pattern_controls, textvariable=self.pattern_name,
                                           values=list(PATTERNS), state="readonly", style="Lab.TCombobox")
        self.pattern_combo.pack(fill="x")
        self.pattern_combo.bind("<<ComboboxSelected>>", lambda _e: self.set_tool("stamp"))
        self.stamp_button = self._button(self.pattern_controls, "ПОСТАВИТЬ ФИГУРУ", lambda: self.set_tool("stamp"))
        self.stamp_button.pack(fill="x", pady=(6, 0))
        self.species_frame = tk.Frame(tools, bg=PANEL, pady=6)
        self._label(self.species_frame, "Тип организма", 9).pack(anchor="w", pady=(3, 6))
        self.species_buttons: dict[int, tk.Button] = {}
        for kind, title in ((1, "Растения"), (2, "Травоядные"), (3, "Хищники")):
            b = self._button(self.species_frame, "●  " + title, lambda k=kind: self.set_species(k))
            b.pack(fill="x", pady=2)
            self.species_buttons[kind] = b

        self.rules_frame = self._section(self.side, "Законы мира")
        self.rules_frame.pack(fill="x", pady=(0, 10))
        self.rule_combo = ttk.Combobox(self.rules_frame, textvariable=self.rule_name,
                                        values=list(RULE_PRESETS), state="readonly", style="Lab.TCombobox")
        self.rule_combo.pack(fill="x")
        self.rule_combo.bind("<<ComboboxSelected>>", self.apply_preset)
        rule_row = tk.Frame(self.rules_frame, bg=PANEL, pady=6)
        rule_row.pack(fill="x")
        self.rule_entry = tk.Entry(rule_row, textvariable=self.rule_entry_var, bg=PANEL_2,
                                   fg=TEXT, insertbackground=TEXT, relief="flat", bd=0,
                                   font=("Consolas", 12), highlightbackground=EDGE, highlightthickness=1)
        self.rule_entry.pack(side="left", fill="x", expand=True, ipady=9)
        self._button(rule_row, "OK", self.apply_rule).pack(side="right", padx=(6, 0))
        self.rule_entry.bind("<Return>", lambda _e: self.apply_rule())

        actions_panel = self._section(self.side, "Редактор мира")
        self.actions_frame = actions_panel
        actions_panel.pack(fill="x", pady=(0, 10))
        self._button(actions_panel, "+  СОЗДАТЬ МИР", self.new_world_dialog, accent=True).pack(fill="x", pady=(0, 7))
        tool_row = tk.Frame(actions_panel, bg=PANEL)
        tool_row.pack(fill="x")
        self._button(tool_row, "СЛУЧАЙНЫЙ", self.randomize).pack(side="left", fill="x", expand=True, padx=(0, 4))
        self._button(tool_row, "ОЧИСТИТЬ", self.clear).pack(side="left", fill="x", expand=True)
        undo_row = tk.Frame(actions_panel, bg=PANEL, pady=7)
        undo_row.pack(fill="x")
        self._button(undo_row, "↶  НАЗАД", self.undo_action).pack(side="left", fill="x", expand=True, padx=(0, 4))
        self._button(undo_row, "↷  ВПЕРЁД", self.redo_action).pack(side="left", fill="x", expand=True)
        self._label(actions_panel, "Случайное заполнение", 9).pack(anchor="w", pady=(7, 5))
        fill_stepper = tk.Frame(actions_panel, bg=PANEL)
        fill_stepper.pack(fill="x")
        self._button(fill_stepper, "−", lambda: self._adjust_density(-0.01), width=3).pack(side="left")
        tk.Label(fill_stepper, textvariable=self.density_label_var, bg=PANEL_2,
                 fg=TEXT, font=(FONT, 10, "bold"), pady=9).pack(side="left", fill="x", expand=True, padx=4)
        self._button(fill_stepper, "+", lambda: self._adjust_density(0.01), width=3).pack(side="right")
        for title, var, command in (("Замкнутые границы", self.wrap_var, self._toggle_wrap),
                                    ("Показать сетку", self.grid_var, self._mark_dirty),
                                    ("Следы поколений", self.trail_var, self._mark_dirty)):
            cb = tk.Checkbutton(actions_panel, text=title, variable=var, command=command,
                                bg=PANEL, fg=TEXT, selectcolor=PANEL_2,
                                activebackground=PANEL, activeforeground=TEXT,
                                highlightthickness=0, anchor="w", font=(FONT, 10))
            cb.pack(fill="x", pady=2)

        files = self._section(self.side, "Файлы")
        files.pack(fill="x", pady=(0, 12))
        f1 = tk.Frame(files, bg=PANEL)
        f1.pack(fill="x")
        self._button(f1, "СОХРАНИТЬ", self.save).pack(side="left", fill="x", expand=True, padx=(0, 4))
        self._button(f1, "ОТКРЫТЬ", self.load).pack(side="left", fill="x", expand=True)
        self.rle_row = tk.Frame(files, bg=PANEL, pady=7)
        self.rle_row.pack(fill="x")
        self._button(self.rle_row, "ИМПОРТ RLE", self.import_rle).pack(side="left", fill="x", expand=True, padx=(0, 4))
        self._button(self.rle_row, "ЭКСПОРТ RLE", self.export_rle).pack(side="left", fill="x", expand=True)
        self.png_button = self._button(files, "ЭКСПОРТ PNG", self.export_png)
        self.png_button.pack(fill="x")
        self._button(files, "ИСТОРИЯ → CSV", self.export_csv).pack(fill="x", pady=(7, 0))

        # Right: live data, navigator and appearance settings.
        right_shell, self.inspector_side, self.inspector_scroll = self._scroll_panel(workspace, 294)
        self.right_shell=right_shell
        right_shell.pack(side="right", fill="y", padx=(12, 0))
        self.research_ui = ResearchWorkspace(self, self.inspector_side)
        self._label(self.research_ui.overview, "МОНИТОРИНГ", 11, TEXT, True).pack(anchor="w", pady=(0, 5))
        self._label(self.research_ui.overview, "Показатели состояния мира", 9, MUTED).pack(anchor="w", pady=(0, 10))

        stats = self._section(self.research_ui.overview, "Статистика")
        stats.pack(fill="x", pady=(0, 10))
        self.stats_labels: dict[str, tk.Label] = {}
        for key, title in (("generation", "Поколение"), ("population", "Живые клетки"),
                           ("births", "Рождения"), ("deaths", "Смерти"),
                           ("plants", "Растения"), ("herbivores", "Травоядные"),
                           ("predators", "Хищники"), ("mutations", "Мутации"),
                           ("metabolism", "Метаболизм, среднее"),
                           ("sight", "Зрение, среднее"), ("fertility", "Плодовитость, среднее")):
            row = tk.Frame(stats, bg=PANEL)
            row.pack(fill="x", pady=2)
            self._label(row, title, 9).pack(side="left")
            label = self._label(row, "0", 10, TEXT, True)
            label.pack(side="right")
            self.stats_labels[key] = label
            if key in ("plants", "herbivores", "predators", "mutations", "metabolism", "sight", "fertility"):
                row.pack_forget()
                setattr(self, f"_statrow_{key}", row)

        graph_section = self._section(self.research_ui.overview, "Динамика")
        graph_section.pack(fill="x", pady=(0, 10))
        self.chart_mode = ttk.Combobox(graph_section, textvariable=self.chart_mode_var,
                                       values=["Численность", "Метаболизм", "Зрение", "Плодовитость"],
                                       state="readonly", style="Lab.TCombobox")
        self.chart_mode.pack(fill="x", pady=(0, 7))
        self.chart_mode.bind("<<ComboboxSelected>>", lambda _e: self._draw_chart())
        self.chart = tk.Canvas(graph_section, height=164, bg="#0D1A28", highlightthickness=0)
        self.chart.pack(fill="x")
        self.chart.bind("<Configure>", lambda _e: self._draw_chart())
        self._label(graph_section, "120 последних записей  ·  по поколениям", 8).pack(anchor="w", pady=(6, 0))

        navigator = self._section(self.research_ui.overview, "Навигатор")
        navigator.pack(fill="x", pady=(0, 10))
        self.minimap_canvas = tk.Canvas(navigator, bg="#0C1927", height=130,
                                        highlightthickness=1, highlightbackground=EDGE, cursor="hand2")
        self.minimap_canvas.pack(fill="x")
        self.minimap_canvas.bind("<Button-1>", self._minimap_click)
        self.minimap_canvas.bind("<Configure>", self._mark_dirty)
        self._label(navigator, "Нажмите на карту, чтобы переместить камеру", 8).pack(anchor="w", pady=(6, 0))
        inspect = tk.Frame(navigator, bg=PANEL_2, padx=10, pady=9)
        inspect.pack(fill="x", pady=(9, 0))
        self.cell_inspector = tk.Label(inspect, text="Наведите курсор на клетку", bg=PANEL_2,
                                        fg=TEXT, anchor="w", justify="left", wraplength=235,
                                        font=(FONT, 9))
        self.cell_inspector.pack(fill="x")

        visual = self._section(self.research_ui.overview, "Оформление мира")
        visual.pack(fill="x", pady=(0, 12))
        self._label(visual, "Цветовая палитра", 9).pack(anchor="w", pady=(0, 5))
        self.palette_combo = ttk.Combobox(visual, textvariable=self.palette_var,
                                          values=list(PALETTES), state="readonly", style="Lab.TCombobox")
        self.palette_combo.pack(fill="x")
        self.palette_combo.bind("<<ComboboxSelected>>", self._appearance_changed)
        self._label(visual, "Вид клеток", 9).pack(anchor="w", pady=(11, 5))
        self.style_combo = ttk.Combobox(visual, textvariable=self.cell_style_var,
                                        values=STYLES, state="readonly", style="Lab.TCombobox")
        self.style_combo.pack(fill="x")
        self.style_combo.bind("<<ComboboxSelected>>", self._appearance_changed)
        legend = tk.Frame(visual, bg=PANEL, pady=10)
        legend.pack(fill="x")
        self.legend_labels: dict[int, tk.Label] = {}
        for kind, title in ((1, "РАСТЕНИЯ"), (2, "ТРАВОЯДНЫЕ"), (3, "ХИЩНИКИ")):
            label = self._label(legend, f"● {title}", 8, GREEN if kind == 1 else YELLOW if kind == 2 else PINK, True)
            label.pack(anchor="w", pady=2)
            self.legend_labels[kind] = label

        # Central stage stays as wide as possible.
        stage = tk.Frame(workspace, bg=PANEL, highlightbackground=EDGE, highlightthickness=1)
        self.stage=stage
        self.focus_mode=False
        stage.pack(side="left", fill="both", expand=True)
        toolbar = tk.Frame(stage, bg=PANEL, padx=17, pady=12)
        toolbar.pack(fill="x")
        self._label(toolbar, "ПОЛЕ СИМУЛЯЦИИ", 10, TEXT, True).pack(side="left")
        self.world_mode_label = self._label(toolbar, "КОНВЕЙ  ·  B3/S23", 9, GREEN, True)
        self.world_mode_label.pack(side="right")
        quick_appearance = tk.Frame(stage, bg=PANEL, padx=17, pady=5)
        quick_appearance.pack(fill="x")
        self._label(quick_appearance, "ТЕМА", 8).pack(side="left", padx=(0, 6))
        chosen_skin = ttk.Combobox(quick_appearance, textvariable=self.skin_var,
                          values=list(SKINS), state="readonly",style="Lab.TCombobox",width=20)
        chosen_skin.pack(side="left",padx=(0,14))
        chosen_skin.bind("<<ComboboxSelected>>",self.change_skin)
        self._label(quick_appearance, "ПАЛИТРА", 8).pack(side="left", padx=(0, 7))
        preset = ttk.Combobox(quick_appearance, textvariable=self.palette_var,
                              values=list(PALETTES), state="readonly", style="Lab.TCombobox", width=13)
        preset.pack(side="left", padx=(0, 12))
        preset.bind("<<ComboboxSelected>>", self._appearance_changed)
        self._label(quick_appearance, "ВИД", 8).pack(side="left", padx=(0, 7))
        kind = ttk.Combobox(quick_appearance, textvariable=self.cell_style_var,
                            values=STYLES, state="readonly", style="Lab.TCombobox", width=11)
        kind.pack(side="left")
        kind.bind("<<ComboboxSelected>>", self._appearance_changed)
        self._label(quick_appearance,"СЛОЙ",8).pack(side="left",padx=(11,5))
        layer=ttk.Combobox(quick_appearance,textvariable=self.layer_var,
                    values=HEATMAP_FIELDS,state="readonly",style="Lab.TCombobox",width=13)
        layer.pack(side="left")
        layer.bind("<<ComboboxSelected>>",self._appearance_changed)

        metrics = tk.Frame(stage, bg=PANEL, padx=12, pady=7)
        metrics.pack(fill="x")
        self.hud_generation = self._metric(metrics, "Поколение", "0", CYAN)
        self.hud_population = self._metric(metrics, "Живых", "0", GREEN)
        self.hud_density = self._metric(metrics, "Заполнение", "0%", YELLOW)
        self.canvas = tk.Canvas(stage, bg="#0C1826", bd=0, highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=12, pady=(5, 0))
        foot = tk.Frame(stage, bg=PANEL, padx=14, pady=10)
        foot.pack(fill="x")
        self.coord_label = self._label(foot, "X: —    Y: —", 9, TEXT, True)
        self.coord_label.pack(side="left")
        self.zoom_label = self._label(foot, "100%", 9, MUTED)
        self.zoom_label.pack(side="right", padx=(10, 0))
        self._button(foot, "−", lambda: self._zoom_center(-1), width=3).pack(side="right", padx=(4, 0))
        self._button(foot, "+", lambda: self._zoom_center(1), width=3).pack(side="right", padx=(4, 0))
        self._button(foot, "ВПИСАТЬ", self.fit_world).pack(side="right", padx=(0, 5))
        self._label(foot, "ПКМ / СКМ — перемещение · F10 — фокус", 8).pack(side="right", padx=(8, 12))

        status = tk.Frame(self.root, bg=BG, padx=25, pady=10)
        status.pack(fill="x")
        self.notice_label = self._label(status, "", 9, MUTED)
        self.notice_label.pack(side="left")
        self.performance_label = self._label(status, "", 9, GREEN)
        self.performance_label.pack(side="right")
        self.root.bind_all("<MouseWheel>", self._wheel_panels)

    def _bind_events(self) -> None:
        self.root.bind_all("<space>", self._key_space)
        self.root.bind_all("<F1>", lambda _e: self.show_help())
        self.root.bind_all("<F10>", lambda _e: self.toggle_focus())
        self.root.bind_all("<F11>",lambda _e:self.toggle_fullscreen())
        self.root.bind_all("<Control-Tab>", lambda _e:self.next_research_tab())
        # File operations are deliberately allowed when a text field has focus.
        self.root.bind_all("<Control-s>", lambda _e: self.save())
        self.root.bind_all("<Control-o>", lambda _e: self.load())
        self.root.bind("<Control-e>", self._key_experiment)
        self.root.bind("<Control-l>", lambda _e: self.research_ui.show("Родословная"))
        shortcuts = {
            "<Control-z>": self.undo_action,
            "<Control-y>": self.redo_action,
            "<KeyPress-n>": self.step_once,
            "<KeyPress-r>": self.randomize,
            "<KeyPress-c>": self.clear,
            "<KeyPress-g>": self._toggle_grid,
            "<KeyPress-f>": self.fit_world,
            "<KeyPress-1>": lambda: self.set_species(1),
            "<KeyPress-2>": lambda: self.set_species(2),
            "<KeyPress-3>": lambda: self.set_species(3),
            "<KeyPress-Escape>": lambda: self.set_tool("draw"),
        }
        for sequence, action in shortcuts.items():
            self.root.bind(sequence, lambda e, callback=action: self._game_shortcut(e, callback))
        self.canvas.bind("<ButtonPress-1>", self._mouse_down)
        self.canvas.bind("<B1-Motion>", self._mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._mouse_up)
        self.canvas.bind("<ButtonPress-3>", self._right_down)
        self.canvas.bind("<B3-Motion>", self._pan_drag)
        self.canvas.bind("<ButtonRelease-3>", self._pan_up)
        self.canvas.bind("<ButtonPress-2>", self._pan_down)
        self.canvas.bind("<B2-Motion>", self._pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._pan_up)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, 1))
        self.canvas.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, -1))
        self.canvas.bind("<Motion>", self._mouse_motion)
        self.canvas.bind("<Leave>", self._mouse_leave)
        self.canvas.bind("<Configure>", self._mark_dirty)

    def _game_shortcut(self, event: tk.Event[Any], callback: Any) -> str | None:
        widget = self.root.focus_get()
        if is_text_input(widget) or event.widget.winfo_toplevel() is not self.root:
            return None
        callback()
        return "break"

    def open_experiment_studio(self) -> None:
        if self.suite_window and self.suite_window.window.winfo_exists():
            self.suite_window.window.lift()
            self.suite_window.window.focus_set()
            return
        self.suite_window = ExperimentStudio(self)

    def _key_experiment(self, event: tk.Event[Any]) -> str | None:
        if event.widget.winfo_toplevel() is not self.root:
            return None
        self.open_experiment_studio()
        return "break"

    def toggle_focus(self) -> None:
        self.focus_mode = not self.focus_mode
        if self.focus_mode:
            self.left_shell.pack_forget()
            self.right_shell.pack_forget()
        else:
            self.left_shell.pack(side="left",fill="y",padx=(0,12),before=self.stage)
            self.right_shell.pack(side="right",fill="y",padx=(12,0),before=self.stage)
        self._mark_dirty()

    def toggle_fullscreen(self) -> None:
        self.root.attributes("-fullscreen", not bool(self.root.attributes("-fullscreen")))
        self._mark_dirty()

    def next_research_tab(self) -> None:
        names=list(self.research_ui.pages)
        self.research_ui.show(names[(names.index(self.research_ui.current)+1)%len(names)])

    def _key_space(self, event: tk.Event[Any]) -> str | None:
        return self._game_shortcut(event, self.toggle_run)

    def _wheel_panels(self, event: tk.Event[Any]) -> str | None:
        """Scroll whichever settings panel lies under the cursor."""
        if not event.delta:
            return None
        x = event.x_root
        for panel in (self.side_scroll, self.inspector_scroll):
            if panel.winfo_rootx() <= x < panel.winfo_rootx() + panel.winfo_width():
                panel.yview_scroll(-int(event.delta / 120) or (-1 if event.delta > 0 else 1), "units")
                return "break"
        return None

    def _appearance_changed(self, _event: Any = None) -> None:
        """Appearance is local UI state; saves retain the actual world only."""
        palette = PALETTES[self.palette_var.get()]
        for kind, label in self.legend_labels.items():
            label.config(fg="#%02x%02x%02x" % palette.species[kind-1])
        self._mark_dirty()

    def _adjust_speed(self, step: int) -> None:
        self.speed_var.set(max(1, min(30, self.speed_var.get() + step)))

    def _adjust_density(self, step: float) -> None:
        self.density_var.set(round(max(0.02, min(0.65, self.density_var.get() + step)), 2))

    def _zoom_center(self, direction: int) -> None:
        self._zoom_at(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, direction)

    def _minimap_click(self, event: tk.Event[Any]) -> None:
        w, h = self.minimap_canvas.winfo_width(), self.minimap_canvas.winfo_height()
        if w <= 1 or h <= 1:
            return
        x = max(0.0, min(1.0, event.x / w)) * self.world.width
        y = max(0.0, min(1.0, event.y / h)) * self.world.height
        self.cam_x = self.canvas.winfo_width() / 2 - x * self.zoom
        self.cam_y = self.canvas.winfo_height() / 2 - y * self.zoom
        self._mark_dirty()

    def _update_cell_inspector(self, point: tuple[int, int] | None) -> None:
        if point == self._last_inspected:
            return
        self._last_inspected = point
        if point is None:
            self.cell_inspector.config(text="Наведите курсор на клетку", fg=TEXT)
            return
        x, y = point
        if not (0 <= x < self.world.width and 0 <= y < self.world.height):
            self.cell_inspector.config(text="За пределами мира", fg=MUTED)
            return
        kind = int(self.world.cells[y, x])
        if isinstance(self.world, Ecosystem):
            title = ("Пустая клетка", "Растение", "Травоядное", "Хищник")[kind]
            desc = f"{title}  ·  {x}:{y}\nПочва: {self.world.nutrients[y,x]:.1f}"
            if kind >= 2:
                desc += (f"  ·  Энергия: {self.world.energy[y,x]:.1f}"
                         f"\nМетаболизм: {self.world.trait[y,x]:.2f}"
                         f"  ·  Зрение: {self.world.sight[y,x]:.2f}"
                         f"\nПлодовитость: {self.world.fertility[y,x]:.2f}"
                         f"  ·  Возраст: {int(self.world.age[y,x])}")
                individual = self.world.lineage.get(y, x)
                if individual is not None:
                    desc += (f"\nID: {individual.id}  ·  Линия: {individual.founder}"
                             f"  ·  Колено: {individual.depth}")
        else:
            desc = f"{'Живая клетка' if kind else 'Пустая клетка'}  ·  {x}:{y}\nПравило {self.world.rule}"
        self.cell_inspector.config(text=desc, fg=TEXT)

    def _draw_minimap(self) -> None:
        nav = self.minimap_canvas
        w, h = nav.winfo_width(), nav.winfo_height()
        if w < 2 or h < 2:
            return
        image = minimap(self.world.cells, palette_name=self.palette_var.get(), width=w, height=h,
                        mode=self.world.mode)
        self.minimap_image = ImageTk.PhotoImage(image)
        nav.delete("all")
        nav.create_image(0, 0, image=self.minimap_image, anchor="nw")
        x0 = (-self.cam_x / self.zoom) / self.world.width * w
        y0 = (-self.cam_y / self.zoom) / self.world.height * h
        x1 = ((self.canvas.winfo_width()-self.cam_x) / self.zoom) / self.world.width * w
        y1 = ((self.canvas.winfo_height()-self.cam_y) / self.zoom) / self.world.height * h
        x0, x1 = max(0, x0), min(w, x1)
        y0, y1 = max(0, y0), min(h, y1)
        if x1 > x0 and y1 > y0:
            nav.create_rectangle(x0, y0, x1, y1, outline=GREEN, width=2)

    def _scroll_sidebar(self, e: tk.Event[Any]) -> str:
        self.side_scroll.yview_scroll(-int(e.delta / 120), "units")
        return "break"

    def _show_hint(self, msg: str) -> None:
        self.notice_label.config(text=msg)

    def _mark_dirty(self, *_: Any) -> None:
        self.dirty = True

    def _toggle_grid(self) -> None:
        self.grid_var.set(not self.grid_var.get())
        self._mark_dirty()

    def _toggle_wrap(self) -> None:
        self.world.wrap = self.wrap_var.get()
        self._mark_dirty()

    def set_tool(self, tool: str) -> None:
        if tool == "stamp" and self.world.mode == "ecosystem":
            self._show_hint("Шаблоны RLE доступны только в классическом режиме")
            return
        self.tool = tool
        self._update_tool_ui()
        self._mark_dirty()

    def _update_tool_ui(self) -> None:
        for b, chosen in ((self.draw_button, "draw"), (self.erase_button, "erase"),
                          (self.stamp_button, "stamp")):
            active = self.tool == chosen
            b.config(bg=GREEN if active else PANEL_2, fg=BG if active else TEXT)
        for kind, b in self.species_buttons.items():
            b.config(bg=EDGE if self.species == kind else PANEL_2)

    def set_species(self, kind: int) -> None:
        if self.world.mode != "ecosystem":
            return
        self.species = kind
        self.tool = "draw"
        self._update_tool_ui()

    def _snapshot(self) -> tuple[Any, ...]:
        if isinstance(self.world, Ecosystem):
            return tuple(a.copy() for a in (self.world.cells, self.world.energy,
                                             self.world.trait, self.world.sight,
                                             self.world.fertility, self.world.nutrients,
                                             self.world.moisture, self.world.age,
                                             self.world.seeds, self.world.organic,
                                             self.world.toxins, self.world.altitude)) + (copy.deepcopy(self.world.lineage),)
        return (self.world.cells.copy(),)

    def _restore(self, snapshot: tuple[Any, ...]) -> None:
        if self.world.cells.shape != snapshot[0].shape:
            return
        if isinstance(self.world, Ecosystem):
            for arr, data in zip((self.world.cells, self.world.energy,
                                  self.world.trait, self.world.sight,
                                  self.world.fertility, self.world.nutrients,
                                  self.world.moisture, self.world.age,
                                  self.world.seeds, self.world.organic,
                                  self.world.toxins, self.world.altitude), snapshot[:12]):
                arr[:] = data
            self.world.lineage = copy.deepcopy(snapshot[12])
        else:
            self.world.cells[:] = snapshot[0]
        self._reset_visuals()

    def _push_undo(self) -> None:
        self.undo.append(self._snapshot())
        if len(self.undo) > 25:
            del self.undo[0]
        self.redo.clear()

    def undo_action(self) -> None:
        self.running = False
        self._update_run_button()
        if self.undo:
            self.redo.append(self._snapshot())
            self._restore(self.undo.pop())
            self._show_hint("Последнее редактирование отменено")

    def redo_action(self) -> None:
        self.running = False
        self._update_run_button()
        if self.redo:
            self.undo.append(self._snapshot())
            self._restore(self.redo.pop())
            self._show_hint("Редактирование повторено")

    def _reset_visuals(self, history: ExperimentHistory | None = None) -> None:
        self.trails = np.zeros_like(self.world.cells, dtype=np.uint8)
        self._last_inspected = None
        self.cell_inspector.config(text="Наведите курсор на клетку", fg=TEXT)
        self.experiment = history if history is not None else ExperimentHistory.start(self.world)
        self.research_log = ResearchLog()
        self.research_log.capture(self.world)
        self.research_ui.baseline = None
        self.research_ui.selected_id = None
        self.research_ui.sync_ecology()
        self.research_ui.refresh(force=True)
        self._update_stats()
        self._mark_dirty()

    def show_help(self) -> None:
        """Short in-app introduction to both simulation modes and the editor."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Справка — Life")
        dw, dh = 530, 560
        x = self.root.winfo_rootx() + max(0, (self.root.winfo_width()-dw)//2)
        y = self.root.winfo_rooty() + max(0, (self.root.winfo_height()-dh)//2)
        dialog.geometry(f"{dw}x{dh}+{x}+{y}")
        dialog.resizable(False, False)
        dialog.configure(bg=PANEL)
        dialog.after(80, lambda: dark_titlebar(dialog, BG))
        dialog.transient(self.root)
        frame = tk.Frame(dialog, bg=PANEL, padx=27, pady=23)
        frame.pack(fill="both", expand=True)
        self._label(frame, "КАК РАБОТАТЬ С LIFE", 17, TEXT, True).pack(anchor="w", pady=(0, 5))
        self._label(frame, "Создавайте миры и наблюдайте, как меняются их законы.", 10).pack(anchor="w", pady=(0, 18))
        for title, desc in (
            ("Классика Конвея", "Каждая клетка либо жива, либо нет. Её судьба зависит от восьми соседей. Попробуйте поставить глайдер и запустить симуляцию."),
            ("Экосистема", "Растения используют питательные вещества, травоядные ищут растения, а хищники охотятся. У животных есть наследуемые признаки."),
            ("Управление", "Пробел — пауза, N — шаг. ЛКМ редактирует, ластик выбирается отдельно. ПКМ или СКМ — перемещение, колесо — масштаб, F — вписать мир."),
            ("Наблюдение", "Справа: обзор, анализ, 3D-графики, климат и каталог структур. F10 — режим без боковых панелей, F11 — полный экран, Ctrl+Tab — вкладки."),
        ):
            self._label(frame, title, 11, GREEN, True).pack(anchor="w", pady=(0, 4))
            tk.Label(frame, text=desc, bg=PANEL, fg=TEXT, justify="left", anchor="w",
                     wraplength=468, font=(FONT, 10)).pack(anchor="w", pady=(0, 15))
        self._button(frame, "ПОНЯТНО", dialog.destroy, accent=True).pack(fill="x", pady=(4, 0))
        dialog.bind("<Escape>", lambda _e: dialog.destroy())
        dialog.focus_set()

    def new_world_dialog(self) -> None:
        """Configure and replace the current world after validating every field."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Новый мир — Life")
        dw, dh = 470, 650
        x = self.root.winfo_rootx() + max(0, (self.root.winfo_width()-dw)//2)
        y = self.root.winfo_rooty() + max(0, (self.root.winfo_height()-dh)//2)
        dialog.geometry(f"{dw}x{dh}+{x}+{y}")
        dialog.resizable(False, False)
        dialog.configure(bg=PANEL)
        dialog.after(80, lambda: dark_titlebar(dialog, BG))
        dialog.transient(self.root)
        dialog.grab_set()

        mode = tk.StringVar(value="Классика" if self.world.mode == "classic" else "Экосистема")
        width = tk.StringVar(value=str(self.world.width))
        height = tk.StringVar(value=str(self.world.height))
        seed = tk.StringVar(value="")
        wrap = tk.BooleanVar(value=self.world.wrap)
        rule = tk.StringVar(value=self.world.rule if isinstance(self.world, CellularAutomaton) else "B3/S23")
        density = tk.StringVar(value="22")
        plants = tk.StringVar(value="17")
        herbivores = tk.StringVar(value="1.8")
        predators = tk.StringVar(value="0.3")

        frame = tk.Frame(dialog, bg=PANEL, padx=23, pady=16)
        frame.pack(fill="both", expand=True)
        self._label(frame, "НОВЫЙ МИР", 17, TEXT, True).pack(anchor="w", pady=(0, 3))
        self._label(frame, "Задайте начальные условия эксперимента", 10, MUTED).pack(anchor="w", pady=(0, 12))
        tk.Frame(frame, height=1, bg=EDGE).pack(fill="x", pady=(0, 10))

        def field(label: str, variable: tk.StringVar, parent: tk.Misc = frame) -> tk.Entry:
            self._label(parent, label, 9).pack(anchor="w", pady=(8, 3))
            entry = tk.Entry(parent, textvariable=variable, bg=PANEL_2, fg=TEXT,
                             insertbackground=TEXT, relief="flat", font=(FONT, 11))
            entry.pack(fill="x", ipady=5)
            return entry

        self._label(frame, "Режим", 9).pack(anchor="w")
        mode_combo = ttk.Combobox(frame, textvariable=mode, values=("Классика", "Экосистема"),
                                  state="readonly", style="Lab.TCombobox")
        mode_combo.pack(fill="x")
        sizes = tk.Frame(frame, bg=PANEL)
        sizes.pack(fill="x")
        left = tk.Frame(sizes, bg=PANEL)
        right = tk.Frame(sizes, bg=PANEL)
        left.pack(side="left", fill="x", expand=True, padx=(0, 6))
        right.pack(side="left", fill="x", expand=True)
        field("Ширина (20–700)", width, left)
        field("Высота (20–500)", height, right)
        field("Seed (необязательно)", seed)
        tk.Checkbutton(frame, text="Замкнутые границы", variable=wrap, bg=PANEL,
                       fg=TEXT, selectcolor=PANEL_2, activebackground=PANEL,
                       activeforeground=TEXT).pack(anchor="w", pady=(9, 0))
        classic_fields = tk.Frame(frame, bg=PANEL)
        field("Правило B/S", rule, classic_fields)
        field("Плотность живых клеток, %", density, classic_fields)
        eco_fields = tk.Frame(frame, bg=PANEL)
        field("Растения, %", plants, eco_fields)
        field("Травоядные, %", herbivores, eco_fields)
        field("Хищники, %", predators, eco_fields)

        def toggle_fields(_event: Any = None) -> None:
            classic_fields.pack_forget()
            eco_fields.pack_forget()
            if mode.get() == "Классика":
                classic_fields.pack(fill="x")
            else:
                eco_fields.pack(fill="x")

        mode_combo.bind("<<ComboboxSelected>>", toggle_fields)
        toggle_fields()
        self._label(frame, "При создании текущий мир будет заменён.\nСохраните его заранее, если он нужен.", 9).pack(anchor="w", pady=(13, 5))

        def confirm() -> None:
            try:
                options = WorldOptions(
                    mode="classic" if mode.get() == "Классика" else "ecosystem",
                    width=int(width.get()), height=int(height.get()), wrap=wrap.get(),
                    rule=rule.get(), density=float(density.get().replace(",", ".")) / 100,
                    plants=float(plants.get().replace(",", ".")) / 100,
                    herbivores=float(herbivores.get().replace(",", ".")) / 100,
                    predators=float(predators.get().replace(",", ".")) / 100,
                    seed=int(seed.get()) if seed.get().strip() else None)
                new_world = create_world(options)
            except (ValueError, OverflowError) as exc:
                messagebox.showerror("Неверные параметры мира", str(exc), parent=dialog)
                return
            self.running = False
            self._update_run_button()
            self.world = new_world
            if isinstance(new_world, Ecosystem):
                new_world.ecology.enabled = True
            self.wrap_var.set(options.wrap)
            self.density_var.set(options.density if options.mode == "classic"
                                 else options.plants + options.herbivores + options.predators)
            if isinstance(new_world, CellularAutomaton):
                self.rule_entry_var.set(new_world.rule)
            self.undo.clear()
            self.redo.clear()
            self.tool = "draw"
            self.chart_mode_var.set("Численность")
            self._update_mode_ui()
            self._update_tool_ui()
            self._reset_visuals()
            dialog.destroy()
            self.fit_world()
            self._show_hint("Создан мир с заданными начальными условиями")

        self._button(frame, "СОЗДАТЬ МИР", confirm, accent=True).pack(fill="x", pady=(8, 0))
        dialog.bind("<Return>", lambda _e: confirm() if not isinstance(dialog.focus_get(), ttk.Combobox) else None)
        dialog.bind("<Escape>", lambda _e: dialog.destroy())
        mode_combo.focus_set()

    def change_mode(self, mode: str) -> None:
        if mode == self.world.mode:
            return
        self.running = False
        self._update_run_button()
        w, h = self.world.width, self.world.height
        if mode == "classic":
            try:
                self.world = CellularAutomaton(w, h, self.rule_entry_var.get() or "B3/S23", self.wrap_var.get())
            except ValueError:
                self.rule_entry_var.set("B3/S23")
                self.world = CellularAutomaton(w, h, "B3/S23", self.wrap_var.get())
        else:
            self.world = Ecosystem(w, h, wrap=self.wrap_var.get())
        self.world.randomize(self.density_var.get())
        if isinstance(self.world, Ecosystem):
            self.world.ecology.enabled = True
        self.undo.clear()
        self.redo.clear()
        self.tool = "draw"
        self.chart_mode_var.set("Численность")
        self._update_mode_ui()
        self._update_tool_ui()
        self._reset_visuals()
        self._show_hint("Режим переключён. Это отдельный мир со своими правилами.")

    def _update_mode_ui(self) -> None:
        classic = self.world.mode == "classic"
        self.classic_button.config(bg=GREEN if classic else PANEL_2, fg=BG if classic else TEXT)
        self.eco_button.config(bg=GREEN if not classic else PANEL_2, fg=BG if not classic else TEXT)
        if classic:
            self.pattern_controls.pack(fill="x")
            self.species_frame.pack_forget()
            self.rules_frame.pack(fill="x", pady=(0, 8), before=self.actions_frame)
            self.rle_row.pack(fill="x", before=self.png_button)
            self.world_mode_label.config(text=f"КОНВЕЙ · {self.world.rule}", fg=GREEN)
        else:
            self.pattern_controls.pack_forget()
            self.species_frame.pack(fill="x")
            self.rules_frame.pack_forget()
            self.rle_row.pack_forget()
            self.world_mode_label.config(text="ЭКОСИСТЕМА · ЭКСПЕРИМЕНТ", fg=YELLOW)
        for key in ("plants", "herbivores", "predators", "mutations",
                       "metabolism", "sight", "fertility"):
            row = getattr(self, f"_statrow_{key}")
            if classic:
                row.pack_forget()
            else:
                row.pack(fill="x", pady=2)
        self._mark_dirty()

    def apply_preset(self, _: Any = None) -> None:
        self.rule_entry_var.set(RULE_PRESETS[self.rule_name.get()])
        self.apply_rule()

    def apply_rule(self) -> None:
        if not isinstance(self.world, CellularAutomaton):
            return
        try:
            self.world.set_rule(self.rule_entry_var.get())
        except ValueError as exc:
            messagebox.showerror("Ошибка правила", str(exc))
            return
        self.rule_entry_var.set(self.world.rule)
        self.world_mode_label.config(text=f"КОНВЕЙ · {self.world.rule}")
        self._show_hint("Правило применено к текущему миру")

    def toggle_run(self) -> None:
        self.running = not self.running
        self.accumulated = 0.0
        self._update_run_button()

    def _update_run_button(self) -> None:
        self.run_button.config(text="Ⅱ  ПАУЗА" if self.running else "▶  ЗАПУСТИТЬ")

    def step_once(self) -> None:
        self.running = False
        self._update_run_button()
        self._advance()

    def _advance(self) -> None:
        was_live = self.world.cells.copy() if self.world.mode == "classic" else None
        stamp=time.perf_counter()
        events = self.world.step()
        self.step_ms=(time.perf_counter()-stamp)*1000
        if was_live is not None:
            newly_dead = (was_live > 0) & (self.world.cells == 0)
            self.trails[self.trails > 0] -= 1
            self.trails[newly_dead] = 7
            self.trails[self.world.cells != 0] = 0
        self.experiment.record(self.world, events)
        self.research_log.capture(self.world, events)
        self.research_ui.check_extinction()
        if self.world.generation % 5 == 0:
            self.research_ui.refresh()
        self._update_stats()
        self._last_inspected = None
        self._update_cell_inspector(self.pointer)
        self._mark_dirty()

    def randomize(self) -> None:
        self.running = False
        self._update_run_button()
        self._push_undo()
        self.world.randomize(self.density_var.get())
        self._reset_visuals()
        self._show_hint("Создан новый случайный мир")

    def clear(self) -> None:
        self.running = False
        self._update_run_button()
        self._push_undo()
        self.world.clear()
        self._reset_visuals()
        self._show_hint("Мир очищен — можно рисовать с нуля")

    def _update_stats(self) -> None:
        counts = self.world.counts()
        values = {
            "generation": self.world.generation,
            "population": counts["population"],
            "births": self.world.total_births,
            "deaths": self.world.total_deaths,
            "plants": counts.get("plants", 0),
            "herbivores": counts.get("herbivores", 0),
            "predators": counts.get("predators", 0),
            "mutations": counts.get("mutations", 0),
        }
        if isinstance(self.world, Ecosystem):
            values.update(self.world.genome_stats())
        for key, val in values.items():
            formatted = f"{val:.2f}" if isinstance(val, float) else f"{val:,}".replace(",", " ")
            self.stats_labels[key].config(text=formatted)
        self.hud_generation.config(text=f"{self.world.generation:,}".replace(",", " "))
        self.hud_population.config(text=f"{counts['population']:,}".replace(",", " "))
        self.hud_density.config(text=f"{counts['population'] / (self.world.width*self.world.height)*100:.1f}%")
        self._draw_chart()

    def _draw_chart(self) -> None:
        chart = self.chart
        chart.delete("all")
        width, height = max(chart.winfo_width(), 220), 164
        x0, x1, y0, y1 = 12, width-13, 14, height-26
        for i in range(4):
            y = y0 + (y1-y0) * i / 3
            chart.create_line(x0, y, x1, y, fill="#294050", dash=(2, 5))
        rows = self.experiment.records[-120:]
        if len(rows) < 2:
            chart.create_text(width/2, height/2, text="История появится после запуска",
                              fill=MUTED, font=(FONT, 9))
            return
        chart_type = self.chart_mode_var.get() if self.world.mode == "ecosystem" else "Численность"
        if chart_type == "Численность":
            names = ["population"] if self.world.mode == "classic" else ["plants", "herbivores", "predators"]
            colors = [GREEN] if self.world.mode == "classic" else [GREEN, YELLOW, PINK]
        else:
            gene = {"Метаболизм": "metabolism", "Зрение": "sight", "Плодовитость": "fertility"}[chart_type]
            names = [f"herbivores_{gene}", f"predators_{gene}"]
            colors = [YELLOW, PINK]
        seen = [float(row[key]) for row in rows for key in names if row.get(key) is not None]
        ymax = max(1.0, max(seen, default=0)) * 1.05
        chart.create_text(x1, 4, text=f"{ymax:.2f}" if chart_type != "Численность" else f"{int(ymax)}",
                          fill=MUTED, anchor="ne", font=(FONT, 8))
        chart.create_text(x0, height-9, text=f"{rows[0]['generation']}", fill=MUTED,
                          anchor="w", font=(FONT, 8))
        chart.create_text(x1, height-9, text=f"{rows[-1]['generation']}", fill=MUTED,
                          anchor="e", font=(FONT, 8))
        for name, color in zip(names, colors):
            segment: list[float] = []
            def flush() -> None:
                if len(segment) >= 4:
                    chart.create_line(*segment, fill=color, width=2, smooth=False)
            for i, row in enumerate(rows):
                value = row.get(name)
                if value is None:
                    flush()
                    segment = []
                    continue
                x = x0 + i / (len(rows)-1) * (x1-x0)
                y = y1 - float(value) / ymax * (y1-y0)
                segment.extend((x, y))
            flush()
            if len(segment) >= 2:
                chart.create_oval(segment[-2]-2, segment[-1]-2,
                                  segment[-2]+2, segment[-1]+2, fill=color, outline="")

    def fit_world(self) -> None:
        w, h = max(self.canvas.winfo_width(), 500), max(self.canvas.winfo_height(), 400)
        self.zoom = max(2, min(26, int(min((w - 25) / self.world.width, (h - 25) / self.world.height))))
        self.cam_x = (w - self.world.width * self.zoom) / 2
        self.cam_y = (h - self.world.height * self.zoom) / 2
        self._mark_dirty()

    def _screen_to_grid(self, sx: int, sy: int) -> tuple[int, int]:
        return math.floor((sx - self.cam_x) / self.zoom), math.floor((sy - self.cam_y) / self.zoom)

    def _draw_line(self, start: tuple[int, int], stop: tuple[int, int], kind: int) -> None:
        x0, y0 = start
        x1, y1 = stop
        dx, dy = abs(x1-x0), -abs(y1-y0)
        sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
        err = dx + dy
        while True:
            self.world.paint(x0, y0, kind)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def _mouse_down(self, e: tk.Event[Any]) -> None:
        if e.state & 0x4:  # Ctrl+click inspects an animal without painting.
            x, y = self._screen_to_grid(e.x, e.y)
            self.research_ui.select_animal(x, y)
            return
        if e.state & 0x1:  # Shift + left drag = pan.
            self._pan_down(e)
            return
        x, y = self._screen_to_grid(e.x, e.y)
        if not (0 <= x < self.world.width and 0 <= y < self.world.height):
            return
        self._push_undo()
        self.dragging = True
        self.pencil_value = 0 if self.tool == "erase" else (self.species if self.world.mode == "ecosystem" else 1)
        if self.tool == "stamp" and isinstance(self.world, CellularAutomaton):
            place_pattern(self.world.cells, self.research_ui.stamp_array(self.pattern_name.get()), x, y)
            self.dragging = False
        else:
            self.world.paint(x, y, self.pencil_value)
        self.last_paint = x, y
        self._update_stats()
        self._mark_dirty()

    def _mouse_drag(self, e: tk.Event[Any]) -> None:
        if self.panning:
            self._pan_drag(e)
            return
        self._mouse_motion(e)
        if not self.dragging:
            return
        here = self._screen_to_grid(e.x, e.y)
        if here != self.last_paint:
            self._draw_line(self.last_paint or here, here, self.pencil_value)
            self.last_paint = here
            self._mark_dirty()

    def _mouse_up(self, _: tk.Event[Any]) -> None:
        self.dragging = False
        self.panning = False
        self.last_paint = None
        self._update_stats()

    def _right_down(self, e: tk.Event[Any]) -> None:
        """Right button always pans; erasing is the left-button eraser tool."""
        self._pan_down(e)

    def _right_drag(self, e: tk.Event[Any]) -> None:
        self._pan_drag(e)

    def _pan_down(self, e: tk.Event[Any]) -> None:
        self.panning = True
        self.pan_anchor = (e.x, e.y)
        self.dragging = False

    def _pan_drag(self, e: tk.Event[Any]) -> None:
        if self.pan_anchor:
            self.cam_x += e.x - self.pan_anchor[0]
            self.cam_y += e.y - self.pan_anchor[1]
            self.pan_anchor = (e.x, e.y)
            self._mark_dirty()

    def _pan_up(self, _: tk.Event[Any]) -> None:
        self.panning = False
        self.pan_anchor = None

    def _mouse_motion(self, e: tk.Event[Any]) -> None:
        xy = self._screen_to_grid(e.x, e.y)
        self.pointer = xy
        self._update_cell_inspector(xy)
        self.coord_label.config(text=f"X: {xy[0]}    Y: {xy[1]}    ·    ×{self.zoom}")
        self._mark_dirty()

    def _mouse_leave(self, _: tk.Event[Any]) -> None:
        self.pointer = None
        self._update_cell_inspector(None)
        self.coord_label.config(text="X: —    Y: —")
        self._mark_dirty()

    def _wheel(self, e: tk.Event[Any]) -> None:
        self._zoom_at(e.x, e.y, 1 if e.delta > 0 else -1)
        return "break"

    def _zoom_at(self, sx: int, sy: int, sign: int) -> None:
        old = self.zoom
        self.zoom = max(2, min(30, self.zoom + (2 if self.zoom >= 14 else 1) * sign))
        if self.zoom == old:
            return
        self.cam_x = sx - (sx - self.cam_x) * self.zoom / old
        self.cam_y = sy - (sy - self.cam_y) * self.zoom / old
        self._mark_dirty()

    def _make_rgb(self, arr: np.ndarray, trails: np.ndarray | None = None,
                  nutrients: np.ndarray | None = None) -> np.ndarray:
        return colorize(arr, self.world.mode, PALETTES[self.palette_var.get()],
                        trails if self.trail_var.get() else None, nutrients)

    def _render(self) -> None:
        canvas = self.canvas
        vw, vh = canvas.winfo_width(), canvas.winfo_height()
        if vw <= 1 or vh <= 1:
            return
        x0 = max(0, math.floor(-self.cam_x / self.zoom))
        y0 = max(0, math.floor(-self.cam_y / self.zoom))
        x1 = min(self.world.width, math.ceil((vw-self.cam_x) / self.zoom))
        y1 = min(self.world.height, math.ceil((vh-self.cam_y) / self.zoom))
        canvas.delete("all")
        if x0 < x1 and y0 < y1:
            crop = self.world.cells[y0:y1, x0:x1]
            soil = self.world.nutrients[y0:y1, x0:x1] if isinstance(self.world, Ecosystem) else None
            fading = self.trails[y0:y1, x0:x1] if self.world.mode == "classic" and self.trail_var.get() else None
            if isinstance(self.world,Ecosystem) and self.layer_var.get()!="Организмы":
                fields={"Минералы":self.world.nutrients,"Влага":self.world.moisture,
                        "Энергия":self.world.energy,"Возраст":self.world.age,
                        "Рельеф":self.world.altitude,"Органика":self.world.organic,
                        "Токсины":self.world.toxins}
                layer=self.layer_var.get()
                limits={"Минералы":100,"Влага":100,"Энергия":100,"Возраст":300,
                        "Рельеф":1,"Органика":50,"Токсины":50}
                vals=fields[layer][y0:y1,x0:x1]/limits[layer]
                image=heatmap_view(vals,crop,zoom=self.zoom,palette_name=self.palette_var.get(),grid=self.grid_var.get())
            else:
                image = draw_view(crop, mode=self.world.mode, zoom=self.zoom,
                                  palette_name=self.palette_var.get(), style=self.cell_style_var.get(),
                                  grid=self.grid_var.get(), trails=fading, nutrients=soil)
            self.current_image = ImageTk.PhotoImage(image)
            canvas.create_image(self.cam_x+x0*self.zoom, self.cam_y+y0*self.zoom,
                                anchor="nw", image=self.current_image)
        canvas.create_rectangle(self.cam_x, self.cam_y,
                                self.cam_x+self.world.width*self.zoom,
                                self.cam_y+self.world.height*self.zoom,
                                outline=PALETTES[self.palette_var.get()].border, width=1)
        if self.pointer:
            x, y = self.pointer
            if 0 <= x < self.world.width and 0 <= y < self.world.height:
                sx, sy = self.cam_x+x*self.zoom, self.cam_y+y*self.zoom
                if self.tool == "stamp" and self.world.mode == "classic":
                    pat = self.research_ui.stamp_array(self.pattern_name.get())
                    for yy, xx in np.argwhere(pat):
                        qx, qy = sx + int(xx)*self.zoom, sy + int(yy)*self.zoom
                        canvas.create_rectangle(qx, qy, qx+self.zoom, qy+self.zoom,
                                                outline=GREEN, width=2)
                else:
                    canvas.create_rectangle(sx, sy, sx+self.zoom, sy+self.zoom,
                                            outline=PINK if self.tool == "erase" else GREEN, width=2)
        self.zoom_label.config(text=f"{self.zoom} px / кл.")
        self._draw_minimap()

    def _tick(self) -> None:
        now = time.perf_counter()
        dt = min(0.15, now - self.last_t)
        self.last_t = now
        self.fps_samples.append(dt)
        if self.running:
            self.accumulated += dt
            period = 1.0 / self.speed_var.get()
            updates = 0
            while self.accumulated >= period and updates < 3:
                self.accumulated -= period
                self._advance()
                updates += 1
            if updates == 3:
                self.accumulated = min(self.accumulated, period)
        if self.dirty:
            self._render()
            self.dirty = False
        if self.world.generation % 5 == 0:
            fps=1/max(0.0001,sum(self.fps_samples)/max(1,len(self.fps_samples)))
            self.performance_label.config(text=(f"{self.world.width}×{self.world.height} · "
                 f"{self.step_ms:.1f} МС/ШАГ · {fps:.0f} FPS · v{__version__}"))
        self.root.after(30, self._tick)

    def save(self) -> None:
        path = filedialog.asksaveasfilename(title="Сохранить мир", defaultextension=".lifelab",
                                            filetypes=[("Мир Life", "*.lifelab")])
        if path:
            try:
                save_world(self.world, path, self.experiment, self.research_log)
                self._show_hint(f"Сохранено: {Path(path).name}")
            except Exception as exc:
                messagebox.showerror("Не удалось сохранить", str(exc))

    def load(self) -> None:
        path = filedialog.askopenfilename(title="Открыть мир", filetypes=[("Мир Life", "*.lifelab")])
        if path:
            try:
                self.world, history = load_document(path)
                saved_research = load_research(path)
                self.running = False
                self._update_run_button()
                self.wrap_var.set(self.world.wrap)
                if isinstance(self.world, CellularAutomaton):
                    self.rule_entry_var.set(self.world.rule)
                self.undo.clear()
                self.redo.clear()
                self.tool = "draw"
                self.chart_mode_var.set("Численность")
                self._update_mode_ui()
                self._update_tool_ui()
                self._reset_visuals(history)
                self.research_log = saved_research
                if not self.research_log.records:
                    self.research_log.capture(self.world)
                self.research_ui.sync_ecology()
                self.research_ui.refresh(force=True)
                self.fit_world()
                self._show_hint(f"Загружено: {Path(path).name}")
            except Exception as exc:
                messagebox.showerror("Не удалось открыть мир", str(exc))

    def import_rle(self) -> None:
        if self.world.mode != "classic":
            return
        path = filedialog.askopenfilename(title="Импорт шаблона", filetypes=[("Life RLE", "*.rle"), ("Текст", "*.txt")])
        if not path:
            return
        try:
            arr, rule = decode_rle(Path(path).read_text(encoding="utf-8-sig"))
            self._push_undo()
            place_pattern(self.world.cells, arr,
                          (self.world.width - arr.shape[1]) // 2,
                          (self.world.height - arr.shape[0]) // 2)
            if rule:
                self.world.set_rule(rule)
                self.rule_entry_var.set(self.world.rule)
                self.world_mode_label.config(text=f"КОНВЕЙ · {self.world.rule}")
            self._mark_dirty()
            self._update_stats()
            self._show_hint(f"Шаблон импортирован: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Ошибка импорта RLE", str(exc))

    def export_rle(self) -> None:
        if not isinstance(self.world, CellularAutomaton):
            return
        path = filedialog.asksaveasfilename(defaultextension=".rle", filetypes=[("Life RLE", "*.rle")])
        if path:
            try:
                Path(path).write_text(encode_rle(self.world.cells, self.world.rule), encoding="utf-8")
                self._show_hint(f"RLE сохранён: {Path(path).name}")
            except Exception as exc:
                messagebox.showerror("Ошибка экспорта RLE", str(exc))

    def export_csv(self) -> None:
        path = filedialog.asksaveasfilename(title="Экспорт истории эксперимента",
                                            defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            try:
                self.experiment.export_csv(path)
                self._show_hint(f"CSV сохранён: {Path(path).name}")
            except Exception as exc:
                messagebox.showerror("Ошибка экспорта CSV", str(exc))

    def export_png(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("Изображение PNG", "*.png")])
        if path:
            try:
                soil = self.world.nutrients if isinstance(self.world, Ecosystem) else None
                fading = self.trails if self.world.mode == "classic" else None
                if isinstance(self.world,Ecosystem) and self.layer_var.get()!="Организмы":
                    layers={"Минералы":(self.world.nutrients,100),"Влага":(self.world.moisture,100),
                            "Энергия":(self.world.energy,100),"Возраст":(self.world.age,300),
                            "Рельеф":(self.world.altitude,1),"Органика":(self.world.organic,50),
                            "Токсины":(self.world.toxins,50)}
                    layer,maximum=layers[self.layer_var.get()]
                    im=heatmap_view(layer/maximum,self.world.cells,zoom=8,
                                    palette_name=self.palette_var.get(),grid=self.grid_var.get())
                else:
                    im = draw_view(self.world.cells, mode=self.world.mode, zoom=8,
                                   palette_name=self.palette_var.get(), style=self.cell_style_var.get(),
                                   grid=self.grid_var.get(), trails=fading if self.trail_var.get() else None,
                                   nutrients=soil)
                im.save(path)
                self._show_hint(f"PNG сохранён: {Path(path).name}")
            except Exception as exc:
                messagebox.showerror("Ошибка экспорта PNG", str(exc))
