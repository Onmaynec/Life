"""Tkinter desktop UI; NumPy runs the simulation, Pillow draws the viewport."""
from __future__ import annotations

from collections import deque
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
from .storage import load_world, save_world
from .world import CellularAutomaton, Ecosystem, RULE_PRESETS


BG = "#0B1018"
PANEL = "#121B28"
PANEL_2 = "#182434"
EDGE = "#273649"
TEXT = "#EAF2FA"
MUTED = "#8799AE"
GREEN = "#6BE6B6"
YELLOW = "#F6C96B"
PINK = "#FF708D"
CYAN = "#71BBFF"
FONT = "Segoe UI"


def rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(color[i:i+2], 16) for i in (1, 3, 5))


class LifeLabApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title(f"Life  —  v{__version__}")
        self.root.geometry("1490x890")
        self.root.minsize(1080, 690)
        self.root.configure(bg=BG)
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)

        self.world: CellularAutomaton | Ecosystem = CellularAutomaton()
        self.world.randomize(0.22, seed=42)
        self.running = False
        self.speed_var = tk.IntVar(value=8)
        self.zoom = 9
        self.cam_x = 55.0
        self.cam_y = 42.0
        self.grid_var = tk.BooleanVar(value=True)
        self.wrap_var = tk.BooleanVar(value=False)
        self.trail_var = tk.BooleanVar(value=True)
        self.density_var = tk.DoubleVar(value=0.22)
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
        self.history: deque[dict[str, int]] = deque(maxlen=120)
        self.trails = np.zeros_like(self.world.cells, dtype=np.uint8)
        self.undo: list[tuple[np.ndarray, ...]] = []
        self.redo: list[tuple[np.ndarray, ...]] = []
        self.current_image: ImageTk.PhotoImage | None = None
        self.fps_samples: deque[float] = deque(maxlen=25)

        self._configure_styles()
        self._build_ui()
        self._bind_events()
        self._update_mode_ui()
        self._update_tool_ui()
        self._update_stats()
        self._show_hint("Пробел — запуск  ·  ЛКМ — рисовать  ·  Колесо — масштаб  ·  СКМ — перемещение")
        self.root.after(35, self._tick)

    def run(self) -> None:
        self.root.mainloop()

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
                      fg=BG if accent else TEXT, activebackground="#8bf9cd" if accent else EDGE,
                      activeforeground=BG if accent else TEXT, relief="flat", borderwidth=0,
                      font=(FONT, 10, "bold"), cursor="hand2", pady=9, padx=12,
                      highlightthickness=0)
        if width is not None:
            b.config(width=width)
        return b

    def _label(self, parent: tk.Misc, text: str, size: int = 10,
               color: str = MUTED, bold: bool = False) -> tk.Label:
        return tk.Label(parent, text=text, fg=color, bg=parent.cget("bg"),
                        anchor="w", font=(FONT, size, "bold" if bold else "normal"))

    def _section(self, parent: tk.Misc, title: str) -> tk.Frame:
        f = tk.Frame(parent, bg=PANEL, pady=2)
        self._label(f, title.upper(), 9, MUTED, True).pack(anchor="w", pady=(8, 10))
        return f

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, bg=BG, padx=26, pady=17)
        header.pack(fill="x")
        branding = tk.Frame(header, bg=BG)
        branding.pack(side="left")
        self._label(branding, "LIFE", 23, TEXT, True).pack(anchor="w")
        self._label(branding, "ИНТЕРАКТИВНАЯ ЛАБОРАТОРИЯ КЛЕТОЧНЫХ МИРОВ", 9, MUTED).pack(anchor="w")
        controls = tk.Frame(header, bg=BG)
        controls.pack(side="right")
        self._label(controls, "ШАГОВ / СЕК", 9).pack(side="left", padx=(0, 10))
        speed = tk.Scale(controls, from_=1, to=30, orient="horizontal", variable=self.speed_var,
                         bg=BG, fg=TEXT, troughcolor=PANEL_2, activebackground=GREEN,
                         highlightthickness=0, borderwidth=0, showvalue=True,
                         length=160, resolution=1)
        speed.pack(side="left", padx=(0, 18))
        self._button(controls, "ОДИН ШАГ", self.step_once).pack(side="left", padx=(0, 9))
        self.run_button = self._button(controls, "▶  ЗАПУСТИТЬ", self.toggle_run, accent=True)
        self.run_button.pack(side="left")

        outer = tk.Frame(self.root, bg=BG, padx=22, pady=4)
        outer.pack(fill="both", expand=True)
        left = tk.Frame(outer, bg=PANEL, highlightbackground=EDGE, highlightthickness=1)
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))
        canvas_top = tk.Frame(left, bg=PANEL, padx=14, pady=12)
        canvas_top.pack(fill="x")
        self._label(canvas_top, "01  /  ЖИВОЙ МИР", 11, TEXT, True).pack(side="left")
        self.world_mode_label = self._label(canvas_top, "CLASSIC · B3/S23", 9, GREEN, True)
        self.world_mode_label.pack(side="right")
        self.canvas = tk.Canvas(left, bg="#0C1420", bd=0, highlightthickness=0,
                                cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=1)
        foot = tk.Frame(left, bg=PANEL, padx=13, pady=10)
        foot.pack(fill="x")
        self.coord_label = self._label(foot, "X: —    Y: —", 10, TEXT)
        self.coord_label.pack(side="left")
        self._label(foot, "СКМ: СДВИГ  ·  КОЛЕСО: МАСШТАБ", 9).pack(side="right")

        right = tk.Frame(outer, bg=PANEL, width=315, highlightbackground=EDGE, highlightthickness=1)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        self.side_scroll = tk.Canvas(right, bg=PANEL, highlightthickness=0, bd=0)
        self.side_scroll.pack(side="left", fill="both", expand=True)
        scrollbar = tk.Scrollbar(right, orient="vertical", command=self.side_scroll.yview,
                                 troughcolor=PANEL, bg=PANEL_2, activebackground=EDGE)
        scrollbar.pack(side="right", fill="y")
        self.side_scroll.configure(yscrollcommand=scrollbar.set)
        self.side = tk.Frame(self.side_scroll, bg=PANEL, padx=18, pady=8)
        window = self.side_scroll.create_window((0, 0), window=self.side, anchor="nw")
        self.side.bind("<Configure>", lambda e: self.side_scroll.configure(scrollregion=self.side_scroll.bbox("all")))
        self.side_scroll.bind("<Configure>", lambda e: self.side_scroll.itemconfigure(window, width=e.width))
        self.side_scroll.bind("<MouseWheel>", self._scroll_sidebar)
        self.side.bind("<MouseWheel>", self._scroll_sidebar)

        mode = self._section(self.side, "Режим симуляции")
        mode.pack(fill="x", pady=(0, 8))
        two = tk.Frame(mode, bg=PANEL)
        two.pack(fill="x")
        self.classic_button = self._button(two, "КЛАССИКА", lambda: self.change_mode("classic"))
        self.classic_button.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.eco_button = self._button(two, "ЭКОСИСТЕМА", lambda: self.change_mode("ecosystem"))
        self.eco_button.pack(side="left", fill="x", expand=True)

        tools = self._section(self.side, "Инструменты")
        tools.pack(fill="x", pady=(0, 8))
        toolbox = tk.Frame(tools, bg=PANEL)
        toolbox.pack(fill="x")
        self.draw_button = self._button(toolbox, "●  РИСОВАТЬ", lambda: self.set_tool("draw"))
        self.draw_button.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.erase_button = self._button(toolbox, "×  СТЕРЕТЬ", lambda: self.set_tool("erase"))
        self.erase_button.pack(side="left", fill="x", expand=True)
        self.pattern_controls = tk.Frame(tools, bg=PANEL, pady=7)
        self._label(self.pattern_controls, "Готовые структуры", 9).pack(anchor="w", pady=(1, 5))
        self.pattern_combo = ttk.Combobox(self.pattern_controls, textvariable=self.pattern_name,
                                           values=list(PATTERNS), state="readonly", style="Lab.TCombobox")
        self.pattern_combo.pack(fill="x")
        self.pattern_combo.bind("<<ComboboxSelected>>", lambda e: self.set_tool("stamp"))
        self.stamp_button = self._button(self.pattern_controls, "ПОСТАВИТЬ ШАБЛОН", lambda: self.set_tool("stamp"))
        self.stamp_button.pack(fill="x", pady=(6, 0))
        self.species_frame = tk.Frame(tools, bg=PANEL, pady=6)
        self._label(self.species_frame, "Выбор типа клетки", 9).pack(anchor="w", pady=(0, 6))
        self.species_buttons: dict[int, tk.Button] = {}
        for kind, title, color in ((1, "Растения", GREEN), (2, "Травоядные", YELLOW), (3, "Хищники", PINK)):
            b = self._button(self.species_frame, "●  " + title, lambda k=kind: self.set_species(k))
            b.pack(fill="x", pady=2)
            self.species_buttons[kind] = b

        self.rules_frame = self._section(self.side, "Законы мира")
        self.rules_frame.pack(fill="x", pady=(0, 8))
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
        self.rule_entry.bind("<Return>", lambda e: self.apply_rule())

        actions = self._section(self.side, "Управление миром")
        self.actions_frame = actions
        actions.pack(fill="x", pady=(0, 8))
        acts = tk.Frame(actions, bg=PANEL)
        acts.pack(fill="x")
        self._button(acts, "СЛУЧАЙНЫЙ", self.randomize).pack(side="left", fill="x", expand=True, padx=(0, 5))
        self._button(acts, "ОЧИСТИТЬ", self.clear).pack(side="left", fill="x", expand=True)
        undo_row = tk.Frame(actions, bg=PANEL, pady=7)
        undo_row.pack(fill="x")
        self._button(undo_row, "↶  ОТМЕНИТЬ", self.undo_action).pack(side="left", fill="x", expand=True, padx=(0, 5))
        self._button(undo_row, "↷  ВЕРНУТЬ", self.redo_action).pack(side="left", fill="x", expand=True)
        self._label(actions, "Плотность случайного заполнения", 9).pack(anchor="w", pady=(6, 0))
        d = tk.Scale(actions, from_=0.02, to=0.65, resolution=0.01, orient="horizontal",
                     variable=self.density_var, bg=PANEL, fg=TEXT, troughcolor=PANEL_2,
                     activebackground=GREEN, highlightthickness=0, borderwidth=0, length=240)
        d.pack(fill="x")
        for title, var, command in (("Замкнутые границы (тор)", self.wrap_var, self._toggle_wrap),
                                    ("Отображать сетку", self.grid_var, self._mark_dirty),
                                    ("Следы прошлых поколений", self.trail_var, self._mark_dirty)):
            cb = tk.Checkbutton(actions, text=title, variable=var, command=command,
                                bg=PANEL, fg=TEXT, selectcolor=PANEL_2,
                                activebackground=PANEL, activeforeground=TEXT,
                                highlightthickness=0, anchor="w", font=(FONT, 10))
            cb.pack(fill="x", pady=1)

        stats = self._section(self.side, "Статистика")
        stats.pack(fill="x", pady=(0, 8))
        self.stats_labels: dict[str, tk.Label] = {}
        for key, title in (("generation", "Поколение"), ("population", "Живых клеток"),
                           ("births", "Всего рождений"), ("deaths", "Всего смертей"),
                           ("plants", "Растения"), ("herbivores", "Травоядные"),
                           ("predators", "Хищники"), ("mutations", "Мутации")):
            row = tk.Frame(stats, bg=PANEL)
            row.pack(fill="x", pady=2)
            self._label(row, title, 10).pack(side="left")
            l = self._label(row, "0", 11, TEXT, True)
            l.pack(side="right")
            self.stats_labels[key] = l
            if key == "plants":
                self.eco_stats_start = row
            if key in ("plants", "herbivores", "predators", "mutations"):
                row.pack_forget()
                setattr(self, f"_statrow_{key}", row)

        graph_section = self._section(self.side, "История популяции")
        graph_section.pack(fill="x", pady=(0, 8))
        self.chart = tk.Canvas(graph_section, height=105, bg="#0D1723", highlightthickness=0)
        self.chart.pack(fill="x")
        self._label(graph_section, "Последние 120 поколений", 9).pack(anchor="w", pady=(4, 0))

        files = self._section(self.side, "Файлы и обмен")
        files.pack(fill="x", pady=(0, 14))
        f1 = tk.Frame(files, bg=PANEL)
        f1.pack(fill="x")
        self._button(f1, "СОХРАНИТЬ", self.save).pack(side="left", fill="x", expand=True, padx=(0, 5))
        self._button(f1, "ОТКРЫТЬ", self.load).pack(side="left", fill="x", expand=True)
        self.rle_row = tk.Frame(files, bg=PANEL, pady=7)
        self.rle_row.pack(fill="x")
        self._button(self.rle_row, "ИМПОРТ RLE", self.import_rle).pack(side="left", fill="x", expand=True, padx=(0, 5))
        self._button(self.rle_row, "ЭКСПОРТ RLE", self.export_rle).pack(side="left", fill="x", expand=True)
        self.png_button = self._button(files, "СОХРАНИТЬ PNG", self.export_png)
        self.png_button.pack(fill="x")

        status = tk.Frame(self.root, bg=BG, padx=26, pady=11)
        status.pack(fill="x")
        self.notice_label = self._label(status, "", 10, MUTED)
        self.notice_label.pack(side="left")
        self.performance_label = self._label(status, "", 9, GREEN)
        self.performance_label.pack(side="right")

    def _bind_events(self) -> None:
        self.root.bind_all("<space>", self._key_space)
        self.root.bind_all("<Control-s>", lambda e: self.save())
        self.root.bind_all("<Control-o>", lambda e: self.load())
        self.root.bind_all("<Control-z>", lambda e: self.undo_action())
        self.root.bind_all("<Control-y>", lambda e: self.redo_action())
        self.root.bind("<KeyPress-n>", lambda e: self.step_once())
        self.root.bind("<KeyPress-r>", lambda e: self.randomize())
        self.root.bind("<KeyPress-c>", lambda e: self.clear())
        self.root.bind("<KeyPress-g>", lambda e: self._toggle_grid())
        self.root.bind("<KeyPress-f>", lambda e: self.fit_world())
        self.root.bind("<KeyPress-1>", lambda e: self.set_species(1))
        self.root.bind("<KeyPress-2>", lambda e: self.set_species(2))
        self.root.bind("<KeyPress-3>", lambda e: self.set_species(3))
        self.root.bind("<KeyPress-Escape>", lambda e: self.set_tool("draw"))
        self.canvas.bind("<ButtonPress-1>", self._mouse_down)
        self.canvas.bind("<B1-Motion>", self._mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._mouse_up)
        self.canvas.bind("<ButtonPress-3>", self._right_down)
        self.canvas.bind("<B3-Motion>", self._right_drag)
        self.canvas.bind("<ButtonRelease-3>", self._mouse_up)
        self.canvas.bind("<ButtonPress-2>", self._pan_down)
        self.canvas.bind("<B2-Motion>", self._pan_drag)
        self.canvas.bind("<ButtonRelease-2>", self._pan_up)
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", lambda e: self._zoom_at(e.x, e.y, 1))
        self.canvas.bind("<Button-5>", lambda e: self._zoom_at(e.x, e.y, -1))
        self.canvas.bind("<Motion>", self._mouse_motion)
        self.canvas.bind("<Leave>", self._mouse_leave)
        self.canvas.bind("<Configure>", self._mark_dirty)

    def _key_space(self, e: tk.Event[Any]) -> None:
        if isinstance(self.root.focus_get(), (tk.Entry, ttk.Combobox)):
            return
        self.toggle_run()

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

    def _snapshot(self) -> tuple[np.ndarray, ...]:
        if isinstance(self.world, Ecosystem):
            return tuple(a.copy() for a in (self.world.cells, self.world.energy,
                                             self.world.trait, self.world.nutrients))
        return (self.world.cells.copy(),)

    def _restore(self, snapshot: tuple[np.ndarray, ...]) -> None:
        if self.world.cells.shape != snapshot[0].shape:
            return
        if isinstance(self.world, Ecosystem):
            for arr, data in zip((self.world.cells, self.world.energy,
                                  self.world.trait, self.world.nutrients), snapshot):
                arr[:] = data
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

    def _reset_visuals(self) -> None:
        self.trails = np.zeros_like(self.world.cells, dtype=np.uint8)
        self.history.clear()
        self._update_stats()
        self._mark_dirty()

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
        self.undo.clear()
        self.redo.clear()
        self.tool = "draw"
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
            self.world_mode_label.config(text=f"CLASSIC · {self.world.rule}", fg=GREEN)
        else:
            self.pattern_controls.pack_forget()
            self.species_frame.pack(fill="x")
            self.rules_frame.pack_forget()
            self.rle_row.pack_forget()
            self.world_mode_label.config(text="ECOSYSTEM · EXPERIMENTAL", fg=YELLOW)
        for key in ("plants", "herbivores", "predators", "mutations"):
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
        self.world_mode_label.config(text=f"CLASSIC · {self.world.rule}")
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
        self.world.step()
        if was_live is not None:
            newly_dead = (was_live > 0) & (self.world.cells == 0)
            self.trails[self.trails > 0] -= 1
            self.trails[newly_dead] = 7
            self.trails[self.world.cells != 0] = 0
        self.history.append(self.world.counts())
        self._update_stats()
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
        for key, val in values.items():
            self.stats_labels[key].config(text=f"{val:,}".replace(",", " "))
        self._draw_chart()

    def _draw_chart(self) -> None:
        chart = self.chart
        chart.delete("all")
        w = max(chart.winfo_width(), 255)
        h = 105
        chart.create_line(0, h - 16, w, h - 16, fill=EDGE)
        if len(self.history) < 2:
            return
        names = ["population"] if self.world.mode == "classic" else ["plants", "herbivores", "predators"]
        colors = [GREEN] if self.world.mode == "classic" else [GREEN, YELLOW, PINK]
        max_v = max(max(float(x.get(name, 0)) for x in self.history) for name in names)
        max_v = max(5, max_v)
        for name, color in zip(names, colors):
            coords: list[float] = []
            for i, entry in enumerate(self.history):
                coords.extend((7 + i / max(1, len(self.history) - 1) * (w - 14),
                               h - 16 - float(entry.get(name, 0)) / max_v * (h - 29)))
            if len(coords) >= 4:
                chart.create_line(*coords, fill=color, width=2, smooth=True)

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
            place_pattern(self.world.cells, pattern_array(self.pattern_name.get()), x, y)
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
        x, y = self._screen_to_grid(e.x, e.y)
        if 0 <= x < self.world.width and 0 <= y < self.world.height:
            self._push_undo()
            self.dragging = True
            self.pencil_value = 0
            self.world.paint(x, y, 0)
            self.last_paint = (x, y)
            self._mark_dirty()

    def _right_drag(self, e: tk.Event[Any]) -> None:
        self._mouse_drag(e)

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
        self.coord_label.config(text=f"X: {xy[0]}    Y: {xy[1]}    ·    ×{self.zoom}")
        self._mark_dirty()

    def _mouse_leave(self, _: tk.Event[Any]) -> None:
        self.pointer = None
        self.coord_label.config(text="X: —    Y: —")
        self._mark_dirty()

    def _wheel(self, e: tk.Event[Any]) -> None:
        self._zoom_at(e.x, e.y, 1 if e.delta > 0 else -1)

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
        h, w = arr.shape
        image = np.zeros((h, w, 3), dtype=np.uint8)
        if self.world.mode == "classic":
            image[:] = [13, 24, 37]
            if trails is not None and self.trail_var.get():
                strength = trails.astype(np.float32) / 7.0
                for c, a in enumerate((24, 91, 74)):
                    image[..., c] = (image[..., c].astype(float) + strength * a).astype(np.uint8)
            image[arr == 1] = (105, 236, 181)
        else:
            # Nutrients subtly illuminate the soil; occupied species overlay it.
            soil = (nutrients if nutrients is not None else np.zeros(arr.shape)).astype(np.float32)
            image[..., 0] = np.clip(12 + soil * 0.10, 0, 255).astype(np.uint8)
            image[..., 1] = np.clip(22 + soil * 0.14, 0, 255).astype(np.uint8)
            image[..., 2] = np.clip(32 + soil * 0.06, 0, 255).astype(np.uint8)
            image[arr == 1] = (100, 214, 140)
            image[arr == 2] = (245, 193, 92)
            image[arr == 3] = (255, 106, 139)
        return image

    def _render(self) -> None:
        canvas = self.canvas
        vw, vh = canvas.winfo_width(), canvas.winfo_height()
        if vw <= 1 or vh <= 1:
            return
        x0 = max(0, math.floor(-self.cam_x / self.zoom))
        y0 = max(0, math.floor(-self.cam_y / self.zoom))
        x1 = min(self.world.width, math.ceil((vw - self.cam_x) / self.zoom))
        y1 = min(self.world.height, math.ceil((vh - self.cam_y) / self.zoom))
        canvas.delete("all")
        if x0 >= x1 or y0 >= y1:
            return
        crop = self.world.cells[y0:y1, x0:x1]
        soil = self.world.nutrients[y0:y1, x0:x1] if isinstance(self.world, Ecosystem) else None
        fading = self.trails[y0:y1, x0:x1] if self.world.mode == "classic" else None
        image = Image.fromarray(self._make_rgb(crop, fading, soil), "RGB")
        iw, ih = (x1-x0)*self.zoom, (y1-y0)*self.zoom
        image = image.resize((iw, ih), Image.Resampling.NEAREST)
        draw = ImageDraw.Draw(image)
        if self.grid_var.get() and self.zoom >= 7:
            line = (27, 44, 56) if self.world.mode == "classic" else (39, 49, 50)
            for x in range(self.zoom, iw, self.zoom):
                draw.line((x, 0, x, ih), fill=line, width=1)
            for y in range(self.zoom, ih, self.zoom):
                draw.line((0, y, iw, y), fill=line, width=1)
        self.current_image = ImageTk.PhotoImage(image)
        canvas.create_image(self.cam_x + x0*self.zoom, self.cam_y + y0*self.zoom,
                            anchor="nw", image=self.current_image)
        canvas.create_rectangle(self.cam_x, self.cam_y,
                                self.cam_x + self.world.width*self.zoom,
                                self.cam_y + self.world.height*self.zoom,
                                outline="#496477", width=1)
        if self.pointer:
            x, y = self.pointer
            if 0 <= x < self.world.width and 0 <= y < self.world.height:
                sx, sy = self.cam_x + x*self.zoom, self.cam_y + y*self.zoom
                if self.tool == "stamp" and self.world.mode == "classic":
                    p = pattern_array(self.pattern_name.get())
                    for yy, xx in np.argwhere(p):
                        qx, qy = sx + int(xx)*self.zoom, sy + int(yy)*self.zoom
                        canvas.create_rectangle(qx, qy, qx+self.zoom, qy+self.zoom,
                                                outline=GREEN, width=2)
                else:
                    canvas.create_rectangle(sx, sy, sx+self.zoom, sy+self.zoom,
                                            outline=PINK if self.tool == "erase" else GREEN, width=2)

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
            self.performance_label.config(text=f"{self.world.width}×{self.world.height}  ·  {self.zoom} PX/КЛЕТКА  ·  v{__version__}")
        self.root.after(30, self._tick)

    def save(self) -> None:
        path = filedialog.asksaveasfilename(title="Сохранить мир", defaultextension=".lifelab",
                                            filetypes=[("Мир Life", "*.lifelab")])
        if path:
            try:
                save_world(self.world, path)
                self._show_hint(f"Сохранено: {Path(path).name}")
            except Exception as exc:
                messagebox.showerror("Не удалось сохранить", str(exc))

    def load(self) -> None:
        path = filedialog.askopenfilename(title="Открыть мир", filetypes=[("Мир Life", "*.lifelab")])
        if path:
            try:
                self.world = load_world(path)
                self.running = False
                self._update_run_button()
                self.wrap_var.set(self.world.wrap)
                if isinstance(self.world, CellularAutomaton):
                    self.rule_entry_var.set(self.world.rule)
                self.undo.clear()
                self.redo.clear()
                self.tool = "draw"
                self._update_mode_ui()
                self._update_tool_ui()
                self._reset_visuals()
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
                self.world_mode_label.config(text=f"CLASSIC · {self.world.rule}")
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

    def export_png(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("Изображение PNG", "*.png")])
        if path:
            try:
                soil = self.world.nutrients if isinstance(self.world, Ecosystem) else None
                fading = self.trails if self.world.mode == "classic" else None
                im = Image.fromarray(self._make_rgb(self.world.cells, fading, soil), "RGB")
                im = im.resize((self.world.width * 8, self.world.height * 8), Image.Resampling.NEAREST)
                im.save(path)
                self._show_hint(f"PNG сохранён: {Path(path).name}")
            except Exception as exc:
                messagebox.showerror("Ошибка экспорта PNG", str(exc))
