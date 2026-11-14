"""Comparison window for previously exported Life experiment JSON files."""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

from .experiment_archive import StudyCollection
from .experiment_ui import SERIES_FIELDS, LINE_COLORS
from .skins import skin


class ArchiveComparison:
    def __init__(self, app: Any):
        self.studies = StudyCollection()
        self.colors = skin(app.skin_var.get())
        c = self.colors
        self.window = tk.Toplevel(app.root)
        self.window.title("Life — сравнение архивов экспериментов")
        self.window.geometry("1020x730")
        self.window.minsize(750, 550)
        self.window.configure(bg=c.base)
        self.metric = tk.StringVar(value="Численность")
        self.variant = tk.StringVar(value="Вариант 1")
        head = tk.Frame(self.window, bg=c.panel, padx=16, pady=12)
        head.pack(fill="x")
        tk.Label(head, text="АРХИВ ЭКСПЕРИМЕНТОВ", bg=c.panel, fg=c.text,
                 font=("Segoe UI", 17, "bold")).pack(anchor="w")
        tk.Label(head, text="Открывайте JSON-отчёты; это не сохранения игрового мира.",
                 bg=c.panel, fg=c.muted).pack(anchor="w")
        bar = tk.Frame(self.window, bg=c.panel, padx=12, pady=9)
        bar.pack(fill="x", pady=(7, 9))
        tk.Button(bar, text="ОТКРЫТЬ JSON", command=self.open, bg=c.surface, fg=c.text).pack(side="left", padx=4)
        tk.Button(bar, text="ЭКСПОРТ ОБЩЕГО CSV", command=self.export, bg=c.surface, fg=c.text).pack(side="left", padx=4)
        ttk.Combobox(bar, values=list(SERIES_FIELDS), textvariable=self.metric,
                     state="readonly", width=23).pack(side="right")
        tk.Label(bar, text="ВАРИАНТ", bg=c.panel, fg=c.muted).pack(side="right", padx=(8, 4))
        self.variant_combo = ttk.Combobox(bar, values=("Вариант 1",), textvariable=self.variant,
                                          state="readonly", width=12)
        self.variant_combo.pack(side="right")
        self.metric.trace_add("write", lambda *_: self.draw())
        self.variant.trace_add("write", lambda *_: self.draw())
        self.list = ttk.Treeview(self.window, columns=("file", "mode", "parameter", "range", "seed", "runs", "status"), show="headings", height=6)
        for name, title, width in (("file", "Файл", 220), ("mode", "Режим", 110),
                                   ("parameter", "Параметр", 140), ("range", "Диапазон", 120), ("seed", "Seed", 85),
                                   ("runs", "Повторов", 95), ("status", "Состояние", 140)):
            self.list.heading(name, text=title)
            self.list.column(name, width=width, stretch=True)
        self.list.pack(fill="x", padx=12)
        self.note = tk.StringVar(value="Выберите минимум два JSON-файла, чтобы сравнить серии.")
        tk.Label(self.window, textvariable=self.note, bg=c.base, fg=c.muted,
                 anchor="w", padx=12).pack(fill="x", pady=8)
        self.plot = tk.Canvas(self.window, bg=c.base, highlightthickness=0)
        self.plot.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.plot.bind("<Configure>", lambda _e: self.draw())

    def open(self) -> None:
        paths = filedialog.askopenfilenames(parent=self.window, title="Отчёты Life",
                                            filetypes=[("JSON", "*.json")])
        for path in paths:
            try:
                study = self.studies.add(path)
            except (OSError, ValueError) as exc:
                messagebox.showerror("Не удалось открыть отчёт", f"{path}\n{exc}", parent=self.window)
                continue
            self.list.insert("", "end", values=(study.label, study.plan.mode,
                    study.plan.parameter, f"{study.plan.minimum:g}–{study.plan.maximum:g}",study.plan.seed, study.completed_runs,
                    "Готово" if study.complete else "Частично"))
        self._refresh_variants()
        self.draw()

    def _refresh_variants(self) -> None:
        count = max((study.plan.variants for study in self.studies.studies), default=1)
        values = tuple(f"Вариант {index}" for index in range(1, count + 1))
        self.variant_combo.configure(values=values)
        if self.variant.get() not in values:
            self.variant.set(values[0])

    def draw(self) -> None:
        cv = self.plot
        cv.delete("all")
        c = self.colors
        n = len(self.studies.studies)
        if n == 0:
            return
        reference = self.studies.studies[0]
        problems = [f"{study.label}: {reference.compatible_with(study)[1]}"
                    for study in self.studies.studies[1:] if not reference.compatible_with(study)[0]]
        self.note.set(("Не сопоставимо напрямую: " + "; ".join(problems)) if problems
                      else f"Вариант {int(self.variant.get().split()[-1])}; среднее только по повторам, без смешивания разных значений параметра.")
        metric = SERIES_FIELDS[self.metric.get()]
        variant = int(self.variant.get().split()[-1])
        curves = [(study, study.curve(metric, variant)) for study in self.studies.studies]
        values = [value for _, curve in curves for _, value in curve]
        if not values:
            cv.create_text(30, 40, anchor="nw", text="Нет точек для этого показателя", fill=c.muted)
            return
        w, h = max(240, cv.winfo_width()), max(180, cv.winfo_height())
        x0, y0, x1, y1 = 65, 22, w-22, h-62
        low, high = min(values), max(values)
        if low == high:
            high += 1
        xmax = max(1, max(step for _, curve in curves for step, _ in curve))
        for t in range(5):
            y = y0 + (y1-y0)*t/4
            cv.create_line(x0, y, x1, y, fill=c.border, dash=(2, 5))
        legend_row = 0
        for i, (study, curve) in enumerate(curves):
            if not curve:
                continue
            color = LINE_COLORS[i % len(LINE_COLORS)]
            coords = []
            for step, value in curve:
                coords.extend((x0+step/xmax*(x1-x0), y1-(value-low)/(high-low)*(y1-y0)))
            if len(coords) >= 4:
                cv.create_line(*coords, fill=color, width=2)
            elif coords:
                cv.create_oval(coords[0]-2, coords[1]-2, coords[0]+2, coords[1]+2, fill=color)
            value = study.plan.values()[variant - 1]
            cv.create_text(x0+10, y1+18+16*legend_row, anchor="w",
                           text=f"{i+1}: {study.label} · V{variant}={value:g}", fill=color)
            legend_row += 1
        cv.create_text(3, y0, anchor="w", text=f"{high:.2f}", fill=c.muted)
        cv.create_text(3, y1, anchor="w", text=f"{low:.2f}", fill=c.muted)
        cv.create_text(x1, y1+13, anchor="e", text=f"{xmax} шагов", fill=c.muted)

    def export(self) -> None:
        if not self.studies.studies:
            return
        path = filedialog.asksaveasfilename(parent=self.window, defaultextension=".csv",
                                            filetypes=[("CSV", "*.csv")])
        if path:
            try:
                self.studies.export_csv(path, SERIES_FIELDS[self.metric.get()])
            except OSError as exc:
                messagebox.showerror("Экспорт", str(exc), parent=self.window)
