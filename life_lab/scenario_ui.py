"""Non-modal scenario comparison studio. The active Life document is read-only."""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

from .chrome import dark_titlebar
from .scenarios import (INTERVENTION_KINDS, METRICS, PRESETS,
                        Intervention, ScenarioPlan, ScenarioSession)
from .skins import skin


class ScenarioWorkbench:
    """Run control and treatment incrementally so Tk never stalls for a whole study."""

    def __init__(self, app: Any):
        self.app = app
        c = skin(app.skin_var.get())
        self.colors = c
        self.window = tk.Toplevel(app.root)
        self.window.title("Life — Лаборатория сценариев")
        screen_w, screen_h = app.root.winfo_screenwidth(), app.root.winfo_screenheight()
        width = min(1100, max(820, screen_w - 40))
        height = min(745, max(670, screen_h - 90))
        self.window.geometry(f"{width}x{height}")
        self.window.minsize(820, min(670, height))
        self.window.configure(bg=c.base)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.after(80, lambda: dark_titlebar(self.window, c.base) if self.window.winfo_exists() else None)
        self.session: ScenarioSession | None = None
        self.paused = False
        self._pump_id: str | None = None
        self.pending: list[Intervention] = list(PRESETS["Засуха"])

        self.preset = tk.StringVar(value="Засуха")
        self.duration = tk.StringVar(value="120")
        self.interval = tk.StringVar(value="5")
        self.at = tk.StringVar(value="0")
        self.amount = tk.StringVar(value="-20")
        self.kind = tk.StringVar(value=INTERVENTION_KINDS["moisture"])
        self.metric = tk.StringVar(value="Численность")
        self.status = tk.StringVar(value="Сравнение двух копий текущего мира. Исходный мир останется без изменений.")
        self.progress = tk.StringVar(value="0 / 120")
        self.result = tk.StringVar(value="Результаты появятся после первого измерения")

        header = tk.Frame(self.window, bg=c.panel, padx=18, pady=13)
        header.pack(fill="x")
        tk.Label(header, text="SCENARIO STUDIO  /  LIFE 1.0", fg=c.accent, bg=c.panel,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(header, text="Исследование воздействий на экосистему", fg=c.text,
                 bg=c.panel, font=("Segoe UI", 17, "bold")).pack(anchor="w", pady=(3, 0))
        main = tk.Frame(self.window, bg=c.base, padx=12, pady=12)
        main.pack(fill="both", expand=True)
        left = tk.Frame(main, bg=c.panel, padx=12, pady=12, width=315)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)
        right = tk.Frame(main, bg=c.base)
        right.pack(side="left", fill="both", expand=True)

        self._heading(left, "ПЛАН ИССЛЕДОВАНИЯ")
        self._label(left, "Готовый сценарий")
        presets = ttk.Combobox(left, textvariable=self.preset, state="readonly",
                                values=list(PRESETS), style="Lab.TCombobox")
        presets.pack(fill="x", pady=(0, 7))
        presets.bind("<<ComboboxSelected>>", lambda _e: self.choose_preset())
        grid = tk.Frame(left, bg=c.panel)
        grid.pack(fill="x")
        self._cell(grid, "Поколений", self.duration, 0)
        self._cell(grid, "Период замера", self.interval, 1)
        self._heading(left, "ДОБАВИТЬ ВОЗДЕЙСТВИЕ")
        self._label(left, "Параметр")
        combo = ttk.Combobox(left, textvariable=self.kind, state="readonly",
                             values=list(INTERVENTION_KINDS.values()), style="Lab.TCombobox")
        combo.pack(fill="x", pady=(0, 7))
        row = tk.Frame(left, bg=c.panel)
        row.pack(fill="x")
        self._cell(row, "После поколения", self.at, 0)
        self._cell(row, "Значение", self.amount, 1)
        self._button(left, "+  ДОБАВИТЬ", self.add_event).pack(fill="x", pady=(3, 7))
        self._label(left, "Планируемые события")
        self.event_list = tk.Listbox(left, bg=c.base, fg=c.text, height=4, borderwidth=0,
                                     exportselection=False, highlightthickness=0,
                                     selectbackground=c.accent, selectforeground=c.base,
                                     font=("Segoe UI", 9))
        self.event_list.pack(fill="x", pady=(3, 4))
        self._button(left, "УДАЛИТЬ ВЫБРАННОЕ", self.remove_event).pack(fill="x", pady=(0, 8))
        row = tk.Frame(left, bg=c.panel)
        row.pack(fill="x", pady=(0, 3))
        self.start_btn = self._button(row, "▶ СТАРТ", self.start, primary=True)
        self.start_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.pause_btn = self._button(row, "ПАУЗА", self.toggle_pause)
        self.pause_btn.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.cancel_btn = self._button(row, "СТОП", self.cancel)
        self.cancel_btn.pack(side="left", fill="x", expand=True)
        self.pause_btn.config(state="disabled")
        self.cancel_btn.config(state="disabled")

        status = tk.Frame(right, bg=c.panel, padx=14, pady=11)
        status.pack(fill="x", pady=(0, 10))
        self._heading(status, "ХОД ОПЫТА")
        tk.Label(status, textvariable=self.status, fg=c.text, bg=c.panel,
                 font=("Segoe UI", 9), wraplength=580, anchor="w",
                 justify="left").pack(fill="x", pady=(0, 7))
        tk.Label(status, textvariable=self.progress, fg=c.accent, bg=c.panel,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(status, textvariable=self.result, fg=c.muted, bg=c.panel,
                 font=("Segoe UI", 9), justify="left", anchor="w",
                 wraplength=580).pack(fill="x", pady=(6, 0))
        chart_box = tk.Frame(right, bg=c.panel, padx=14, pady=11)
        chart_box.pack(fill="both", expand=True)
        self._heading(chart_box, "КОНТРОЛЬ  /  ВОЗДЕЙСТВИЕ")
        metric_select = ttk.Combobox(chart_box, textvariable=self.metric, state="readonly",
                                     values=list(METRICS), style="Lab.TCombobox")
        metric_select.pack(fill="x", pady=(0, 7))
        metric_select.bind("<<ComboboxSelected>>", lambda _e: self.draw_chart())
        self.chart = tk.Canvas(chart_box, bg=c.base, highlightthickness=0, height=205)
        self.chart.pack(fill="both", expand=True)
        self.chart.bind("<Configure>", lambda _e: self.draw_chart())
        legend = tk.Frame(chart_box, bg=c.panel)
        legend.pack(fill="x", pady=(7, 0))
        for color, label in ((c.accent, "●  Контроль"), (c.secondary, "●  Воздействие")):
            tk.Label(legend, text=label, fg=color, bg=c.panel,
                     font=("Segoe UI", 9, "bold")).pack(side="left", padx=(0, 15))
        tools = tk.Frame(right, bg=c.base)
        tools.pack(fill="x", pady=(9, 0))
        self._button(tools, "ЭКСПОРТ CSV", self.export_csv).pack(side="left", fill="x", expand=True, padx=(0, 6))
        self._button(tools, "ЭКСПОРТ JSON", self.export_json).pack(side="left", fill="x", expand=True, padx=(0, 6))
        self._button(tools, "АРХИВ ОПЫТА", self.export_bundle).pack(side="left", fill="x", expand=True)
        self._refresh_events()

    def _button(self, parent: tk.Misc, label: str, command: Any, primary: bool = False) -> tk.Button:
        c = self.colors
        return tk.Button(parent, text=label, command=command,
                         bg=c.accent if primary else c.surface,
                         fg=c.base if primary else c.text,
                         activebackground=c.secondary, activeforeground=c.base,
                         relief="flat", bd=0, padx=8, pady=8, cursor="hand2",
                         font=("Segoe UI", 9, "bold"))

    def _heading(self, parent: tk.Misc, label: str) -> None:
        tk.Label(parent, text=label, bg=self.colors.panel, fg=self.colors.text,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 8))

    def _label(self, parent: tk.Misc, label: str) -> None:
        tk.Label(parent, text=label, bg=self.colors.panel, fg=self.colors.muted,
                 font=("Segoe UI", 9)).pack(anchor="w", pady=(0, 4))

    def _cell(self, parent: tk.Frame, label: str, variable: tk.StringVar, column: int) -> None:
        frame = tk.Frame(parent, bg=self.colors.panel)
        frame.grid(row=0, column=column, padx=(0, 7), pady=(0, 9), sticky="ew")
        parent.columnconfigure(column, weight=1)
        self._label(frame, label)
        tk.Entry(frame, textvariable=variable, width=9,
                 bg=self.colors.surface, fg=self.colors.text, relief="flat", bd=0,
                 insertbackground=self.colors.text,
                 font=("Segoe UI", 10)).pack(fill="x", ipady=7)

    def _refresh_events(self) -> None:
        self.event_list.delete(0, tk.END)
        for event in self.pending:
            self.event_list.insert(tk.END,
                f"{event.at:>3} · {INTERVENTION_KINDS[event.kind]}: {event.value:g}")

    def choose_preset(self) -> None:
        self.pending = list(PRESETS[self.preset.get()])
        self._refresh_events()

    def add_event(self) -> None:
        try:
            kind = next(k for k, title in INTERVENTION_KINDS.items() if title == self.kind.get())
            event = Intervention(int(self.at.get()), kind, float(self.amount.get()))
            event.validate(int(self.duration.get()), ecology_enabled=self.app.world.ecology.enabled)
            if len(self.pending) >= 32:
                raise ValueError("Не более 32 воздействий")
            self.pending.append(event)
            self.pending.sort(key=lambda e: e.at)
            self._refresh_events()
        except (StopIteration, ValueError) as error:
            messagebox.showerror("Проверка воздействия", str(error), parent=self.window)

    def remove_event(self) -> None:
        selected = self.event_list.curselection()
        if selected:
            self.pending.pop(selected[0])
            self._refresh_events()

    def start(self) -> None:
        try:
            plan = ScenarioPlan(steps=int(self.duration.get()),
                                sample_every=int(self.interval.get()),
                                interventions=tuple(self.pending),
                                label=self.preset.get())
            new_session = ScenarioSession(self.app.world, plan)
        except (ValueError, TypeError) as error:
            messagebox.showerror("Ошибка плана", str(error), parent=self.window)
            return
        self.cancel()
        self.session = new_session
        self.paused = False
        self.start_btn.config(state="disabled")
        self.pause_btn.config(state="normal", text="ПАУЗА")
        self.cancel_btn.config(state="normal")
        self.status.set("Два мира запущены от одного состояния и с одинаковыми RNG.")
        self.draw_chart()
        self._pump_id = self.window.after(15, self._pump)

    def _pump(self) -> None:
        self._pump_id = None
        if self.session is None or self.paused or not self.window.winfo_exists():
            return
        try:
            for _ in range(3):
                if not self.session.advance():
                    break
        except Exception as error:
            self.status.set(f"Эксперимент остановлен: {error}")
            self.paused = True
            self.start_btn.config(state="normal")
            return
        self.progress.set(f"{self.session.completed} / {self.session.plan.steps}")
        summary = self.session.summary()
        self.result.set(f"Популяция: контроль {summary['control_population']}  ·  "
                        f"воздействие {summary['scenario_population']}  ·  "
                        f"разница {summary['population_difference']:+}")
        self.draw_chart()
        if self.session.done:
            self.status.set(f"Завершено. Применено воздействий: {len(self.session.fired)}. "
                            "Оба мира доступны для экспорта, исходный не изменён.")
            self.start_btn.config(state="normal")
            self.pause_btn.config(state="disabled")
            self.cancel_btn.config(state="disabled")
        else:
            self._pump_id = self.window.after(12, self._pump)

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.pause_btn.config(text="ПРОДОЛЖИТЬ" if self.paused else "ПАУЗА")
        if not self.paused and self.session is not None and not self.session.done:
            self._pump_id = self.window.after(12, self._pump)

    def cancel(self) -> None:
        if self._pump_id is not None:
            try:
                self.window.after_cancel(self._pump_id)
            except tk.TclError:
                pass
            self._pump_id = None
        self.paused = False
        self.start_btn.config(state="normal")
        self.pause_btn.config(state="disabled")
        self.cancel_btn.config(state="disabled")

    def draw_chart(self) -> None:
        if not hasattr(self, "chart") or not self.chart.winfo_exists():
            return
        c = self.colors
        self.chart.delete("all")
        width = max(100, self.chart.winfo_width())
        height = max(100, self.chart.winfo_height())
        left, right, top, bottom = 45, width - 14, 19, height - 28
        self.chart.create_line(left, top, left, bottom, right, bottom, fill=c.border)
        if not self.session:
            self.chart.create_text(width/2, height/2, text="Запустите сценарий для сравнения",
                                   fill=c.muted, font=("Segoe UI", 10))
            return
        data = self.session.curves(METRICS[self.metric.get()])
        vals = [val for lines in data.values() for _, val in lines]
        if not vals:
            self.chart.create_text(width/2, height/2, text="Нет данных для выбранного показателя",
                                   fill=c.muted)
            return
        low = min(0, min(vals)); high = max(vals)
        if high <= low:
            high = low + 1
        for fraction in (0, 0.25, 0.5, 0.75, 1):
            y = bottom - (bottom - top) * fraction
            self.chart.create_line(left, y, right, y, fill=c.border, dash=(2, 5))
            self.chart.create_text(left-5, y, anchor="e", text=f"{low+(high-low)*fraction:.1f}",
                                   fill=c.muted, font=("Segoe UI", 8))
        self.chart.create_text(right, bottom+8, anchor="n", text=f"{self.session.plan.steps} поколений",
                               fill=c.muted, font=("Segoe UI", 8))
        for intervention in self.session.plan.interventions:
            x = left + (right-left) * intervention.at / max(1,self.session.plan.steps)
            self.chart.create_line(x, top, x, bottom, fill=c.warning, dash=(2,5))
        for branch, color in (("control", c.accent), ("scenario", c.secondary)):
            points = []
            for step, value in data[branch]:
                x = left + (right-left) * step / max(1,self.session.plan.steps)
                y = bottom - (bottom-top) * (value-low) / (high-low)
                points.extend([x, y])
            if len(points) >= 4:
                self.chart.create_line(*points, fill=color, width=2, smooth=False)
            elif points:
                self.chart.create_oval(points[0]-3, points[1]-3, points[0]+3, points[1]+3,
                                       fill=color, outline="")

    def export_csv(self) -> None:
        if not self.session:
            messagebox.showinfo("Нет данных", "Запустите сценарий", parent=self.window)
            return
        path = filedialog.asksaveasfilename(parent=self.window, defaultextension=".csv",
                 filetypes=[("CSV", "*.csv")], title="Экспорт измерений двух миров")
        if path:
            try:
                self.session.export_csv(path)
                self.status.set(f"CSV сохранён: {path}")
            except OSError as error:
                messagebox.showerror("Ошибка экспорта", str(error), parent=self.window)

    def export_json(self) -> None:
        if not self.session:
            messagebox.showinfo("Нет данных", "Запустите сценарий", parent=self.window)
            return
        path = filedialog.asksaveasfilename(parent=self.window, defaultextension=".json",
                 filetypes=[("JSON", "*.json")], title="Экспорт плана и результатов")
        if path:
            try:
                self.session.export_json(path)
                self.status.set(f"JSON сохранён: {path}")
            except OSError as error:
                messagebox.showerror("Ошибка экспорта", str(error), parent=self.window)

    def close(self) -> None:
        self.cancel()
        self.window.destroy()

    def export_bundle(self) -> None:
        if not self.session:
            messagebox.showinfo("Нет данных", "Запустите сценарий", parent=self.window)
            return
        path = filedialog.asksaveasfilename(parent=self.window, defaultextension=".zip",
                 filetypes=[("Архив исследования", "*.zip")], title="Сохранить весь опыт и обе ветви")
        if path:
            try:
                self.session.export_bundle(path)
                self.status.set(f"Сохранён архив исследования: {path}")
            except (OSError, ValueError) as error:
                messagebox.showerror("Ошибка экспорта", str(error), parent=self.window)
