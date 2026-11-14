"""Tkinter front-end for reproducible parameter studies.

The simulation runs incrementally using Tk.after(), never in a blocking loop.
Results belong to the study window and do not modify the active game world.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import statistics
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

from .chrome import dark_titlebar
from .experiment_suite import ExperimentPlan, ExperimentSession, PARAMETERS
from .skins import skin
from .world import Ecosystem


RESULT_FIELDS = {
    "Конечная численность": "final_population",
    "Средняя численность": "mean_population",
    "Максимум численности": "peak_population",
    "Рождения": "total_births",
    "Смерти": "total_deaths",
    "Растения в конце": "final_plants",
    "Травоядные в конце": "final_herbivores",
    "Хищники в конце": "final_predators",
}
SERIES_FIELDS = {
    "Численность": "population",
    "Растения": "plants",
    "Травоядные": "herbivores",
    "Хищники": "predators",
    "Питательные вещества": "soil",
    "Влажность": "moisture",
    "Энергия животных": "mean_energy",
}
LINE_COLORS = ("#52D5DC", "#C19BFC", "#EBAE65", "#8FDAA0", "#F48AA5", "#79A9F1",
               "#E6D38C", "#A6AEBB", "#7CD4B2", "#FA9B72", "#A4B7FF", "#F0B9DD")


def series_for_chart(session: ExperimentSession, metric: str) -> dict[int, list[tuple[int, float]]]:
    """Average repetitions at each recorded step; missing values stay missing."""
    if metric not in SERIES_FIELDS.values():
        raise ValueError("Неизвестная серия")
    data: dict[tuple[int, int], list[float]] = defaultdict(list)
    for record in session.series:
        value = record[metric]
        if isinstance(value, (int, float)):
            data[(record["variant"], record["step"])].append(float(value))
    curves: dict[int, list[tuple[int, float]]] = {}
    for (variant, step), values in data.items():
        curves.setdefault(variant, []).append((step, statistics.mean(values)))
    for curve in curves.values():
        curve.sort(key=lambda pair: pair[0])
    return curves


class ExperimentStudio:
    """Non-modal experiment designer and result viewer."""

    def __init__(self, app: Any):
        self.app = app
        self.colors = skin(app.skin_var.get())
        c = self.colors
        self.window = tk.Toplevel(app.root)
        self.window.title("Life — Experiment Lab")
        self.window.configure(bg=c.base)
        self.window.geometry("1250x840")
        self.window.minsize(990, 700)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.window.after(120, lambda:dark_titlebar(self.window, c.base) if self.window.winfo_exists() else None)
        self.session: ExperimentSession | None = None
        self.paused = False
        self._pump_id: str | None = None
        self._last_result_count = -1
        self._shown_progress = -1
        current = app.world
        self.mode = current.mode
        choices = [p for p in PARAMETERS.values() if p.mode == self.mode]
        initial = "density" if self.mode == "classic" else "rain"
        self._choice_by_title = {p.title: p.key for p in choices}
        self._choice_by_key = {p.key: p.title for p in choices}
        self.parameter = tk.StringVar(value=self._choice_by_key[initial])
        self.start_mode = tk.StringVar(value="Новый мир")
        self.minimum = tk.StringVar(value="0.12" if initial == "density" else "0.2")
        self.maximum = tk.StringVar(value="0.36" if initial == "density" else "0.9")
        self.variants = tk.StringVar(value="4")
        self.repetitions = tk.StringVar(value="2")
        self.generations = tk.StringVar(value="120")
        self.sample_every = tk.StringVar(value="10")
        self.seed = tk.StringVar(value="42")
        self.width = tk.StringVar(value=str(min(80, current.width)))
        self.height = tk.StringVar(value=str(min(60, current.height)))
        self.stop_extinction = tk.BooleanVar(value=False)
        self.result_metric = tk.StringVar(value="Конечная численность")
        self.chart_metric = tk.StringVar(value="Численность")
        self.status = tk.StringVar(value="Настройте серию, затем нажмите «Запустить». Основной мир не изменится.")
        self.progress_text = tk.StringVar(value="0 / 0")
        self.progress_value = tk.DoubleVar(value=0)

        header = tk.Frame(self.window, bg=c.panel, padx=22, pady=15)
        header.pack(fill="x")
        tk.Label(header, text="EXPERIMENT LAB", fg=c.accent, bg=c.panel,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        tk.Label(header, text="Серии экспериментов и сравнение параметров", fg=c.text, bg=c.panel,
                 font=("Segoe UI", 19, "bold")).pack(anchor="w", pady=(4, 2))
        tk.Label(header, text=f"Режим: {'клеточный автомат' if self.mode == 'classic' else 'экосистема'}  ·  одинаковые seed по вариантам  ·  исходный мир защищён",
                 fg=c.muted, bg=c.panel, font=("Segoe UI", 9)).pack(anchor="w")

        content = tk.Frame(self.window, bg=c.base, padx=15, pady=14)
        content.pack(fill="both", expand=True)
        left = tk.Frame(content, bg=c.panel, padx=17, pady=14, width=300)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)
        right = tk.Frame(content, bg=c.base)
        right.pack(side="left", fill="both", expand=True)

        self._subheading(left, "ПЛАН ОПЫТА")
        self._field_label(left, "Изменяемый параметр")
        self.parameter_box = ttk.Combobox(left, textvariable=self.parameter,
            values=list(self._choice_by_title), state="readonly", style="Lab.TCombobox")
        self.parameter_box.pack(fill="x", pady=(0, 9))
        self.parameter_box.bind("<<ComboboxSelected>>", self._param_changed)
        self._field_label(left, "Исходное состояние")
        self.source_picker = ttk.Combobox(left, textvariable=self.start_mode,
            values=("Новый мир", "Копия текущего"), state="readonly", style="Lab.TCombobox")
        self.source_picker.pack(fill="x", pady=(0, 3))
        tk.Label(left, text="Варианты: новый мир / копия текущего",
                 fg=c.muted, bg=c.panel, font=("Segoe UI", 8)).pack(anchor="w", pady=(0, 12))
        self.range_box = tk.Frame(left, bg=c.panel)
        self.range_box.pack(fill="x")
        self._inline(self.range_box, "От", self.minimum, "До", self.maximum)
        self._inline(left, "Вариантов", self.variants, "Повторов", self.repetitions)
        self._inline(left, "Поколений", self.generations, "Каждые N", self.sample_every)
        self._field_label(left, "Базовый seed (один на повтор)")
        self._entry(left, self.seed).pack(fill="x", pady=(0, 9))
        self._inline(left, "Ширина", self.width, "Высота", self.height)
        check=tk.Checkbutton(left, text="Завершать опыт при исчезновении животных",
            variable=self.stop_extinction, fg=c.text, bg=c.panel, selectcolor=c.surface,
            activebackground=c.panel, activeforeground=c.text, highlightthickness=0,
            font=("Segoe UI", 9), anchor="w", wraplength=265, justify="left")
        check.pack(fill="x", pady=(6, 10))
        self.run_button = self._button(left,"▶  ЗАПУСТИТЬ СЕРИЮ",self.start,primary=True)
        self.run_button.pack(fill="x", pady=(3, 7))
        commands=tk.Frame(left,bg=c.panel)
        commands.pack(fill="x")
        self.pause_button=self._button(commands,"ПАУЗА",self.toggle_pause)
        self.pause_button.pack(side="left",expand=True,fill="x",padx=(0,5))
        self.stop_button=self._button(commands,"ОТМЕНИТЬ",self.cancel)
        self.stop_button.pack(side="left",expand=True,fill="x")
        self.pause_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")

        top=tk.Frame(right,bg=c.panel,padx=15,pady=12)
        top.pack(fill="x",pady=(0,10))
        self._subheading(top,"ХОД ЭКСПЕРИМЕНТА")
        tk.Label(top,textvariable=self.status,fg=c.text,bg=c.panel,anchor="w",
                 font=("Segoe UI",9),wraplength=590,justify="left").pack(fill="x",pady=(0,9))
        ttk.Style(self.window).configure("Suite.Horizontal.TProgressbar",troughcolor=c.surface,
                                          background=c.accent,bordercolor=c.border)
        ttk.Progressbar(top,variable=self.progress_value,maximum=100,
                        style="Suite.Horizontal.TProgressbar").pack(fill="x")
        tk.Label(top,textvariable=self.progress_text,fg=c.muted,bg=c.panel,
                 font=("Segoe UI",9)).pack(anchor="e",pady=(5,0))

        results=tk.Frame(right,bg=c.panel,padx=13,pady=11)
        results.pack(fill="x",pady=(0,10))
        choices=tk.Frame(results,bg=c.panel)
        choices.pack(fill="x",pady=(0,8))
        self._subheading(choices,"СРАВНЕНИЕ ВАРИАНТОВ")
        ttk.Combobox(choices,textvariable=self.result_metric,values=list(RESULT_FIELDS),
                     state="readonly",style="Lab.TCombobox",width=24).pack(side="right")
        self.result_metric.trace_add("write",lambda *_:self._update_results())
        self.table=ttk.Treeview(results,columns=("variant","value","n","mean","std","delta"),
                                show="headings",height=6,selectmode="browse")
        column_data=(('variant','№',50),('value','Параметр',110),('n','N',45),
                     ('mean','Среднее',110),('std','Разброс σ',110),('delta','Δ к базе',110))
        for key,title,width in column_data:
            self.table.heading(key,text=title)
            self.table.column(key,width=width,minwidth=36,stretch=True,anchor="center")
        self.table.pack(fill="x")
        self.table.configure(style="Lab.Treeview")

        graph=tk.Frame(right,bg=c.panel,padx=12,pady=9)
        graph.pack(fill="both",expand=True)
        graph_bar=tk.Frame(graph,bg=c.panel)
        graph_bar.pack(fill="x")
        self._subheading(graph_bar,"ДИНАМИКА ПО ПОКОЛЕНИЯМ")
        self.series_picker=ttk.Combobox(graph_bar,textvariable=self.chart_metric,
                                        values=list(SERIES_FIELDS),state="readonly",
                                        style="Lab.TCombobox",width=22)
        self.series_picker.pack(side="right")
        self.chart_metric.trace_add("write",lambda *_:self._draw_chart())
        self.chart=tk.Canvas(graph,height=218,bg=c.base,highlightthickness=0)
        self.chart.pack(fill="both",expand=True,pady=(6,0))
        self.chart.bind("<Configure>",lambda _e:self._draw_chart())

        footer=tk.Frame(self.window,bg=c.panel,padx=16,pady=11)
        footer.pack(fill="x",side="bottom",before=content)
        for label, callback in (("ИТОГИ CSV",self.export_summary),("РЯДЫ CSV",self.export_series),
                                ("ПОЛНЫЙ JSON",self.export_json),("СРАВНИТЬ JSON",self.open_archives),("ОТЧЁТ MD",self.export_report)):
            self._button(footer,label,callback).pack(side="left",padx=(0,9))
        tk.Label(footer,text="Сравнение относительное: первый вариант — базовый; σ — выборочное отклонение.",
                 fg=c.muted,bg=c.panel,font=("Segoe UI",8)).pack(side="right")
        style=ttk.Style(self.window)
        style.configure("Lab.Treeview",background=c.surface,fieldbackground=c.surface,
                        foreground=c.text,rowheight=25,borderwidth=0)
        style.configure("Lab.Treeview.Heading",background=c.panel,foreground=c.text,
                        font=("Segoe UI",9,"bold"))
        style.map("Lab.Treeview",background=[("selected",c.border)],foreground=[("selected",c.text)])
        self._param_changed()
        self.window.bind("<Escape>",lambda _event:self.close())
        self._draw_chart()

    def _button(self,parent:tk.Misc,label:str,command:Any,primary:bool=False)->tk.Button:
        c=self.colors
        return tk.Button(parent,text=label,command=command,bg=c.accent if primary else c.surface,
                        fg=c.base if primary else c.text,activebackground=c.secondary,
                        activeforeground=c.base if primary else c.text,
                        relief="flat",borderwidth=0,padx=11,pady=9,cursor="hand2",
                        font=("Segoe UI",9,"bold"))

    def _entry(self,parent:tk.Misc,variable:tk.StringVar)->tk.Entry:
        c=self.colors
        return tk.Entry(parent,textvariable=variable,bg=c.surface,fg=c.text,
                        insertbackground=c.accent,relief="flat",highlightthickness=1,
                        highlightbackground=c.border,highlightcolor=c.accent,
                        font=("Segoe UI",10))

    def _field_label(self,parent:tk.Misc,text:str)->None:
        tk.Label(parent,text=text,fg=self.colors.muted,bg=self.colors.panel,
                 font=("Segoe UI",9),anchor="w").pack(fill="x",pady=(4,5))

    def _subheading(self,parent:tk.Misc,text:str)->None:
        tk.Label(parent,text=text,fg=self.colors.accent,bg=self.colors.panel,
                 font=("Segoe UI",10,"bold"),anchor="w").pack(anchor="w",pady=(0,9))

    def _inline(self,parent:tk.Misc,left:str,a:tk.StringVar,right:str,b:tk.StringVar)->None:
        c=self.colors
        line=tk.Frame(parent,bg=c.panel)
        line.pack(fill="x",pady=(0,9))
        for idx,(name,value) in enumerate(((left,a),(right,b))):
            side=tk.Frame(line,bg=c.panel)
            side.pack(side="left",fill="x",expand=True,padx=(0,8) if idx==0 else (4,0))
            tk.Label(side,text=name,fg=c.muted,bg=c.panel,font=("Segoe UI",9),
                     anchor="w").pack(fill="x",pady=(0,4))
            self._entry(side,value).pack(fill="x")

    def _param_changed(self,_event:Any=None)->None:
        name=self._choice_by_title[self.parameter.get()]
        spec=PARAMETERS[name]
        if spec.integer:
            self.minimum.set("90")
            self.maximum.set("360")
        elif name in ("density","plants"):
            self.minimum.set("0.10")
            self.maximum.set("0.35")
        elif name == "herbivores":
            self.minimum.set("0.01")
            self.maximum.set("0.05")
        elif name == "predators":
            self.minimum.set("0.001")
            self.maximum.set("0.02")
        elif name == "moisture_flow":
            self.minimum.set("0.05")
            self.maximum.set("0.35")
        elif name == "mineral_flow":
            self.minimum.set("0.01")
            self.maximum.set("0.12")
        else:
            self.minimum.set("0.2")
            self.maximum.set("0.8")
        if not spec.setting:
            self.start_mode.set("Новый мир")
            self.source_picker.configure(state="disabled")
        else:
            self.source_picker.configure(state="readonly")

    def _plan(self)->ExperimentPlan:
        world=self.app.world
        ec=world.ecology.as_dict() if isinstance(world,Ecosystem) else {}
        return ExperimentPlan(
            mode=self.mode,parameter=self._choice_by_title[self.parameter.get()],
            minimum=float(self.minimum.get().strip().replace(",",".")),
            maximum=float(self.maximum.get().strip().replace(",",".")),
            variants=int(self.variants.get()),repetitions=int(self.repetitions.get()),
            steps=int(self.generations.get()),sample_every=int(self.sample_every.get()),
            seed=int(self.seed.get()),start_mode="snapshot" if self.start_mode.get()=="Копия текущего" else "fresh",
            width=int(self.width.get()),height=int(self.height.get()),
            wrap=world.wrap,rule=world.rule if world.mode=="classic" else "B3/S23",
            density=self.app.density_var.get(),ecology=ec,
            stop_on_extinction=self.stop_extinction.get())

    def start(self)->None:
        if self.session and not (self.session.complete or self.session.cancelled):
            messagebox.showinfo("Серия выполняется","Сначала завершите или отмените предыдущую серию.",parent=self.window)
            return
        try:
            plan=self._plan()
            source=self.app.world if plan.start_mode=="snapshot" else None
            self.session=ExperimentSession(plan,source)
        except (ValueError,TypeError,OverflowError) as exc:
            messagebox.showerror("Проверьте параметры",str(exc),parent=self.window)
            return
        self.paused=False
        self._shown_progress=-1
        self._last_result_count=-1
        self.pause_button.config(state="normal",text="ПАУЗА")
        self.stop_button.config(state="normal")
        self.run_button.config(state="disabled")
        self.table.delete(*self.table.get_children())
        self.status.set("Серия запущена. Основной мир не изменяется.")
        self.progress_value.set(0)
        self._queue()

    def _queue(self)->None:
        if self._pump_id is None and self.window.winfo_exists():
            self._pump_id=self.window.after(1,self._pump)

    def _pump(self)->None:
        self._pump_id=None
        session=self.session
        if session is None or session.complete or session.cancelled or self.paused:
            return
        # Small work slices keep the research window responsive, including on
        # maps where the ecosystem simulation is expensive.
        import time
        started=time.perf_counter()
        for _ in range(8):
            session.advance()
            if session.complete or time.perf_counter()-started>0.025:
                break
        progress=int(session.progress*1000)
        if progress != self._shown_progress:
            self._shown_progress=progress
            self.progress_value.set(progress/10)
            self.progress_text.set(f"{session.steps_done:,} выполнено / {session.total_steps:,} запланировано")
            self.status.set(f"Вариант {min(session.variant_index+1,session.plan.variants)} из {session.plan.variants}  ·  "
                            f"повтор {session.repetition_index+1} из {session.plan.repetitions}  ·  "
                            f"{progress/10:.1f}%")
        if len(session.summary)!=self._last_result_count:
            self._last_result_count=len(session.summary)
            self._update_results()
            self._draw_chart()
        if session.complete:
            self.pause_button.config(state="disabled")
            self.stop_button.config(state="disabled")
            self.run_button.config(state="normal")
            self.status.set(f"Готово: {len(session.summary)} независимых запусков. Результаты можно экспортировать.")
            self.progress_value.set(100)
            self._draw_chart()
        else:
            self._queue()

    def toggle_pause(self)->None:
        if self.session is None or self.session.complete:
            return
        self.paused=not self.paused
        self.pause_button.config(text="ПРОДОЛЖИТЬ" if self.paused else "ПАУЗА")
        if self.paused:
            self.status.set("Пауза: результаты уже завершённых опытов сохранены в окне.")
        else:
            self._queue()

    def cancel(self)->None:
        if self.session and not self.session.complete:
            self.session.cancel()
            self.status.set(f"Остановлено вручную: завершено {len(self.session.summary)} опытов.")
        if self._pump_id is not None:
            self.window.after_cancel(self._pump_id)
            self._pump_id=None
        self.run_button.config(state="normal")
        self.pause_button.config(state="disabled")
        self.stop_button.config(state="disabled")
        self._update_results()
        self._draw_chart()

    def _update_results(self)->None:
        if not hasattr(self,"table"):
            return
        self.table.delete(*self.table.get_children())
        if not self.session:
            return
        metric=RESULT_FIELDS[self.result_metric.get()]
        for row in self.session.comparison(metric):
            sigma="—" if row["std"] is None else f"{row['std']:,.2f}"
            delta="—" if row["paired_delta"] is None else f"{row['paired_delta']:+,.2f}"
            self.table.insert("","end",values=(row["variant"],f"{row['value']:g}",
                                                    row["n"],f"{row['mean']:,.2f}",sigma,delta))

    def _draw_chart(self)->None:
        if not hasattr(self,"chart"):
            return
        c=self.colors
        cv=self.chart
        cv.delete("all")
        w=max(200,cv.winfo_width());h=max(120,cv.winfo_height())
        xl,yt,xr,yb=62,24,w-24,h-42
        for i in range(5):
            y=yt+(yb-yt)*i/4
            cv.create_line(xl,y,xr,y,fill=c.border,dash=(2,5))
        if not self.session:
            cv.create_text(w/2,h/2,text="Здесь появятся траектории поколений",fill=c.muted)
            return
        curves=series_for_chart(self.session,SERIES_FIELDS[self.chart_metric.get()])
        values=[y for curve in curves.values() for _,y in curve]
        if not values:
            cv.create_text(w/2,h/2,text="Нет данных для выбранного показателя",fill=c.muted)
            return
        ymin=min(values);ymax=max(values)
        if ymax == ymin:
            ymax += 1
        for variant,curve in sorted(curves.items()):
            color=LINE_COLORS[(variant-1)%len(LINE_COLORS)]
            coords=[]
            for step,value in curve:
                x=xl+step/max(1,self.session.plan.steps)*(xr-xl)
                y=yb-(value-ymin)/(ymax-ymin)*(yb-yt)
                coords.extend((x,y))
            if len(coords)>2:
                cv.create_line(*coords,fill=color,width=2.1)
            elif coords:
                x,y=coords
                cv.create_oval(x-2,y-2,x+2,y+2,fill=color,outline="")
        cv.create_text(7,yt,anchor="w",text=f"{ymax:.2f}",fill=c.muted,font=("Segoe UI",8))
        cv.create_text(7,yb,anchor="w",text=f"{ymin:.2f}",fill=c.muted,font=("Segoe UI",8))
        cv.create_text(xr,yb+9,anchor="ne",text=f"{self.session.plan.steps} пок.",fill=c.muted,font=("Segoe UI",8))
        legend="   ".join(f"{n}: {value:g}" for n,value in enumerate(self.session.plan.values(),1))
        cv.create_text((xl+xr)/2,h-3,anchor="s",text=legend[:105],fill=c.muted,font=("Segoe UI",8))

    def _export(self,kind:str)->None:
        if self.session is None or not self.session.summary:
            messagebox.showinfo("Нет данных","Сначала выполните хотя бы один опыт.",parent=self.window)
            return
        extension={"json":".json","report":".md"}.get(kind,".csv")
        filename=filedialog.asksaveasfilename(parent=self.window,defaultextension=extension,
                    filetypes={"json":[("JSON","*.json")],"report":[("Markdown","*.md")]}.get(kind,[("CSV","*.csv")]),
                    initialfile="life-experiment-"+kind+extension)
        if filename:
            try:
                {"json":self.session.export_json,"summary":self.session.export_summary_csv,
                 "series":self.session.export_series_csv,"report":self.session.export_report}[kind](filename)
            except (OSError,ValueError) as exc:
                messagebox.showerror("Экспорт не удался",str(exc),parent=self.window)
            else:
                self.status.set(f"Сохранено: {Path(filename).name}")

    def export_summary(self)->None:
        self._export("summary")

    def export_series(self)->None:
        self._export("series")

    def export_json(self)->None:
        self._export("json")

    def export_report(self)->None:
        self._export("report")

    def open_archives(self)->None:
        from .archive_ui import ArchiveComparison
        ArchiveComparison(self.app)

    def close(self)->None:
        if self._pump_id is not None:
            self.window.after_cancel(self._pump_id)
            self._pump_id=None
        if self.session and not self.session.complete:
            self.session.cancel()
        self.window.destroy()
        if getattr(self.app,"suite_window",None) is self:
            self.app.suite_window=None
