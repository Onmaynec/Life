"""Tabbed workbench: live metrics, 3D spatial plots, environment controls and pattern catalogue."""
from __future__ import annotations

import math
import time
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from .ecology import EcologySettings, climate_values
from .instruments import MEASURES, measure, spatial_field
from .pattern_tools import transform, shift
from .patterns import PATTERNS, pattern_array, place_pattern
from .skins import skin
from .world import CellularAutomaton, Ecosystem

METRIC_TITLES: dict[str, str] = {
    "population":"Численность", "occupancy":"Заполнение, %", "plants":"Растения",
    "herbivores":"Травоядные", "predators":"Хищники", "births":"Рождения за шаг",
    "deaths":"Смерти за шаг", "net_growth":"Прирост", "diversity":"Число видов",
    "entropy":"Энтропия Шеннона", "mean_neighbours":"Среднее число соседей",
    "lonely":"Одинокие клетки", "crowded":"Перенаселённые клетки",
    "clusters":"Кластеры (4 связи)", "largest_cluster":"Крупнейший кластер",
    "edge_contacts":"Клетки на границе", "centroid_x":"Центр X", "centroid_y":"Центр Y",
    "dispersion":"Разброс от центра", "soil":"Минералы", "moisture":"Влага",
    "energy":"Энергия животных", "herbivore_energy":"Энергия травоядных",
    "predator_energy":"Энергия хищников", "mean_age":"Средний возраст",
    "max_age":"Максимальный возраст", "mean_metabolism":"Метаболизм",
    "mean_sight":"Зрение", "mean_fertility":"Плодовитость",
    "sunlight":"Освещённость", "temperature":"Температурный индекс",
    "plant_biomass":"Биомасса растений", "animal_biomass":"Энергетическая биомасса",
    "predator_prey_ratio":"Отношение хищник / травоядное",
    "birth_rate":"Рождения, %", "death_rate":"Смерти, %",
    "organic":"Органика", "toxins":"Токсины", "seed_bank":"Банк семян",
    "altitude":"Средняя высота",
}
LABEL_TO_METRIC = {label:key for key,label in METRIC_TITLES.items()}
GRAPH_FIELDS = ["Плотность", "Типы клеток", "Соседи", "Энергия", "Минералы", "Влага",
                "Возраст", "Метаболизм", "Зрение", "Плодовитость",
                "Семена", "Органика", "Токсины", "Рельеф"]


def formatted(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.2f}"
    return f"{value:,}".replace(",", " ") if isinstance(value,int) else str(value)


class ResearchWorkspace:
    """Controller for the right-hand research tabs, all running on Tk's main thread."""
    def __init__(self, app: Any, host: tk.Frame):
        self.app = app
        self.host = host
        self.colors = skin(app.skin_var.get())
        self.pages: dict[str, tk.Frame] = {}
        self.tab_buttons: dict[str, tk.Button] = {}
        self.current = "Обзор"
        self.last_refresh = -1
        self.baseline: dict[str,float | int | None] | None = None
        self.pattern_selected = "Глайдер"
        self.pattern_image = pattern_array("Глайдер")
        self.alert_extinction = tk.BooleanVar(value=False)
        self.animate_graphs = tk.BooleanVar(value=True)
        self.field_name = tk.StringVar(value="Плотность")
        self.metric_name = tk.StringVar(value="Численность")
        self.tabs = tk.Frame(host, bg=self.colors.panel)
        self.tabs.pack(fill="x", pady=(0, 11))
        for name in ("Обзор","Анализ","3D","Среда","Каталог"):
            button = app._button(self.tabs, name.upper(), lambda n=name:self.show(n))
            button.configure(font=("Segoe UI",8,"bold"),padx=4,pady=8)
            button.pack(side="left",expand=True,fill="x",padx=1)
            self.tab_buttons[name]=button
            self.pages[name]=tk.Frame(host,bg=self.colors.panel)
        self.overview = self.pages["Обзор"]
        self.overview.pack(fill="both")
        self._build_analysis()
        self._build_3d()
        self._build_environment()
        self._build_library()
        self._select_tab()

    def _heading(self, parent: tk.Frame, title: str, description: str = "") -> None:
        self.app._label(parent,title,12,self.colors.text,True).pack(anchor="w",pady=(2,7))
        if description:
            tk.Label(parent,text=description,bg=self.colors.panel,fg=self.colors.muted,
                     font=("Segoe UI",9),anchor="w",wraplength=240,justify="left").pack(anchor="w",pady=(0,13))

    def show(self, name: str) -> None:
        if name not in self.pages:
            return
        self.pages[self.current].pack_forget()
        self.current=name
        self.pages[name].pack(fill="both")
        self._select_tab()
        self.host.update_idletasks()
        self.app.inspector_scroll.yview_moveto(0)
        self.refresh(force=True)

    def _select_tab(self) -> None:
        for name, button in self.tab_buttons.items():
            active=name==self.current
            button.config(bg=self.colors.accent if active else self.colors.surface,
                          fg=self.colors.base if active else self.colors.muted)

    def _build_analysis(self) -> None:
        p=self.pages["Анализ"]
        self._heading(p,"АНАЛИТИКА","Реальные измерения поля. Сравнивайте поколения и сохраняйте данные для исследования.")
        self.metric_picker=ttk.Combobox(p,textvariable=self.metric_name,
                          values=list(METRIC_TITLES.values()),state="readonly",style="Lab.TCombobox")
        self.metric_picker.pack(fill="x",pady=(0,9))
        self.metric_picker.bind("<<ComboboxSelected>>",lambda _e:self.refresh(force=True))
        self.metric_chart=tk.Canvas(p,height=166,bg=self.colors.base,highlightthickness=0)
        self.metric_chart.pack(fill="x",pady=(0,12))
        self.metric_chart.bind("<Configure>",lambda _e:self._draw_metric_chart())
        self.metric_value=tk.Label(p,text="—",bg=self.colors.panel,fg=self.colors.accent,
                                   font=("Segoe UI",21,"bold"),anchor="w")
        self.metric_value.pack(fill="x")
        self.metric_description=self.app._label(p,"",9,self.colors.muted)
        self.metric_description.pack(anchor="w",pady=(0,13))
        self.metrics_grid=tk.Frame(p,bg=self.colors.panel)
        self.metrics_grid.pack(fill="x",pady=(0,10))
        self.metric_rows={}
        for name in ("occupancy","entropy","clusters","largest_cluster","mean_neighbours",
                     "soil","moisture","mean_age","energy","sunlight","temperature"):
            row=tk.Frame(self.metrics_grid,bg=self.colors.panel)
            row.pack(fill="x",pady=3)
            self.app._label(row,METRIC_TITLES[name],9).pack(side="left")
            label=self.app._label(row,"—",9,self.colors.text,True)
            label.pack(side="right")
            self.metric_rows[name]=label
        self.compare_label=tk.Label(p,text="Зафиксируйте контрольную точку для сравнения.",
                                    bg=self.colors.surface,fg=self.colors.muted,justify="left",
                                    wraplength=230,padx=10,pady=12,font=("Segoe UI",9))
        self.compare_label.pack(fill="x",pady=(0,9))
        self.app._button(p,"ЗАФИКСИРОВАТЬ ПОКОЛЕНИЕ",self.take_baseline,accent=True).pack(fill="x",pady=2)
        self.app._button(p,"СЕРИИ ЭКСПЕРИМЕНТОВ  /  Ctrl+E",self.app.open_experiment_studio,accent=True).pack(fill="x",pady=(3,9))
        self.app._button(p,"ЭКСПОРТ ИЗМЕРЕНИЙ CSV",self.export_csv).pack(fill="x",pady=2)
        self.app._button(p,"ОТЧЁТ ЭКСПЕРИМЕНТА MD",self.export_report).pack(fill="x",pady=2)
        self.pause_checkbox=tk.Checkbutton(p,text="Пауза при исчезновении видов",variable=self.alert_extinction,
                        bg=self.colors.panel,fg=self.colors.text,selectcolor=self.colors.surface,
                        activebackground=self.colors.panel,anchor="w",highlightthickness=0)
        self.pause_checkbox.pack(fill="x",pady=(12,2))

    def take_baseline(self) -> None:
        self.baseline=measure(self.app.world)
        self.refresh(force=True)
        self.app._show_hint(f"Контрольная точка: поколение {self.app.world.generation}")

    def export_csv(self) -> None:
        filename=filedialog.asksaveasfilename(defaultextension=".csv",filetypes=[("CSV","*.csv")],
                                               title="Экспорт исследовательских измерений")
        if filename:
            try:
                self.app.research_log.export_csv(filename)
                self.app._show_hint(f"Измерения сохранены: {Path(filename).name}")
            except OSError as error:
                messagebox.showerror("Ошибка экспорта",str(error))

    def export_report(self) -> None:
        filename=filedialog.asksaveasfilename(defaultextension=".md",filetypes=[("Markdown","*.md")],
                                               title="Экспорт отчёта")
        if not filename:
            return
        m=measure(self.app.world)
        lines=["# Life — отчёт исследования", "",f"Режим: {self.app.world.mode}",
               f"Поколение: {self.app.world.generation}",
               f"Размер: {self.app.world.width} × {self.app.world.height}","",
               "## Текущие измерения", "", "Показатель | Значение", "--- | ---"]
        for key in MEASURES:
            if key != "generation":
                lines.append(f"{METRIC_TITLES.get(key,key)} | {formatted(m.get(key))}")
        if self.baseline:
            lines.extend(["","## Контрольное поколение", "",
                          f"Поколение: {self.baseline['generation']}","",
                          "Показатель | Тогда | Сейчас | Разница", "--- | ---: | ---: | ---:"])
            for key in ("population","occupancy","entropy","soil","energy","mean_age"):
                before=self.baseline.get(key);now=m.get(key)
                delta=now-before if isinstance(before,(int,float)) and isinstance(now,(int,float)) else None
                lines.append(f"{METRIC_TITLES.get(key,key)} | {formatted(before)} | {formatted(now)} | {formatted(delta)}")
        lines.extend(["","Измерения получены из текущего состояния модели Life.",
                      "Это экспериментальная клеточная модель, а не биологический прогноз.",""])
        try:
            Path(filename).write_text("\n".join(lines),encoding="utf-8")
            self.app._show_hint(f"Отчёт сохранён: {Path(filename).name}")
        except OSError as error:
            messagebox.showerror("Ошибка экспорта",str(error))

    def _draw_metric_chart(self) -> None:
        cv=self.metric_chart
        cv.delete("all")
        w=max(180,cv.winfo_width());h=max(100,cv.winfo_height())
        xmin,ymin,xmax,ymax=30,20,w-14,h-30
        for i in range(5):
            y=ymin+(ymax-ymin)*i/4
            cv.create_line(xmin,y,xmax,y,fill=self.colors.border,dash=(2,4))
        key=LABEL_TO_METRIC.get(self.metric_name.get(),"population")
        series=[(int(row["generation"]),row[key]) for row in self.app.research_log.records[-250:]
                if isinstance(row.get(key),(int,float))]
        if len(series)<2:
            cv.create_text(w/2,h/2,text="Появится после нескольких шагов",fill=self.colors.muted)
            return
        values=np.array([v for _,v in series],dtype=float)
        a,b=float(np.min(values)),float(np.max(values))
        if b==a:
            b=a+1
        xfirst,xlast=series[0][0],series[-1][0]
        coords=[]
        for generation,value in series:
            x=xmin+(generation-xfirst)/max(1,xlast-xfirst)*(xmax-xmin)
            y=ymax-(value-a)/(b-a)*(ymax-ymin)
            coords.extend((x,y))
        cv.create_line(*coords,fill=self.colors.accent,width=2.5,smooth=False)
        cv.create_text(xmin,ymin-9,text=f"{b:.2f}",fill=self.colors.muted,anchor="w",font=("Segoe UI",8))
        cv.create_text(xmin,ymax+10,text=f"{a:.2f}",fill=self.colors.muted,anchor="w",font=("Segoe UI",8))
        cv.create_text(xmax,ymax+13,text=f"g {xlast}",fill=self.colors.muted,anchor="e",font=("Segoe UI",8))

    def _build_3d(self) -> None:
        p=self.pages["3D"]
        self._heading(p,"3D-ПРОСТРАНСТВО", "Высота столбцов соответствует значениям выбранного показателя. В полной студии можно вращать модель.")
        self.field_combo=ttk.Combobox(p,textvariable=self.field_name,values=GRAPH_FIELDS,
                                state="readonly",style="Lab.TCombobox")
        self.field_combo.pack(fill="x",pady=(0,10))
        self.field_combo.bind("<<ComboboxSelected>>",lambda _e:self.update_preview())
        self.preview_canvas=tk.Canvas(p,height=205,bg=self.colors.base,highlightthickness=0)
        self.preview_canvas.pack(fill="x",pady=(0,9))
        self.preview_canvas.bind("<Configure>",lambda _e:self.update_preview())
        self.preview_photo=None
        self.preview_label=self.app._label(p,"Средние значения по блокам · масштаб автоматически нормализован",8)
        self.preview_label.pack(fill="x",pady=(0,12))
        self.app._button(p,"ОТКРЫТЬ 3D-СТУДИЮ",self.open_3d,accent=True).pack(fill="x",pady=(0,8))
        self.app._label(p,"В студии доступны вращение, масштабирование, проекция поверхности и сохранение PNG.",9).pack(fill="x")

    def update_preview(self) -> None:
        if self.current != "3D":
            return
        cv=self.preview_canvas
        w=max(140,cv.winfo_width());h=max(130,cv.winfo_height())
        arr=spatial_field(self.app.world,self.field_name.get(),max_side=17)
        image=Image.new("RGB",(w,h),self.colors.base)
        draw=ImageDraw.Draw(image)
        ny,nx=arr.shape
        mx=float(arr.max()) if arr.size else 0.0
        step=min((w-35)/(nx+ny)*1.48,(h-60)/(nx+ny)*3.2,8.5)
        vert=step*0.52
        elevation=min(110,max(25,h*0.42))
        xcenter=w/2
        ybase=h-19
        accent=tuple(int(self.colors.accent[i:i+2],16) for i in (1,3,5))
        # Depth-first ordering of isometric blocks prevents foreground occlusion.
        for iy in range(ny):
            for ix in range(nx):
                v=float(arr[iy,ix])
                z=(v/max(1e-7,mx))*elevation
                x=xcenter+(ix-iy)*step
                y=ybase-(ix+iy)*vert
                x0,y0=x,y-z
                light=tuple(min(255,int(channel*(0.42+0.58*(v/max(1e-7,mx))))) for channel in accent)
                dark=tuple(int(c*0.40) for c in light)
                medium=tuple(int(c*0.66) for c in light)
                poly_top=[(x0,y0),(x0+step,y0+vert),(x0,y0+2*vert),(x0-step,y0+vert)]
                draw.polygon([(x0-step,y0+vert),(x0,y0+2*vert),(x0,y+2*vert),(x0-step,y+vert)],fill=dark)
                draw.polygon([(x0,y0+2*vert),(x0+step,y0+vert),(x0+step,y+vert),(x0,y+2*vert)],fill=medium)
                draw.polygon(poly_top,fill=light)
        self.preview_photo=ImageTk.PhotoImage(image)
        cv.delete("all")
        cv.create_image(0,0,anchor="nw",image=self.preview_photo)

    def open_3d(self) -> None:
        """Real Matplotlib mplot3d view; mouse rotation and PNG export."""
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.colors import LinearSegmentedColormap
        except ImportError:
            messagebox.showerror("3D не найден", "Установите зависимости командой: pip install -r requirements.txt")
            return
        window=tk.Toplevel(self.app.root)
        window.title("Life — 3D Spatial Research")
        window.configure(bg=self.colors.base)
        window.geometry("1100x760")
        window.minsize(760,540)
        from .chrome import dark_titlebar
        window.after(100,lambda:dark_titlebar(window,self.colors.base))
        top=tk.Frame(window,bg=self.colors.panel,padx=13,pady=9)
        top.pack(fill="x")
        field=tk.StringVar(value=self.field_name.get())
        style=tk.StringVar(value="Столбцы")
        self.app._label(top,"ПОЛЕ",9).pack(side="left",padx=(0,5))
        selector=ttk.Combobox(top,textvariable=field,values=GRAPH_FIELDS,state="readonly",width=15,style="Lab.TCombobox")
        selector.pack(side="left",padx=(0,13))
        chosen=ttk.Combobox(top,textvariable=style,values=("Столбцы","Поверхность","Сетка"),state="readonly",width=14,style="Lab.TCombobox")
        chosen.pack(side="left")
        figure=Figure(figsize=(9.8,6.2),dpi=105,facecolor=self.colors.base)
        ax=figure.add_subplot(111,projection="3d",computed_zorder=False)
        canvas=FigureCanvasTkAgg(figure,master=window)
        canvas.get_tk_widget().configure(bg=self.colors.base,highlightthickness=0)
        canvas.get_tk_widget().pack(fill="both",expand=True)
        tk.Label(window,text="ЛКМ на графике — вращение  ·  колесо — масштаб  ·  кнопка «Обновить данные» — новый снимок",
                 bg=self.colors.panel,fg=self.colors.muted,font=("Segoe UI",9),anchor="w",padx=14,pady=8).pack(fill="x",side="bottom")
        cmap=LinearSegmentedColormap.from_list("life",[self.colors.base,self.colors.border,self.colors.accent,"#FFF0BD"])
        def redraw(*_args:Any) -> None:
            data=spatial_field(self.app.world,field.get(),max_side=29)
            yy,xx=np.indices(data.shape)
            elevation, azimuth = ax.elev, ax.azim
            ax.clear()
            ax.set_facecolor(self.colors.base)
            ax.xaxis.pane.fill=False
            ax.yaxis.pane.fill=False
            ax.zaxis.pane.fill=False
            for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
                axis.line.set_color(self.colors.muted)
                axis.set_tick_params(colors=self.colors.muted,labelsize=8)
            if style.get()=="Столбцы":
                v=data.ravel()
                rgba=cmap(v/max(1e-8,float(v.max())))
                ax.bar3d(xx.ravel(),yy.ravel(),np.zeros(v.size),.89,.89,v,
                         color=rgba,shade=True,zsort="average",alpha=0.96)
            elif style.get()=="Поверхность":
                ax.plot_surface(xx,yy,data,cmap=cmap,linewidth=0,antialiased=True,alpha=0.97)
            else:
                ax.plot_wireframe(xx,yy,data,rstride=1,cstride=1,color=self.colors.accent,linewidth=.8)
            ax.set_title(f"{field.get()} · поколение {self.app.world.generation}",color=self.colors.text,pad=18)
            ax.set_xlabel("X",color=self.colors.muted)
            ax.set_ylabel("Y",color=self.colors.muted)
            ax.set_zlabel(field.get(),color=self.colors.muted)
            ax.view_init(elev=elevation,azim=azimuth)
            figure.tight_layout(pad=2)
            canvas.draw_idle()
        selector.bind("<<ComboboxSelected>>",redraw)
        chosen.bind("<<ComboboxSelected>>",redraw)
        self.app._button(top,"ОБНОВИТЬ ДАННЫЕ",redraw).pack(side="left",padx=(10,0))
        def reset_camera():
            ax.view_init(elev=33,azim=-52)
            canvas.draw_idle()
        self.app._button(top,"КАМЕРА",reset_camera).pack(side="left",padx=(7,0))
        def export() -> None:
            path=filedialog.asksaveasfilename(parent=window,defaultextension=".png",filetypes=[("PNG","*.png")])
            if path:
                figure.savefig(path,dpi=170,facecolor=self.colors.base)
        self.app._button(top,"СОХРАНИТЬ PNG",export).pack(side="right")
        redraw()

    def _build_environment(self) -> None:
        p=self.pages["Среда"]
        self._heading(p,"КЛИМАТ И ЖИЗНЕННЫЕ ЦИКЛЫ", "Расширенная экосистема работает отдельно от правил Конвея. Все изменения применяются к текущему миру.")
        self.eco_enabled=tk.BooleanVar(value=False)
        self._eco_toggle(p,"Включить динамику среды","enabled",self.eco_enabled,accent=True)
        self.climate_desc=tk.Label(p,text="",bg=self.colors.surface,fg=self.colors.text,
                                   font=("Segoe UI",9),padx=10,pady=10,wraplength=235,justify="left")
        self.climate_desc.pack(fill="x",pady=(8,13))
        self.eco_sliders={}
        for title,key,frm,to,res in (
            ("Осадки","rain",0,1,.05),
            ("Температурный фон","heat",0,1,.05),
            ("Распространение влаги","moisture_flow",0,1,.05),
            ("Перенос минералов","mineral_flow",0,.5,.025),
            ("Вероятность пожара","wildfire_chance",0,.02,.001),
            ("Длительность года","season_length",20,600,10),
        ):
            self.app._label(p,title,9).pack(anchor="w",pady=(4,0))
            scale=tk.Scale(p,from_=frm,to=to,resolution=res,orient="horizontal",showvalue=True,
                           bg=self.colors.panel,fg=self.colors.text,troughcolor=self.colors.surface,
                           highlightthickness=0,activebackground=self.colors.accent,
                           command=lambda value,k=key:self._set_ecology(k,value))
            scale.pack(fill="x",pady=(0,4))
            self.eco_sliders[key]=scale
        self.eco_toggles={}
        for key,title in (
            ("season_cycle","Смена сезонов"), ("solar_cycle","Суточное освещение"),
            ("storms","Дождевые циклы"),("drought","Засухи"),
            ("photosynthesis","Фотосинтез"),("erosion","Вымывание минералов"),
            ("decomposition","Разложение"),("aging","Старение"),
            ("competition","Конкуренция растений"),("dispersal","Рассеивание семян"),
            ("temperature_stress","Температурный стресс"),("disease","Эпидемии"),
            ("wildfire","Пожары"),
            ("seed_bank","Семенной банк"),("organic_cycle","Минерализация органики"),
            ("toxins","Токсичные отходы"),("toxin_diffusion","Диффузия токсинов"),
            ("water_springs","Родники"),("terrain_runoff","Сток с возвышенностей"),
            ("microhabitats","Локальный климат"),("plant_shade","Затенение растений"),
            ("pollination","Распространение пыльцы"),("grazing_fertilizer","Удобрение от травоядных"),
            ("herbivore_flee","Бегство от хищников"),("herd_defense","Защита стада"),
            ("pack_hunt","Групповая охота"),("territoriality","Территориальная конкуренция"),
            ("nocturnal_predators","Ночные хищники"),("winter_sleep","Зимняя экономия энергии"),
            ("juvenile_stage","Неполовозрелые животные"),("animal_thirst","Обезвоживание"),
            ("seasonal_breeding","Сезонное размножение"),("migrations","Переселение к воде"),
        ):
            variable=tk.BooleanVar(value=False)
            self.eco_toggles[key]=variable
            self._eco_toggle(p,title,key,variable)
        self.app._button(p,"ВОССТАНОВИТЬ СРЕДУ",self.reset_ecology).pack(fill="x",pady=(13,4))
        self.sync_ecology()

    def _eco_toggle(self,parent:tk.Frame,text:str,key:str,variable:tk.BooleanVar,accent:bool=False)->None:
        tk.Checkbutton(parent,text=text,variable=variable,
                      command=lambda:self._set_ecology(key,variable.get()),
                      bg=self.colors.panel,fg=self.colors.accent if accent else self.colors.text,
                      selectcolor=self.colors.surface,activeforeground=self.colors.accent,
                      activebackground=self.colors.panel,highlightthickness=0,anchor="w",
                      font=("Segoe UI",9,"bold") if accent else ("Segoe UI",9)).pack(fill="x",pady=(2,3))

    def _set_ecology(self,key:str,value:Any) -> None:
        if not isinstance(self.app.world,Ecosystem):
            return
        if key=="season_length":
            value=int(float(value))
        elif key in ("enabled","season_cycle","solar_cycle","storms","drought","photosynthesis",
                     "erosion","decomposition","aging","competition","dispersal","temperature_stress","disease","wildfire",
                     "seed_bank","organic_cycle","toxins","toxin_diffusion","water_springs","terrain_runoff",
                     "microhabitats","plant_shade","pollination","grazing_fertilizer","herbivore_flee",
                     "herd_defense","pack_hunt","territoriality","nocturnal_predators","winter_sleep",
                     "juvenile_stage","animal_thirst","seasonal_breeding","migrations"):
            value=bool(value)
        else:
            value=float(value)
        setattr(self.app.world.ecology,key,value)
        self.app._mark_dirty()
        self._show_climate()

    def reset_ecology(self) -> None:
        if isinstance(self.app.world,Ecosystem):
            self.app.world.ecology=EcologySettings(enabled=True)
            self.sync_ecology()

    def sync_ecology(self) -> None:
        if not hasattr(self,"eco_enabled"):
            return
        if not isinstance(self.app.world,Ecosystem):
            self.eco_enabled.set(False)
            self.climate_desc.configure(text="Переключитесь в режим «Экосистема».")
            return
        settings=self.app.world.ecology
        self.eco_enabled.set(settings.enabled)
        for name,var in self.eco_toggles.items():
            var.set(getattr(settings,name))
        for name,scale in self.eco_sliders.items():
            scale.set(getattr(settings,name))
        self._show_climate()

    def _show_climate(self) -> None:
        if not isinstance(self.app.world,Ecosystem):
            return
        values=climate_values(self.app.world.generation,self.app.world.ecology)
        world=self.app.world
        mode="ВКЛЮЧЕНО" if world.ecology.enabled else "ВЫКЛЮЧЕНО"
        self.climate_desc.configure(text=f"Динамика среды: {mode}\nСезон: {values['season']} · Свет: {values['sunlight']:.2f}\nВода: {world.moisture.mean():.1f}% · Минералы: {world.nutrients.mean():.1f}%")

    def _build_library(self) -> None:
        p=self.pages["Каталог"]
        self._heading(p,"БИБЛИОТЕКА СТРУКТУР", "Устойчивые фигуры, осцилляторы, корабли и метузелахи. Выберите форму и установите её на поле.")
        self.pattern_choice=tk.StringVar(value=self.pattern_selected)
        choice=ttk.Combobox(p,textvariable=self.pattern_choice,values=list(PATTERNS),state="readonly",style="Lab.TCombobox")
        choice.pack(fill="x",pady=(0,10))
        choice.bind("<<ComboboxSelected>>",lambda _e:self.choose_pattern(self.pattern_choice.get()))
        self.pattern_label=self.app._label(p,"",9,self.colors.muted)
        self.pattern_label.pack(anchor="w",pady=(0,9))
        self.pattern_canvas=tk.Canvas(p,width=220,height=160,bg=self.colors.base,highlightthickness=0)
        self.pattern_canvas.pack(fill="x",pady=(0,10))
        self.pattern_canvas.bind("<Configure>",lambda _e:self.draw_pattern())
        for row_items in (
            (("↻ 90°","clockwise"),("↺ 90°","counterclockwise"),("180°","rotate180")),
            (("↔ ОТРАЗИТЬ","flip_horizontal"),("↕ ОТРАЗИТЬ","flip_vertical")),
        ):
            row=tk.Frame(p,bg=self.colors.panel)
            row.pack(fill="x",pady=3)
            for title,action in row_items:
                self.app._button(row,title,lambda op=action:self.transform_pattern(op)).pack(side="left",expand=True,fill="x",padx=2)
        self.app._button(p,"ВЫБРАТЬ ШТАМП",self.prepare_stamp,accent=True).pack(fill="x",pady=(12,5))
        self.app._button(p,"ПОСТАВИТЬ ПО ЦЕНТРУ",self.stamp_center).pack(fill="x",pady=(0,10))
        for name,dx,dy in (("СДВИНУТЬ МИР ←",-1,0),("СДВИНУТЬ МИР →",1,0),
                           ("СДВИНУТЬ МИР ↑",0,-1),("СДВИНУТЬ МИР ↓",0,1)):
            self.app._button(p,name,lambda x=dx,y=dy:self.shift_world(x,y)).pack(fill="x",pady=2)
        self.draw_pattern()

    def choose_pattern(self,name:str) -> None:
        self.pattern_selected=name
        self.pattern_image=pattern_array(name)
        self.app.pattern_name.set(name)
        self.draw_pattern()

    def transform_pattern(self,op:str)->None:
        self.pattern_image=transform(self.pattern_image,op)
        self.draw_pattern()
        self.app._mark_dirty()

    def draw_pattern(self) -> None:
        if not hasattr(self,"pattern_canvas"):
            return
        cv=self.pattern_canvas
        cv.delete("all")
        pat=self.pattern_image
        h,w=pat.shape
        self.pattern_label.config(text=f"{w} × {h}  ·  {int(pat.sum())} живых клеток")
        W=max(100,cv.winfo_width());H=max(100,cv.winfo_height())
        scale=max(1,min(15,(W-15)//w,(H-15)//h))
        startx=(W-w*scale)//2
        starty=(H-h*scale)//2
        for y,x in np.argwhere(pat):
            xx,yy=startx+int(x)*scale,starty+int(y)*scale
            cv.create_rectangle(xx,yy,xx+scale-1,yy+scale-1,fill=self.colors.accent,outline=self.colors.border)

    def stamp_array(self,name:str)->np.ndarray:
        return self.pattern_image if name==self.pattern_selected else pattern_array(name)

    def prepare_stamp(self)->None:
        if not isinstance(self.app.world,CellularAutomaton):
            self.app._show_hint("Штампы доступны только в классическом мире")
            return
        self.app.pattern_name.set(self.pattern_selected)
        self.app.set_tool("stamp")
        self.app._show_hint("ЛКМ — поставить структуру · ПКМ — перемещение камеры")

    def stamp_center(self)->None:
        if not isinstance(self.app.world,CellularAutomaton):
            self.app._show_hint("Штампы доступны только в классическом мире")
            return
        self.app._push_undo()
        arr=self.pattern_image
        place_pattern(self.app.world.cells,arr,(self.app.world.width-arr.shape[1])//2,
                      (self.app.world.height-arr.shape[0])//2)
        self.app._mark_dirty()
        self.app._update_stats()

    def shift_world(self,dx:int,dy:int)->None:
        if not isinstance(self.app.world,CellularAutomaton):
            self.app._show_hint("Сдвиг доступен только в классическом мире")
            return
        self.app._push_undo()
        self.app.world.cells[:]=shift(self.app.world.cells,dx,dy,self.app.world.wrap)
        self.app._mark_dirty()
        self.app._update_stats()

    def refresh(self,force:bool=False)->None:
        w=self.app.world
        if not force and self.last_refresh==w.generation:
            return
        self.last_refresh=w.generation
        if self.current=="Анализ":
            reading=measure(w)
            name=LABEL_TO_METRIC.get(self.metric_name.get(),"population")
            self.metric_value.config(text=formatted(reading.get(name)))
            self.metric_description.config(text=METRIC_TITLES.get(name,name))
            for field,label in self.metric_rows.items():
                label.config(text=formatted(reading[field]))
            if self.baseline:
                before=self.baseline.get(name)
                after=reading.get(name)
                delta=after-before if isinstance(before,(int,float)) and isinstance(after,(int,float)) else None
                self.compare_label.config(text=(f"Контрольная точка: g {self.baseline['generation']}\n"
                                               f"{METRIC_TITLES.get(name,name)}: {formatted(before)} → {formatted(after)}\n"
                                               f"Разница: {formatted(delta)}"))
            self._draw_metric_chart()
        elif self.current=="3D":
            self.update_preview()
        elif self.current=="Среда":
            self._show_climate()

    def check_extinction(self)->bool:
        if not self.alert_extinction.get() or not isinstance(self.app.world,Ecosystem):
            return False
        current=self.app.world.counts()
        rows=self.app.research_log.records
        if len(rows)<2:
            return False
        previous=rows[-2]
        if any(previous.get(k,0) and current[k]==0 for k in ("herbivores","predators")):
            self.app.running=False
            self.app._update_run_button()
            self.app._show_hint("Эксперимент остановлен: один из видов исчез")
            return True
        return False
