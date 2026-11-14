"""Lineage comparison and spatial genetic survey."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Any
import numpy as np
from PIL import Image, ImageTk

from .lineage_analysis import TRAITS, compare_families, export_family_history, history, trait_map
from .skins import skin
from .world import Ecosystem


class LineageComparison:
    def __init__(self, app: Any):
        self.app = app
        self.colors = skin(app.skin_var.get())
        self.window = tk.Toplevel(app.root)
        self.window.title("Life — сравнение родовых линий")
        self.window.geometry("990x760")
        self.window.minsize(750,580)
        self.window.configure(bg=self.colors.base)
        self.first = tk.StringVar()
        self.second = tk.StringVar()
        self.trait = tk.StringVar(value="Метаболизм")
        top = tk.Frame(self.window, bg=self.colors.panel, padx=15, pady=12)
        top.pack(fill="x")
        tk.Label(top,text="ГЕНЕТИЧЕСКИЕ ЛИНИИ",font=("Segoe UI",17,"bold"),
                 bg=self.colors.panel,fg=self.colors.text).pack(anchor="w")
        toolbar = tk.Frame(self.window,bg=self.colors.panel,padx=15,pady=8)
        toolbar.pack(fill="x",pady=(7,9))
        families = [str(uid) for uid,_,_ in app.world.lineage.dominant_families(50)] if isinstance(app.world,Ecosystem) else []
        self.first.set(families[0] if families else "")
        self.second.set(families[1] if len(families)>1 else (families[0] if families else ""))
        for title,variable in (("Линия A",self.first),("Линия B",self.second)):
            tk.Label(toolbar,text=title,bg=self.colors.panel,fg=self.colors.muted).pack(side="left",padx=5)
            ttk.Combobox(toolbar,textvariable=variable,values=families,width=8).pack(side="left",padx=4)
        ttk.Combobox(toolbar,textvariable=self.trait,values=list(TRAITS),
                     state="readonly",width=16).pack(side="left",padx=9)
        tk.Button(toolbar,text="ОБНОВИТЬ",command=self.refresh,bg=self.colors.surface,
                  fg=self.colors.text).pack(side="left",padx=4)
        tk.Button(toolbar,text="CSV",command=self.export,bg=self.colors.surface,
                  fg=self.colors.text).pack(side="right")
        self.summary = tk.Label(self.window,text="",bg=self.colors.base,fg=self.colors.text,
                                font=("Consolas",10),justify="left",anchor="w")
        self.summary.pack(fill="x",padx=16,pady=(0,10))
        tk.Label(self.window,text="РАСПРЕДЕЛЕНИЕ ПРИЗНАКА И ДИНАМИКА ЧИСЛЕННОСТИ",bg=self.colors.base,
                 fg=self.colors.muted,font=("Segoe UI",10,"bold")).pack(anchor="w",padx=15)
        self.chart = tk.Canvas(self.window,height=255,bg=self.colors.panel,highlightthickness=0)
        self.chart.pack(fill="x",padx=15,pady=8)
        self.map_canvas = tk.Canvas(self.window,bg=self.colors.base,highlightthickness=0)
        self.map_canvas.pack(fill="both",expand=True,padx=15,pady=(5,15))
        self.preview = None
        self.refresh()

    def _ids(self) -> tuple[int,int]:
        try:
            return int(self.first.get()),int(self.second.get())
        except ValueError as exc:
            raise ValueError("Введите числовые ID двух основателей") from exc

    def refresh(self) -> None:
        world = self.app.world
        if not self.first.get().strip() or not self.second.get().strip():
            self.summary.config(text="Нет живых семей для сравнения. Создайте экосистему с животными.")
            return
        if not isinstance(world,Ecosystem):
            self.summary.config(text="Родословные доступны только в режиме экосистемы")
            return
        try:
            a,b = self._ids()
            stats = compare_families(world,a,b)
            rows = history(world,(a,b),every=max(1,world.generation//100))
            field = trait_map(world,self.trait.get(),a)
        except ValueError as exc:
            messagebox.showerror("Генетика",str(exc),parent=self.window)
            return
        columns = (("metabolism","Метаболизм"),("sight","Зрение"),("fertility","Плодовитость"))
        text = ["                    Линия A          Линия B", f"Основатель          #{a:<14} #{b}",
                f"Живых                {stats[0]['living']:<16} {stats[1]['living']}"]
        for key,title in columns:
            numbers = [("—" if row[key] is None else f"{row[key]:.3f}") for row in stats]
            text.append(f"{title:<20} {numbers[0]:<16} {numbers[1]}")
        text.append(f"Известных потомков  {stats[0]['offspring']:<16} {stats[1]['offspring']}")
        self.summary.config(text="\n".join(text))
        self.chart.delete("all")
        w=max(600,self.chart.winfo_width())
        trait_key={"Метаболизм":"metabolism","Зрение":"sight","Плодовитость":"fertility"}[self.trait.get()]
        max_count=max(1,max(max(row[trait_key+"_distribution"]) for row in stats))
        for k in range(8):
            bx=50+k*(w-95)/8
            for i,row in enumerate(stats):
                count=row[trait_key+"_distribution"][k]
                bh=count/max_count*70
                self.chart.create_rectangle(bx+i*12,103-bh,bx+i*12+10,103,
                                            fill=("#62d4df","#f3aa80")[i],outline="")
        self.chart.create_text(20,5,anchor="nw",text=f"Гистограмма: {self.trait.get()} (8 одинаковых интервалов)",fill=self.colors.muted)
        self.chart.create_line(45,104,w-22,104,fill=self.colors.border)
        self.chart.create_text(20,124,anchor="nw",text="Численность по поколениям · данные из доступного архива",fill=self.colors.muted)
        for index,founder in enumerate((a,b)):
            samples=[row for row in rows if row["founder"] == founder]
            if samples:
                height=max(1,max(x["living"] for x in rows))
                coords=[]
                for row in samples:
                    coords.extend((40 + row["generation"]/max(1,world.generation)*(w-70),
                                   234 - row["living"]/height*90))
                if len(coords)>2:
                    self.chart.create_line(*coords,fill=("#62d4df","#f3aa80")[index],width=2)
        self.chart.create_text(w-12,9,anchor="ne",text="A — голубой  ·  B — оранжевый",fill=self.colors.muted)
        # Transparent-by-data display: NaN (no animal of this family) is the dark background.
        h2,w2=field.shape
        arr=np.zeros((h2,w2,3),dtype=np.uint8)
        arr[:]=(14,17,22)
        valid=np.isfinite(field)
        if valid.any():
            values=field[valid]
            lo,hi=float(np.min(values)),float(np.max(values))
            fraction=np.clip((values-lo)/max(1e-8,hi-lo),0,1)
            arr[valid,0]=np.uint8(30+fraction*220)
            arr[valid,1]=np.uint8(105+fraction*100)
            arr[valid,2]=np.uint8(160-fraction*100)
        img=Image.fromarray(arr,"RGB")
        img.thumbnail((900,340),Image.Resampling.NEAREST)
        scale=max(1,min(700//max(1,img.width),300//max(1,img.height)))
        self.preview=ImageTk.PhotoImage(img.resize((img.width*scale,img.height*scale),Image.Resampling.NEAREST))
        self.map_canvas.delete("all")
        self.map_canvas.create_image(9,20,anchor="nw",image=self.preview)
        legend=(f"{lo:.2f} → {hi:.2f}" if valid.any() else "нет живых потомков")
        self.map_canvas.create_text(9,6,anchor="nw",text=f"Карта: {self.trait.get()} · линия #{a} · {legend} · тёмное = нет потомка",
                                    fill=self.colors.muted,font=("Segoe UI",9))

    def export(self) -> None:
        try:
            a,b=self._ids()
        except ValueError as exc:
            messagebox.showerror("Генетика",str(exc),parent=self.window)
            return
        path=filedialog.asksaveasfilename(parent=self.window,defaultextension=".csv",
                                          filetypes=[("CSV","*.csv")])
        if path:
            export_family_history(path,self.app.world,a,b)
