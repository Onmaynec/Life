"""3D spatial preview and interactive Matplotlib viewer for ResearchWorkspace."""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from .instruments import spatial_field, SPATIAL_UNITS

GRAPH_FIELDS = ["Плотность", "Типы клеток", "Соседи", "Энергия", "Минералы", "Влага",
                "Возраст", "Метаболизм", "Зрение", "Плодовитость",
                "Семена", "Органика", "Токсины", "Рельеф", "Глубина родословной"]


class Research3DTools:
    """Mixin containing the independent 3D tab and its full-size viewer."""

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
        aggregation=tk.StringVar(value="Среднее")
        detail=tk.StringVar(value="29 × 29")
        last_key=[None]
        self.app._label(top,"ПОЛЕ",9).pack(side="left",padx=(0,5))
        selector=ttk.Combobox(top,textvariable=field,values=GRAPH_FIELDS,state="readonly",width=15,style="Lab.TCombobox")
        selector.pack(side="left",padx=(0,13))
        chosen=ttk.Combobox(top,textvariable=style,values=("Столбцы","Поверхность","Сетка"),state="readonly",width=14,style="Lab.TCombobox")
        chosen.pack(side="left",padx=(0,8))
        ttk.Combobox(top,textvariable=aggregation,values=("Среднее","Максимум","Сумма"),
                     width=10,state="readonly",style="Lab.TCombobox").pack(side="left",padx=(0,8))
        ttk.Combobox(top,textvariable=detail,values=("17 × 17","29 × 29","45 × 45","65 × 65"),
                     width=10,state="readonly",style="Lab.TCombobox").pack(side="left")
        figure=Figure(figsize=(9.8,6.2),dpi=105,facecolor=self.colors.base)
        ax=figure.add_subplot(111,projection="3d",computed_zorder=False)
        canvas=FigureCanvasTkAgg(figure,master=window)
        canvas.get_tk_widget().configure(bg=self.colors.base,highlightthickness=0)
        canvas.get_tk_widget().pack(fill="both",expand=True)
        tk.Label(window,text="ЛКМ на графике — вращение  ·  колесо — масштаб  ·  кнопка «Обновить данные» — новый снимок",
                 bg=self.colors.panel,fg=self.colors.muted,font=("Segoe UI",9),anchor="w",padx=14,pady=8).pack(fill="x",side="bottom")
        cmap=LinearSegmentedColormap.from_list("life",[self.colors.base,self.colors.border,self.colors.accent,"#FFF0BD"])
        def redraw(*_args:Any, force: bool = False) -> None:
            key=(id(self.app.world),self.app.world.generation,field.get(),style.get(),
                 aggregation.get(),detail.get())
            if key == last_key[0] and not force:
                return
            last_key[0] = key
            method={"Среднее":"mean","Максимум":"max","Сумма":"sum"}[aggregation.get()]
            data=spatial_field(self.app.world,field.get(),max_side=int(detail.get().split()[0]),aggregation=method)
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
            unit=SPATIAL_UNITS.get(field.get(),"ед.")
            ax.set_title(f"{field.get()} · g{self.app.world.generation} · {aggregation.get().lower()} · {unit}",
                         color=self.colors.text,pad=18)
            ax.set_xlabel("X",color=self.colors.muted)
            ax.set_ylabel("Y",color=self.colors.muted)
            ax.set_zlabel(f"{field.get()} ({unit})",color=self.colors.muted)
            ax.view_init(elev=elevation,azim=azimuth)
            figure.tight_layout(pad=2)
            canvas.draw_idle()
        selector.bind("<<ComboboxSelected>>",redraw)
        chosen.bind("<<ComboboxSelected>>",redraw)
        for var in (aggregation,detail):
            var.trace_add("write",lambda *_:redraw())
        self.app._button(top,"ОБНОВИТЬ ДАННЫЕ",lambda:redraw(force=True)).pack(side="left",padx=(10,0))
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

