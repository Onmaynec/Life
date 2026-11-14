"""Shared chart geometry for the on-screen Life research panels.

The chart never writes into the simulation or experiment history. Missing
observations remain gaps instead of being converted to zeroes.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def hex_blend(a: str, b: str, fraction: float) -> str:
    """Blend two six-digit colours; useful for muted surfaces in every skin."""
    if not (0 <= fraction <= 1):
        raise ValueError("blend must be between zero and one")
    if len(a) != 7 or len(b) != 7 or a[0] != "#" or b[0] != "#":
        raise ValueError("colours must be #RRGGBB")
    try:
        first = [int(a[i:i+2], 16) for i in (1, 3, 5)]
        second = [int(b[i:i+2], 16) for i in (1, 3, 5)]
    except ValueError as exc:
        raise ValueError("colours must be #RRGGBB") from exc
    return "#" + "".join(f"{round(x * (1-fraction) + y * fraction):02x}" for x,y in zip(first,second))


def contiguous_segments(records: Sequence[Mapping[str, Any]], key: str) -> list[list[tuple[int, float]]]:
    """Return chart segments, retaining the original timeline across gaps."""
    segments: list[list[tuple[int, float]]] = []
    current: list[tuple[int, float]] = []
    for i, record in enumerate(records):
        value = record.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and float("-inf") < float(value) < float("inf"):
            current.append((i, float(value)))
        elif current:
            segments.append(current)
            current = []
    if current:
        segments.append(current)
    return segments


def nearest_sample_index(x: float, left: float, right: float, count: int) -> int:
    """Map a cursor position to its nearest sample for interactive inspection."""
    if count < 1 or right <= left:
        return 0
    return min(count-1, max(0, round((x-left)/(right-left)*(count-1))))


def draw_history_chart(canvas: Any, rows: Sequence[Mapping[str, Any]],
                       keys: Sequence[str], colors: Sequence[str], *,
                       background: str, border: str, muted: str,
                       fill: bool = True, height: int = 164) -> None:
    """Render an accessible, annotated multi-series chart on a Tk Canvas.

    Positions are based on the *original record indices*: missing values cause
    line breaks but never squeeze or shift the generation axis.
    """
    canvas.delete("all")
    w = max(180, canvas.winfo_width())
    h = max(100, canvas.winfo_height(), height)
    left, right, top, bottom = 16, w-15, 24, h-28
    for i in range(5):
        y = top + (bottom-top)*i/4
        canvas.create_line(left, y, right, y, fill=border, dash=(2, 5))
    if len(rows) < 2:
        canvas.create_text(w/2, h/2, text="История появится после запуска",
                           fill=muted, font=("Segoe UI", 9))
        return
    observed = [value for key in keys for segment in contiguous_segments(rows,key)
                for _,value in segment]
    min_val = min(0.0, min(observed, default=0.0))
    max_val = max(1.0, max(observed, default=1.0))
    if max_val == min_val:
        max_val += 1
    max_val += (max_val-min_val)*0.06
    def xy(index: int, value: float) -> tuple[float, float]:
        return (left + index/max(1,len(rows)-1)*(right-left),
                bottom - (value-min_val)/(max_val-min_val)*(bottom-top))
    canvas.create_text(right, 8, text=f"{max_val:.2f}" if max_val < 20 else f"{max_val:,.0f}",
                       fill=muted,anchor="ne",font=("Segoe UI",8))
    for i, key in enumerate(keys):
        color=colors[i]
        for seg in contiguous_segments(rows,key):
            coords=[xy(idx,value) for idx,value in seg]
            if len(coords) < 2:
                continue
            flattened=[q for pair in coords for q in pair]
            if fill and i == 0:
                low = hex_blend(background,color,0.35)
                polygon=[coords[0][0],bottom,*flattened,coords[-1][0],bottom]
                canvas.create_polygon(*polygon,fill=low,outline="",stipple="gray25")
            canvas.create_line(*flattened,fill=color,width=2.3,joinstyle="round")
            x,y=coords[-1]
            canvas.create_oval(x-3,y-3,x+3,y+3,fill=color,outline=background,width=1)
    first = rows[0].get("generation", "—")
    last = rows[-1].get("generation", "—")
    canvas.create_text(left,h-9,text=f"{first}",anchor="w",fill=muted,font=("Segoe UI",8))
    canvas.create_text(right,h-9,text=f"{last}",anchor="e",fill=muted,font=("Segoe UI",8))
    if len(keys)>1:
        legend_x=left+4
        for color in colors:
            canvas.create_oval(legend_x, 6, legend_x+6,12,fill=color,outline="")
            legend_x+=14
