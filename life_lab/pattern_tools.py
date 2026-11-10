"""Lossless editing of binary Conway pattern matrices."""
from __future__ import annotations
import numpy as np


def transform(pattern: np.ndarray, operation: str) -> np.ndarray:
    if pattern.ndim != 2 or not np.isin(pattern, (0, 1)).all():
        raise ValueError("Ожидается двоичная матрица")
    if operation == "clockwise":
        return np.rot90(pattern, -1).copy()
    if operation == "counterclockwise":
        return np.rot90(pattern, 1).copy()
    if operation == "rotate180":
        return np.rot90(pattern, 2).copy()
    if operation == "flip_horizontal":
        return np.fliplr(pattern).copy()
    if operation == "flip_vertical":
        return np.flipud(pattern).copy()
    if operation == "original":
        return pattern.copy()
    raise ValueError("Неизвестное преобразование фигуры")


def shift(cells: np.ndarray, dx: int, dy: int, wrap: bool = False) -> np.ndarray:
    if wrap:
        return np.roll(np.roll(cells, dy, axis=0), dx, axis=1)
    out = np.zeros_like(cells)
    h, w = cells.shape
    src_x0, src_x1 = max(0, -dx), min(w, w-dx)
    src_y0, src_y1 = max(0, -dy), min(h, h-dy)
    if src_x1 > src_x0 and src_y1 > src_y0:
        out[src_y0+dy:src_y1+dy,src_x0+dx:src_x1+dx] = cells[src_y0:src_y1,src_x0:src_x1]
    return out


def fill_region(cells: np.ndarray, x0: int, y0: int, x1: int, y1: int, value: int) -> None:
    h,w = cells.shape
    x0,x1=sorted((x0,x1))
    y0,y1=sorted((y0,y1))
    cells[max(y0,0):min(h,y1+1),max(x0,0):min(w,x1+1)]=value
