"""Pure Pillow drawing for the Life viewport and navigator.

World data never changes when a palette, grid, or drawing style is selected.
The same renderer is used by the on-screen viewport and PNG export.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Palette:
    background: tuple[int, int, int]
    soil: tuple[int, int, int]
    grid: tuple[int, int, int]
    species: tuple[tuple[int, int, int], ...]
    trail: tuple[int, int, int]
    border: str


PALETTES: dict[str, Palette] = {
    "Графит": Palette((11, 14, 18), (14, 18, 22), (37, 46, 54),
                        ((82, 216, 231), (240, 183, 91), (222, 112, 142)), (40, 98, 116), "#36434E"),
    "Оникс": Palette((10, 13, 12), (16, 20, 18), (38, 48, 40),
                      ((173, 226, 103), (113, 200, 164), (231, 128, 106)), (53, 89, 62), "#34433B"),
    "Титан": Palette((14, 16, 19), (19, 22, 25), (43, 49, 55),
                      ((238, 182, 105), (137, 193, 219), (228, 137, 139)), (80, 71, 55), "#404852"),
    "Аметист": Palette((11, 10, 17), (19, 17, 26), (44, 37, 59),
                        ((171, 141, 255), (99, 193, 224), (245, 121, 185)), (83, 62, 130), "#413A57"),
    "Аврора": Palette((10, 18, 30), (13, 28, 34), (32, 51, 62),
                      ((107, 234, 183), (248, 199, 108), (255, 117, 151)), (45, 105, 94), "#4d7183"),
    "Биосфера": Palette((13, 24, 22), (20, 31, 25), (39, 59, 48),
                        ((164, 228, 106), (252, 191, 111), (237, 130, 111)), (69, 106, 64), "#638771"),
    "Ультрафиолет": Palette((17, 13, 34), (24, 20, 42), (50, 42, 78),
                           ((113, 222, 245), (178, 148, 255), (255, 113, 194)), (85, 66, 132), "#77649c"),
}
STYLES = ("Свечение", "Мозаика", "Схема")


def colorize(cells: np.ndarray, mode: str, palette: Palette,
             trails: np.ndarray | None = None, nutrients: np.ndarray | None = None) -> np.ndarray:
    """Return an H x W x 3 uint8 image without touching the model arrays."""
    if cells.ndim != 2 or cells.dtype.kind not in "uib":
        raise ValueError("Ожидается двумерный массив клеток")
    if mode not in ("classic", "ecosystem"):
        raise ValueError("Неизвестный режим визуализации")
    image = np.empty((*cells.shape, 3), dtype=np.uint8)
    image[:] = palette.background if mode == "classic" else palette.soil
    if mode == "classic" and trails is not None:
        if trails.shape != cells.shape:
            raise ValueError("Размер следов не совпадает с миром")
        intensity = np.clip(trails.astype(np.float32) / 7, 0, 1)
        for channel, color in enumerate(palette.trail):
            image[..., channel] = np.clip(image[..., channel].astype(np.float32)
                                          + intensity * color * 0.7, 0, 255).astype(np.uint8)
    if mode == "ecosystem" and nutrients is not None:
        if nutrients.shape != cells.shape:
            raise ValueError("Размер почвы не совпадает с миром")
        strength = np.clip(nutrients.astype(np.float32), 0, 100) / 100
        for channel, gain in enumerate((11, 22, 13)):
            image[..., channel] = np.clip(image[..., channel].astype(np.float32)
                                          + strength * gain, 0, 255).astype(np.uint8)
    for kind, color in enumerate(palette.species, start=1):
        if mode == "classic" and kind != 1:
            break
        image[cells == kind] = color
    return image


def _pale(color: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    return tuple(min(255, round(c + (255 - c) * amount)) for c in color)


@lru_cache(maxsize=320)
def _light_sprite(size: int, kind: int, palette_name: str) -> Image.Image:
    """Subpixel-sized light with an antialiased center, cached by cell size."""
    color = PALETTES[palette_name].species[kind-1]
    k = 3
    s = size * k
    image = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image, "RGBA")
    draw.ellipse((0, 0, s-1, s-1), fill=(*color, 44))
    rim = max(2, s // 7)
    draw.ellipse((rim, rim, s-rim, s-rim), fill=(*color, 95))
    pad = max(3, round(s * 0.23))
    if kind == 1:
        draw.ellipse((pad, pad, s-pad, s-pad), fill=(*color, 235))
    elif kind == 2:
        draw.rounded_rectangle((pad, pad, s-pad, s-pad), radius=s//7, fill=(*color, 245))
    else:
        draw.polygon(((s//2, pad), (s-pad, s//2), (s//2, s-pad), (pad, s//2)), fill=(*color, 244))
    highlight = _pale(color, 0.73)
    dot = max(1, s // 14)
    draw.ellipse((s//2-dot, s//2-dot, s//2+dot, s//2+dot), fill=(*highlight, 245))
    return image.resize((size, size), Image.Resampling.LANCZOS)


def draw_view(cells: np.ndarray, *, mode: str, zoom: int, palette_name: str,
              style: str = "Свечение", grid: bool = True,
              trails: np.ndarray | None = None,
              nutrients: np.ndarray | None = None) -> Image.Image:
    """Draw a cropped patch in its exact on-screen pixel dimensions."""
    if palette_name not in PALETTES:
        raise ValueError("Неизвестная палитра")
    if style not in STYLES:
        raise ValueError("Неизвестный стиль клеток")
    if not 1 <= zoom <= 60:
        raise ValueError("Масштаб должен быть в пределах от 1 до 60")
    palette = PALETTES[palette_name]
    arr = colorize(cells, mode, palette, trails, nutrients)
    h, w = cells.shape
    if w == 0 or h == 0:
        raise ValueError("Нельзя рисовать пустой фрагмент")
    if zoom >= 6 and style in ("Свечение", "Мозаика"):
        # Keep a dark gutter between the individual, lit cells.
        amount = 0.36 if style == "Свечение" else 0.22
        arr[cells != 0] = (arr[cells != 0].astype(np.float32) * amount).astype(np.uint8)
    image = Image.fromarray(arr, "RGB").resize((w * zoom, h * zoom), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    if grid and zoom >= 7:
        for x in range(zoom, w * zoom, zoom):
            draw.line((x, 0, x, h * zoom - 1), fill=palette.grid)
        for y in range(zoom, h * zoom, zoom):
            draw.line((0, y, w * zoom - 1, y), fill=palette.grid)
    if style == "Свечение" and zoom >= 7:
        canvas = image.convert("RGBA")
        sprites = {kind: _light_sprite(zoom, kind, palette_name) for kind in (1, 2, 3)}
        for y, x in np.argwhere(cells != 0):
            kind = int(cells[y, x])
            if kind in sprites:
                canvas.alpha_composite(sprites[kind], (int(x)*zoom, int(y)*zoom))
        return canvas.convert("RGB")
    if style == "Мозаика" and zoom >= 6:
        inset = max(1, zoom // 8)
        for y, x in np.argwhere(cells != 0):
            kind = int(cells[y, x])
            if kind > (1 if mode == "classic" else 3):
                continue
            color = palette.species[kind-1]
            x0, y0 = int(x)*zoom+inset, int(y)*zoom+inset
            x1, y1 = (int(x)+1)*zoom-1-inset, (int(y)+1)*zoom-1-inset
            draw.rounded_rectangle((x0, y0, x1, y1), radius=max(1, zoom//6),
                                   fill=_pale(color, 0.12), outline=_pale(color, 0.3), width=1)
            if zoom >= 11:
                draw.line((x0+2, y0+2, x1-2, y0+2), fill=_pale(color, 0.6), width=1)
    return image


def minimap(cells: np.ndarray, *, palette_name: str, width: int = 256,
            height: int = 128, mode: str = "classic") -> Image.Image:
    """Fast full-map navigator with colours matched to the current palette."""
    if width < 1 or height < 1:
        raise ValueError("Размер карты должен быть положительным")
    palette = PALETTES[palette_name]
    image = colorize(cells, mode, palette)
    return Image.fromarray(image, "RGB").resize((width, height), Image.Resampling.NEAREST)

# Research layers expose scalar arrays without modifying the simulation state.
HEATMAP_FIELDS = ("Организмы", "Минералы", "Влага", "Энергия", "Возраст", "Рельеф", "Органика", "Токсины")


def heatmap_view(values: np.ndarray, cells: np.ndarray, *, zoom: int,
                 palette_name: str, grid: bool = False) -> Image.Image:
    """Render a quantitative field with a neutral-to-accent ramp and faint organisms.

    Intensities are normalized to each field's *fixed* units at the caller, not
    per-image maxima, allowing meaningful visual comparison between generations.
    """
    if values.shape != cells.shape or values.ndim != 2:
        raise ValueError("Размер поля не совпадает с картой клеток")
    if palette_name not in PALETTES or not 1 <= zoom <= 60:
        raise ValueError("Неверный масштаб или палитра")
    palette=PALETTES[palette_name]
    intensity=np.clip(values.astype(np.float32),0,1)[...,None]
    low=np.asarray(palette.background,dtype=np.float32)
    high=np.asarray(palette.species[0],dtype=np.float32)
    pixels=np.clip(low+(high-low)*intensity,0,255).astype(np.uint8)
    for kind,color in enumerate(palette.species,start=1):
        mask=cells==kind
        if mask.any():
            pixels[mask]=np.clip(pixels[mask].astype(np.float32)*0.75 +
                                 np.asarray(color,dtype=np.float32)*0.25,0,255).astype(np.uint8)
    h,w=cells.shape
    im=Image.fromarray(pixels,"RGB").resize((w*zoom,h*zoom),Image.Resampling.NEAREST)
    if grid and zoom>=8:
        draw=ImageDraw.Draw(im)
        for x in range(zoom,w*zoom,zoom):
            draw.line((x,0,x,h*zoom),fill=palette.grid)
        for y in range(zoom,h*zoom,zoom):
            draw.line((0,y,w*zoom,y),fill=palette.grid)
    return im
