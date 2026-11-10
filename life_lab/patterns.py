"""Ready-made Conway patterns and standard RLE import/export."""
from __future__ import annotations

import re
import numpy as np

# Patterns are defined by generations, not image files.
PATTERNS: dict[str, tuple[str, ...]] = {
    "Блок": ("OO", "OO"),
    "Улей": (".OO.", "O..O", ".OO."),
    "Каравай": (".OO.", "O..O", ".O.O", "..O."),
    "Лодка": ("OO.", "O.O", ".O."),
    "Ванна": (".O.", "O.O", ".O."),
    "Змея": ("OO.O", "O.OO"),
    "Мигалка": ("OOO",),
    "Жаба": (".OOO", "OOO."),
    "Маяк": ("OO..", "OO..", "..OO", "..OO"),
    "Часы": ("..O.", "OO..", "..OO", ".O.."),
    "Галактика": ("OO.OO.", "OO.OO.", "OO.OO.", ".OO.OO", ".OO.OO", ".OO.OO"),
    "Большой маяк": ("OO....", "OO....", "......", "....OO", "....OO"),
    "Крест из блоков": ("..OO..", "..OO..", "OO..OO", "OO..OO", "..OO..", "..OO.."),
    "Diehard": ("......O.", "OO......", ".O...OOO"),
    "Acorn": (".O.....", "...O...", "OO..OOO"),
    "B-гептомино": ("OO.", "O.O", ".OO", "..O"),
    "Пи-гептомино": ("OOO", "O.O", "O.O"),
    "Два глайдера": (".O......O.", "..O......O", "OOO....OOO"),
    "Три глайдера": (".O......O.......O.", "..O......O.......O", "OOO....OOO.....OOO"),
    "Квадрат из глайдеров": (".O.........O.", "..O.........O", "OOO.......OOO", ".............", ".............", "OOO.......OOO", "..O.........O", ".O.........O."),
    "Глайдер": (
        ".O.",
        "..O",
        "OOO",
    ),
    "Лёгкий корабль": (
        ".O..O",
        "O....",
        "O...O",
        "OOOO.",
    ),
    "R-пентамино": (
        ".OO",
        "OO.",
        ".O.",
    ),
    "Пульсар": (
        "..OOO...OOO..",
        ".............",
        "O....O.O....O",
        "O....O.O....O",
        "O....O.O....O",
        "..OOO...OOO..",
        ".............",
        "..OOO...OOO..",
        "O....O.O....O",
        "O....O.O....O",
        "O....O.O....O",
        ".............",
        "..OOO...OOO..",
    ),
    "Пушка Госпера": (
        "........................O...........",
        "......................O.O...........",
        "............OO......OO............OO",
        "...........O...O....OO............OO",
        "OO........O.....O...OO..............",
        "OO........O...O.OO....O.O...........",
        "..........O.....O.......O...........",
        "...........O...O....................",
        "............OO......................",
    ),
}


def pattern_array(name: str) -> np.ndarray:
    lines = PATTERNS[name]
    width = max(len(line) for line in lines)
    return np.array([[1 if x < len(line) and line[x] == "O" else 0
                      for x in range(width)] for line in lines], dtype=np.uint8)


def place_pattern(cells: np.ndarray, pattern: np.ndarray, x: int, y: int, replace: bool = False) -> int:
    """Place a finite pattern clipping to the grid, return count of changed cells."""
    h, w = cells.shape
    ph, pw = pattern.shape
    x0, x1 = max(0, x), min(w, x + pw)
    y0, y1 = max(0, y), min(h, y + ph)
    if x0 >= x1 or y0 >= y1:
        return 0
    target = cells[y0:y1, x0:x1]
    crop = pattern[y0-y:y1-y, x0-x:x1-x]
    before = target.copy()
    if replace:
        target[:] = crop
    else:
        target[crop != 0] = crop[crop != 0]
    return int(np.count_nonzero(target != before))


def decode_rle(data: str) -> tuple[np.ndarray, str | None]:
    """Decode Life RLE: comments, x/y header, b/o/$/! runs, optional rule."""
    lines = [l.strip() for l in data.splitlines() if l.strip() and not l.startswith("#")]
    if not lines:
        raise ValueError("Пустой RLE-файл")
    header = lines[0]
    mx, my = re.search(r"\bx\s*=\s*(\d+)", header), re.search(r"\by\s*=\s*(\d+)", header)
    if not mx or not my:
        raise ValueError("Заголовок RLE должен содержать x и y")
    w, h = int(mx.group(1)), int(my.group(1))
    if not (1 <= w <= 700 and 1 <= h <= 500):
        raise ValueError("Размер RLE ограничен 700×500")
    mr = re.search(r"\brule\s*=\s*([^,\s]+)", header, flags=re.I)
    rule = mr.group(1).upper() if mr else None
    body = "".join(lines[1:]).replace(" ", "")
    arr = np.zeros((h, w), dtype=np.uint8)
    x = y = 0
    count = ""
    stopped = False
    for c in body:
        if c.isdigit():
            count += c
            continue
        n = int(count) if count else 1
        count = ""
        if c in "bo":
            if y >= h or x + n > w:
                raise ValueError("Данные RLE выходят за границы изображения")
            if c == "o":
                arr[y, x:x+n] = 1
            x += n
        elif c == "$":
            y += n
            x = 0
            if y > h:
                raise ValueError("Слишком много строк RLE")
        elif c == "!":
            stopped = True
            break
        else:
            raise ValueError(f"Неизвестный символ RLE: {c}")
    if not stopped:
        raise ValueError("RLE-файл должен заканчиваться символом !")
    return arr, rule


def encode_rle(cells: np.ndarray, rule: str = "B3/S23") -> str:
    """Encode minimal live bounding box, wrapped at ~70 columns."""
    ys, xs = np.nonzero(cells)
    if len(xs):
        arr = cells[int(ys.min()):int(ys.max())+1, int(xs.min()):int(xs.max())+1]
    else:
        arr = np.zeros((1, 1), dtype=np.uint8)
    h, w = arr.shape
    pieces = []
    for y in range(h):
        row = "".join("o" if v else "b" for v in arr[y]).rstrip("b")
        if row:
            groups = re.findall(r"(o+|b+)", row)
            for g in groups:
                pieces.append((str(len(g)) if len(g) > 1 else "") + g[0])
        if y < h - 1:
            pieces.append("$")
    pieces.append("!")
    body = "".join(pieces)
    return f"x = {w}, y = {h}, rule = {rule}\n" + "\n".join(body[i:i+70] for i in range(0, len(body), 70)) + "\n"
