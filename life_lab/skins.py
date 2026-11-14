"""Dark workspace accents; neutral UI surfaces stay the same across themes."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Skin:
    base: str
    panel: str
    surface: str
    border: str
    text: str
    muted: str
    accent: str
    secondary: str
    warning: str
    danger: str


SKINS: dict[str, Skin] = {
    "Графит / циан": Skin("#0B0D10", "#14171C", "#1D2229", "#303640", "#F0F2F5", "#949DAA", "#54D9E6", "#9DAEFF", "#E3B56C", "#EB728B"),
    "Оникс / лайм": Skin("#0A0C0C", "#141A19", "#1E2724", "#303C37", "#EFF4F0", "#92A199", "#B8E46D", "#72D8BA", "#F6C77C", "#F18C83"),
    "Сталь / янтарь": Skin("#0D0F12", "#181B20", "#22262E", "#373E47", "#F0F0EF", "#9DA5AB", "#F1BA63", "#92C8D8", "#F1BA63", "#E77979"),
    "Полночь / фиолет": Skin("#0B0B11", "#16151E", "#211E2B", "#3A3447", "#F2F0F7", "#ABA4B8", "#BA9CFA", "#7EC6E4", "#F5BD76", "#EE82A9"),
    "Обсидиан / лёд": Skin("#080C11", "#10171D", "#1B252E", "#2D3E4B", "#EDF7FF", "#91A6B7", "#77E5F6", "#A6BBFF", "#F0C879", "#EE8494"),
    "Уголь / серебро": Skin("#090A0C", "#141518", "#202329", "#353A42", "#EDEFF1", "#A8B0BB", "#D4DEE8", "#99BED2", "#F4BD71", "#E37C87"),
    "Тёмный нефрит": Skin("#090F10", "#101A1B", "#1A2928", "#294542", "#EAF7F1", "#97B8AB", "#72E3BB", "#B7D991", "#EFC17B", "#EE8997"),
    "Тёмная медь": Skin("#100D0C", "#1D1917", "#2B2421", "#463933", "#FAF0E9", "#B5A49B", "#FFAF79", "#EEC9A5", "#F2C46C", "#E6848F"),
}


def skin(name: str) -> Skin:
    return SKINS.get(name, SKINS["Графит / циан"])
