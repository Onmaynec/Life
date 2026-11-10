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
}


def skin(name: str) -> Skin:
    return SKINS.get(name, SKINS["Графит / циан"])
