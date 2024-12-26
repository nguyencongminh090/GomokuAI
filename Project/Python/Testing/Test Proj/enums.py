# enums.py

from enum import Enum, auto

class Color(Enum):
    BLACK = 1
    WHITE = 2

class ColorFlag(Enum):
    SELF = auto()
    OPPO = auto()
    EMPT = auto()

class Pattern(Enum):
    DEAD = 0
    OL = 1
    B1 = 2
    F1 = 3
    B2 = 4
    F2 = 5
    F2A = 6
    F2B = 7
    B3 = 8
    F3 = 9
    F3S = 10
    B4 = 11
    F4 = 12
    F5 = 13
    PATTERN_NB = 14
    OLR = 15
    OLF = 16
