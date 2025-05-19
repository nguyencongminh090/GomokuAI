import numpy as np
from enum import IntEnum

# Enums from core/types.h
class Color(IntEnum):
    """Matches Color enum in core/types.h."""
    BLACK = 0
    WHITE = 1
    EMPTY = 2
    WALL = 3
    COLOR_NB = 2  # Number of playable colors (BLACK, WHITE)

    @staticmethod
    def opposite(side):
        """Matches opposite() inline function."""
        return Color.WHITE if side == Color.BLACK else Color.BLACK

class Rule(IntEnum):
    """Matches Rule enum in core/types.h."""
    FREESTYLE = 0
    STANDARD = 1
    RENJU = 2
    RULE_NB = 3

class Pattern(IntEnum):
    """Matches Pattern enum in core/types.h."""
    DEAD = 0
    B1 = 1
    F1 = 2
    B2 = 3
    F2 = 4
    F2A = 5
    F2B = 6
    B3 = 7
    F3 = 8
    F3S = 9
    B4 = 10
    F4 = 11
    F5 = 12
    OL = 13
    FORBID = 14
    PATTERN_NB = 15

class Pattern4(IntEnum):
    """Matches Pattern4 enum in core/types.h."""
    NONE = 0
    L_FLEX2 = 1
    J_FLEX2_2X = 2
    K_BLOCK3 = 3
    I_BLOCK3_PLUS = 4
    H_FLEX3 = 5
    G_FLEX3_PLUS = 6
    F_FLEX3_2X = 7
    E_BLOCK4 = 8
    D_BLOCK4_PLUS = 9
    C_BLOCK4_FLEX3 = 10
    B_FLEX4 = 11
    A_FIVE = 12
    FORBID = 13
    PATTERN4_NB = 14

class Bound(IntEnum):
    """Matches Bound enum in core/types.h (used in TT)."""
    NONE = 0
    LOWER = 1
    UPPER = 2
    EXACT = 3

# Custom Types from core/types.h
class Value:
    """Matches Value type (int16_t) in core/types.h."""
    def __init__(self, value):
        self.value = np.int16(value)

    def __add__(self, other):
        return Value(self.value + other.value)

    def __sub__(self, other):
        return Value(self.value - other.value)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Value(self.value * other)
        return Value(self.value * other.value)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Value(self.value / other)
        return Value(self.value / other.value)

    def __neg__(self):
        return Value(-self.value)

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value

    def __int__(self):
        return int(self.value)

    def __repr__(self):
        return f"Value({self.value})"

class Depth:
    """Matches Depth type (float) in core/types.h."""
    def __init__(self, value):
        self.value = float(value)

    def __add__(self, other):
        return Depth(self.value + other.value)

    def __sub__(self, other):
        return Depth(self.value - other.value)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Depth(self.value * other)
        return Depth(self.value * other.value)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Depth(self.value / other)
        return Depth(self.value / other.value)

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return self.value

    def __repr__(self):
        return f"Depth({self.value})"

class Pos:
    """Matches Pos struct in core/types.h."""
    def __init__(self, x, y):
        self.x = np.int16(x)
        self.y = np.int16(y)

    @staticmethod
    def from_index(index):
        """Matches Pos(int index) constructor."""
        from .board import FULL_BOARD_SIZE, BOARD_BOUNDARY  # Deferred import
        if index >= FULL_BOARD_SIZE * FULL_BOARD_SIZE:
            return Pos.NONE
        x = (index % FULL_BOARD_SIZE) - BOARD_BOUNDARY
        y = (index // FULL_BOARD_SIZE) - BOARD_BOUNDARY
        return Pos(x, y)

    def to_index(self):
        """Matches toIndex()."""
        from .board import FULL_BOARD_SIZE, BOARD_BOUNDARY
        return (self.y + BOARD_BOUNDARY) * FULL_BOARD_SIZE + (self.x + BOARD_BOUNDARY)

    def is_valid(self):
        """Matches isValid()."""
        from .board import MAX_BOARD_SIZE
        return 0 <= self.x < MAX_BOARD_SIZE and 0 <= self.y < MAX_BOARD_SIZE

    def is_in_board(self, width, height):
        """Matches isInBoard()."""
        return 0 <= self.x < width and 0 <= self.y < height

    def __add__(self, other):
        """Matches operator+ with direction tuple."""
        return Pos(self.x + other[0], self.y + other[1])

    def __sub__(self, other):
        """Matches operator- with direction tuple."""
        return Pos(self.x - other[0], self.y - other[1])

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __lt__(self, other):
        return self.to_index() < other.to_index()

    def __repr__(self):
        return f"Pos({self.x}, {self.y})"

    NONE = None  # type: ignore # Set in board.py