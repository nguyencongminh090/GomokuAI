from enum import Enum, auto
from typing import NewType

# Color representation
class Color(Enum):
    BLACK = auto()
    WHITE = auto()
    WALL = auto()
    EMPTY = auto()
    COLOR_NB = 4  # Total number of colors on board
    SIDE_NB = 2   # Two sides (Black and White)

# Pattern for a single line (aligned with Rapfi's Pattern)
class Pattern(Enum):
    DEAD = auto()  # X_.__X, can never make a five
    OL = auto()    # OO.OOO, one step before overline (forbidden in Standard/Renju)
    B1 = auto()    # X.____, one step before B2
    F1 = auto()    # X_.___, one step before F2
    B2 = auto()    # XO.___, one step before B3
    F2 = auto()    # _O__._, one step before two F3
    F2A = auto()   # _O_.__, one step before three F3
    F2B = auto()   # _O.___, one step before four F3
    B3 = auto()    # XOO.__, one step before B4
    F3 = auto()    # _OO_._, one step before one F4
    F3S = auto()   # __OO.__, one step before two F4
    B4 = auto()    # XOOO._X, one step before F5
    F4 = auto()    # _OOO._X, one step before two F5
    F5 = auto()    # XOO.OOX, making a five
    PATTERN_NB = 14

# Pattern4 for combined four lines (aligned with Rapfi's Pattern4)
class Pattern4(Enum):
    NONE = auto()            # Anything else
    FORBID = auto()          # Forbidden point (for Renju)
    L_FLEX2 = auto()         # F2+Any
    K_BLOCK3 = auto()        # B3+Any
    J_FLEX2_2X = auto()      # F2x2
    I_BLOCK3_PLUS = auto()   # B3x2 | B3+F2
    H_FLEX3 = auto()         # F3+Any
    G_FLEX3_PLUS = auto()    # F3+F2 | F3+B3
    F_FLEX3_2X = auto()      # F3x2
    E_BLOCK4 = auto()        # B4+Any
    D_BLOCK4_PLUS = auto()   # B4+F2 | B4+B3
    C_BLOCK4_FLEX3 = auto()  # B4+F3
    B_FLEX4 = auto()         # F4 | F4S | B4x2
    A_FIVE = auto()          # F5
    PATTERN4_NB = 14

# Value for search evaluation
Value = NewType('Value', int)
VALUE_ZERO = Value(0)
VALUE_DRAW = Value(0)
VALUE_MATE = Value(30000)
VALUE_INFINITE = Value(30001)
VALUE_NONE = Value(-30002)
VALUE_BLOCKED = Value(-30003)
VALUE_MATE_IN_MAX_PLY = Value(VALUE_MATE - 500)
VALUE_MATED_IN_MAX_PLY = Value(-VALUE_MATE + 500)
VALUE_MATE_FROM_DATABASE = VALUE_MATE_IN_MAX_PLY
VALUE_MATED_FROM_DATABASE = VALUE_MATED_IN_MAX_PLY
VALUE_EVAL_MAX = Value(20000)
VALUE_EVAL_MIN = Value(-20000)

# Rule for the game
class Rule(Enum):
    FREESTYLE = auto()
    STANDARD = auto()
    RENJU = auto()
    RULE_NB = 4

# Bound for alpha-beta search
class Bound(Enum):
    NONE = 0
    UPPER = 1  # Alpha bound
    LOWER = 2  # Beta bound
    EXACT = UPPER | LOWER  # PV bound

# Depth bounds
DEPTH_QVCF_FULL = -1.0
DEPTH_QVCF = -2.0
DEPTH_NONE = -3.0
DEPTH_LOWER_BOUND = -20.0
DEPTH_UPPER_BOUND = DEPTH_LOWER_BOUND + 255

# Candidate range for move generation
class CandidateRange(Enum):
    SQUARE2 = auto()
    SQUARE2_LINE3 = auto()
    SQUARE3 = auto()
    SQUARE3_LINE4 = auto()
    SQUARE4 = auto()
    FULL_BOARD = auto()
    CAND_RANGE_NB = 6

# Utility functions for Value
def mate_in(ply: int) -> Value:
    return Value(VALUE_MATE - ply)

def mated_in(ply: int) -> Value:
    return Value(-VALUE_MATE + ply)

def mate_step(value: Value, ply: int) -> int:
    return VALUE_MATE - ply - (value if value < 0 else -value)

def search_value_to_stored_value(value: Value, search_ply: int) -> int:
    if value == VALUE_NONE:
        return VALUE_NONE
    elif value >= VALUE_MATE_IN_MAX_PLY:
        return value + search_ply
    elif value <= VALUE_MATED_IN_MAX_PLY:
        return value - search_ply
    return value

def stored_value_to_search_value(stored_value: int, search_ply: int) -> Value:
    if stored_value == VALUE_NONE:
        return VALUE_NONE
    elif stored_value >= VALUE_MATE_IN_MAX_PLY:
        return Value(stored_value - search_ply)
    elif stored_value <= VALUE_MATED_IN_MAX_PLY:
        return Value(stored_value + search_ply)
    return Value(stored_value)