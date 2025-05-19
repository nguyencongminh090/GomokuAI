"""
Core types and enumerations for the Gomoku engine, based on Rapfi's types.h.
"""
from enum import Enum, IntEnum, Flag
import math

# -------------------------------------------------
# Common type aliases (for type hinting)
PatternCode = int  # uint16_t
Score = int        # int16_t
Eval = int         # int16_t
Depth = float      # float
HashKey = int      # uint64_t

# -------------------------------------------------
# Range of searching depth and bound

class Bound(Flag):
    """Type of value bound of the alpha-beta search"""
    BOUND_NONE = 0
    BOUND_UPPER = 1  # Alpha bound
    BOUND_LOWER = 2  # Beta bound
    BOUND_EXACT = BOUND_UPPER | BOUND_LOWER  # PV bound

DEPTH_QVCF_FULL: Depth = -1.0
DEPTH_QVCF: Depth = -2.0
DEPTH_NONE: Depth = -3.0
DEPTH_LOWER_BOUND: Depth = -20.0
DEPTH_UPPER_BOUND: Depth = DEPTH_LOWER_BOUND + 255

# -------------------------------------------------

class Color(IntEnum):
    """Color represents the type of piece on board"""
    BLACK = 0
    WHITE = 1
    WALL = 2
    EMPTY = 3
    # COLOR_NB is implicitly len(Color)
    # SIDE_NB = 2 (Black and White)

    @staticmethod
    def get_opponent(player_color: 'Color') -> 'Color':
        if player_color == Color.BLACK:
            return Color.WHITE
        if player_color == Color.WHITE:
            return Color.BLACK
        raise ValueError("Cannot get opponent for non-player color")

    def __invert__(self) -> 'Color':
        """Returns the opposite of a color (Black <-> White, Wall <-> Empty)"""
        return Color(self.value ^ 1)

COLOR_NB = 4 # Total number of color on board (BLACK, WHITE, WALL, EMPTY)
SIDE_NB = 2  # Two sides of stones (Black and White)


# -------------------------------------------------

class Pattern(Enum):
    """Pattern is the type of a single line at one cell"""
    DEAD = 0   # X_.__X, can never make a five
    OL = 1     # OO.OOO, one step before overline
    B1 = 2     # X.____, one step before B2
    F1 = 3     # X_.___, one step before F2
    B2 = 4     # XO.___, one step before B3
    F2 = 5     # _O__._, one step before two F3
    F2A = 6    # _O_.__, one step before three F3
    F2B = 7    # _O.___, one step before four F3
    B3 = 8     # XOO.__, one step before B4
    F3 = 9     # _OO_._, one step before one F4
    F3S = 10   # __OO.__, one step before two F4
    B4 = 11    # XOOO._X, one step before F5
    F4 = 12    # _OOO._X, one step before two F5
    F5 = 13    # XOO.OOX, making a five
    # PATTERN_NB is implicitly len(Pattern)

PATTERN_NB = 14

class Pattern4(Enum):
    """Pattern4 is the combined type of 4 lines at one cell"""
    NONE = 0            # Anything else
    FORBID = 1          # Forbidden point (for renju)
    L_FLEX2 = 2         # F2+Any
    K_BLOCK3 = 3        # B3+Any
    J_FLEX2_2X = 4      # F2x2
    I_BLOCK3_PLUS = 5   # B3x2 | B3+F2
    H_FLEX3 = 6         # F3+Any
    G_FLEX3_PLUS = 7    # F3+F2 | F3+B3
    F_FLEX3_2X = 8      # F3x2
    E_BLOCK4 = 9        # B4+Any
    D_BLOCK4_PLUS = 10  # B4+F2 | B4+B3
    C_BLOCK4_FLEX3 = 11 # B4+F3
    B_FLEX4 = 12        # F4 | F4S | B4x2
    A_FIVE = 13         # F5
    # PATTERN4_NB is implicitly len(Pattern4)

PATTERN4_NB = 14

# -------------------------------------------------

# Integer value that representing the result of a search
# Using IntEnum so instances are also ints and arithmetic is natural
class Value(IntEnum):
    """Integer value that representing the result of a search"""
    VALUE_BLOCKED = -30003
    VALUE_NONE = -30002
    # VALUE_INFINITE should be greater than VALUE_MATE
    VALUE_MATE = 30000 
    VALUE_INFINITE = VALUE_MATE + 1 # Or 30001 as in Rapfi

    VALUE_ZERO = 0
    VALUE_DRAW = 0

    # VALUE_MATED_IN_MAX_PLY and VALUE_MATE_IN_MAX_PLY are conceptual bounds.
    # Actual mate scores are calculated relative to VALUE_MATE.
    # Rapfi: VALUE_MATE_IN_MAX_PLY = VALUE_MATE - 500
    # Rapfi: VALUE_MATED_IN_MAX_PLY = -VALUE_MATE + 500
    # These are more like thresholds for when a value is considered a "long mate/mated"
    # rather than specific enum members for all possible plies.
    # For now, keep the structure for compatibility with search_value_to_stored_value logic
    VALUE_MATED_IN_MAX_PLY = -VALUE_MATE + 500
    VALUE_MATE_IN_MAX_PLY = VALUE_MATE - 500
    
    VALUE_MATE_FROM_DATABASE = VALUE_MATE_IN_MAX_PLY # Alias
    VALUE_MATED_FROM_DATABASE = VALUE_MATED_IN_MAX_PLY # Alias

    VALUE_EVAL_MIN = -20000
    VALUE_EVAL_MAX = 20000


def mate_in(ply: int) -> int: # Return type is int
    """Construct an integer value for mate in N ply"""
    # return Value(Value.VALUE_MATE - ply) # Old incorrect way
    return Value.VALUE_MATE.value - ply

def mated_in(ply: int) -> int: # Return type is int
    """Construct an integer value for being mated in N ply"""
    # return Value(-Value.VALUE_MATE + ply) # Old incorrect way
    return -Value.VALUE_MATE.value + ply

def mate_step(v: int, ply: int) -> int: # v is now an int
    """Get number of steps to mate from value and current ply"""
    # VALUE_MATE - ply - abs(v)
    return Value.VALUE_MATE.value - ply - abs(v)

def search_value_to_stored_value(value: int, search_ply: int) -> int: # value is int
    """Converts a search value to a value storable in TT."""
    if value == Value.VALUE_NONE.value: # Compare with .value
        return Value.VALUE_NONE.value
    # Use the integer values for comparison
    elif value >= Value.VALUE_MATE_IN_MAX_PLY.value:
        return value + search_ply
    elif value <= Value.VALUE_MATED_IN_MAX_PLY.value:
        return value - search_ply
    else:
        return value

def stored_value_to_search_value(stored_value: int, search_ply: int) -> int: # Returns int
    """Converts a TT stored value back to a search value."""
    if stored_value == Value.VALUE_NONE.value:
        return Value.VALUE_NONE.value
    
    # Check against the conceptual bounds after potential ply adjustment
    # Original value if it was a mate: stored_value - search_ply
    # Original value if it was mated: stored_value + search_ply
    original_if_mate = stored_value - search_ply
    original_if_mated = stored_value + search_ply

    if original_if_mate >= Value.VALUE_MATE_IN_MAX_PLY.value:
        return original_if_mate
    elif original_if_mated <= Value.VALUE_MATED_IN_MAX_PLY.value:
        return original_if_mated
    # If not clearly a mate/mated after adjustment, or it's a normal eval
    return stored_value


# -------------------------------------------------

class Rule(Enum):
    """Rule is the fundamental rule of the game"""
    FREESTYLE = 0
    STANDARD = 1
    RENJU = 2
    # RULE_NB is implicitly len(Rule)

RULE_NB = 3

class OpeningRule(Enum):
    """Opening rule variants"""
    FREEOPEN = 0
    SWAP1 = 1
    SWAP2 = 2

class GameRule:
    """GameRule is composed of Rule and OpeningRule"""
    def __init__(self, rule: Rule, op_rule: OpeningRule):
        self.rule: Rule = rule
        self.op_rule: OpeningRule = op_rule

    def __int__(self) -> int: # To mimic `operator Rule() const` behavior if needed for indexing
        return self.rule.value

    def __eq__(self, other):
        if isinstance(other, GameRule):
            return self.rule == other.rule and self.op_rule == other.op_rule
        if isinstance(other, Rule): # Allow comparison like GameRule == Rule.RENJU
            return self.rule == other
        return False

    def __hash__(self):
        return hash((self.rule, self.op_rule))

class ActionType(Enum):
    """Thinking result action type"""
    MOVE = 0
    MOVE2 = 1
    SWAP = 2
    SWAP2_PUT_TWO = 3

# -------------------------------------------------

class CandidateRange(Enum):
    """CandidateRange represents the options for different range of move candidates."""
    SQUARE2 = 0
    SQUARE2_LINE3 = 1
    SQUARE3 = 2
    SQUARE3_LINE4 = 3
    SQUARE4 = 4
    FULL_BOARD = 5
    # CAND_RANGE_NB is implicitly len(CandidateRange)

CAND_RANGE_NB = 6

if __name__ == '__main__':
    # ... (existing tests for Color, Pattern, Pattern4, Bound are fine) ...

    print(f"Value.VALUE_MATE: {Value.VALUE_MATE.value}")
    print(f"Value.VALUE_MATE + 10: {Value.VALUE_MATE.value + 10}") # int op int
    
    mate_val = mate_in(10)
    print(f"mate_in(10): {mate_val} (type: {type(mate_val)})") # Should be int
    assert isinstance(mate_val, int)
    assert mate_val == Value.VALUE_MATE.value - 10

    mated_val = mated_in(10)
    print(f"mated_in(10): {mated_val} (type: {type(mated_val)})") # Should be int
    assert isinstance(mated_val, int)
    assert mated_val == -Value.VALUE_MATE.value + 10
    
    # Test search_value_to_stored_value and stored_value_to_search_value with ints
    ply_test = 10
    original_mate_val_int = mate_in(ply_test + 5) 
    print(f"Original mate value (int) at ply {ply_test}: {original_mate_val_int}")
    stored_val_int = search_value_to_stored_value(original_mate_val_int, ply_test)
    print(f"Stored value (int): {stored_val_int}")
    retrieved_val_int = stored_value_to_search_value(stored_val_int, ply_test)
    print(f"Retrieved value (int): {retrieved_val_int}")
    assert original_mate_val_int == retrieved_val_int

    original_eval_val_int = 100 # Plain int for eval
    print(f"Original eval value (int): {original_eval_val_int}")
    stored_eval_int = search_value_to_stored_value(original_eval_val_int, ply_test)
    print(f"Stored eval (int): {stored_eval_int}") # Eval scores are not ply-adjusted
    retrieved_eval_int = stored_value_to_search_value(stored_eval_int, ply_test)
    print(f"Retrieved eval (int): {retrieved_eval_int}")
    assert original_eval_val_int == retrieved_eval_int
    assert stored_eval_int == original_eval_val_int # For normal evals

    original_mated_val_int = mated_in(ply_test + 7)
    print(f"Original mated value (int) at ply {ply_test}: {original_mated_val_int}")
    stored_mated_int = search_value_to_stored_value(original_mated_val_int, ply_test)
    print(f"Stored mated (int): {stored_mated_int}")
    retrieved_mated_int = stored_value_to_search_value(stored_mated_int, ply_test)
    print(f"Retrieved mated (int): {retrieved_mated_int}")
    assert original_mated_val_int == retrieved_mated_int

    print("Value.BOUND_EXACT:", Bound.BOUND_EXACT)
    print("Value.BOUND_LOWER in Value.BOUND_EXACT:", Bound.BOUND_LOWER in Bound.BOUND_EXACT)