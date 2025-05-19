"""
Helper classes (CandArea, StateInfo, Cell) used by the Board class,
based on Rapfi's board.h.
"""
import sys
from typing import List, Tuple, cast

from .types import (Pattern, Pattern4, Color, Rule, PatternCode, Score, Value,
                    SIDE_NB, PATTERN_NB, PATTERN4_NB) # EVAL_PCODE_NB for EvalInfo
from .config import PCODE_NB as EVAL_PCODE_NB
from .pos import Pos, Direction, FULL_BOARD_CELL_COUNT # Assuming pos.py defines these
from .pattern_utils import Pattern2x, lookup_pattern_from_luts, get_pcode_from_patterns # For Cell methods
from . import config as engine_config # For P4SCORES, getValueBlack, config.PCODE_NB

# Constants from board.h or implied
# MAX_BOARD_SIZE is in pos.py
INT8_MAX = 127
INT8_MIN = -128

class CandArea:
    """
    Represents a rectangle area on board which can be considered for move candidates.
    Mirrors Rapfi's CandArea struct.
    Coordinates are 0-indexed relative to the playable board_size.
    """
    def __init__(self, x0: int = INT8_MAX, y0: int = INT8_MAX,
                 x1: int = INT8_MIN, y1: int = INT8_MIN):
        self.x0: int = x0
        self.y0: int = y0
        self.x1: int = x1
        self.y1: int = y1

    def expand(self, pos: Pos, board_size: int, dist: int):
        """
        Expands the candidate area to include a square region around `pos`.
        `pos` provides 0-indexed x, y coordinates.
        `board_size` is the dimension of the playable area (e.g., 15 for a 15x15 board).
        `dist` is the half-width of the square to include (e.g., dist=2 means a 5x5 area).
        """
        if pos == Pos.PASS or pos == Pos.NONE: # Cannot expand around PASS/NONE
            return

        px, py = pos.x, pos.y

        # Clamp coordinates to be within the board [0, board_size - 1]
        self.x0 = min(self.x0, max(px - dist, 0))
        self.y0 = min(self.y0, max(py - dist, 0))
        self.x1 = max(self.x1, min(px + dist, board_size - 1))
        self.y1 = max(self.y1, min(py + dist, board_size - 1))

    def is_empty(self) -> bool:
        """Checks if the area is uninitialized or invalid."""
        return self.x0 > self.x1 or self.y0 > self.y1

    def __repr__(self):
        if self.is_empty():
            return "CandArea(empty)"
        return f"CandArea(x0={self.x0}, y0={self.y0}, x1={self.x1}, y1={self.y1})"

class StateInfo:
    """
    Records all incremental board information used in one ply.
    Mirrors Rapfi's StateInfo struct.
    """
    def __init__(self):
        self.cand_area: CandArea = CandArea()
        self.last_move: Pos = Pos.NONE

        # lastFlex4AttackMove[color_value] -> Pos
        self.last_flex4_attack_move: List[Pos] = [Pos.NONE] * SIDE_NB

        # lastPattern4Move[color_value][pattern4_index_offset] -> Pos
        # Pattern4 index offset: A_FIVE-C_BLOCK4_FLEX3, B_FLEX4-C_BLOCK4_FLEX3, C_BLOCK4_FLEX3-C_BLOCK4_FLEX3
        # These are for C_BLOCK4_FLEX3, B_FLEX4, A_FIVE. So size is 3.
        self.last_pattern4_move: List[List[Pos]] = [[Pos.NONE] * 3 for _ in range(SIDE_NB)]

        # p4Count[color_value][pattern4_enum_value] -> count
        self.p4_count: List[List[int]] = [[0] * PATTERN4_NB for _ in range(SIDE_NB)]

        # valueBlack stores the classical evaluation score from Black's perspective (as int)
        self.value_black: int = 0

    def get_last_pattern4(self, side: Color, p4: Pattern4) -> Pos:
        """
        Queries the last emerged pattern4 pos.
        p4 must be one of [C_BLOCK4_FLEX3, B_FLEX4, A_FIVE].
        """
        # Map Pattern4 enum to the 0-2 index used in last_pattern4_move
        # C_BLOCK4_FLEX3 is the base for indexing in Rapfi
        if not (Pattern4.C_BLOCK4_FLEX3.value <= p4.value <= Pattern4.A_FIVE.value):
            # print(f"Warning: Invalid Pattern4 '{p4.name}' for get_last_pattern4.", file=sys.stderr)
            return Pos.NONE
        
        index_offset = p4.value - Pattern4.C_BLOCK4_FLEX3.value
        if not (0 <= index_offset < 3):
             # This should not happen if the above check is correct and enum values are contiguous
            # print(f"Warning: Calculated index_offset {index_offset} is out of bounds for Pattern4 '{p4.name}'.", file=sys.stderr)
            return Pos.NONE
            
        return self.last_pattern4_move[side.value][index_offset]

    def __deepcopy__(self, memo): # For proper copying if needed by board.clone()
        new_si = StateInfo()
        new_si.cand_area = CandArea(self.cand_area.x0, self.cand_area.y0, self.cand_area.x1, self.cand_area.y1)
        new_si.last_move = self.last_move # Pos is immutable enough
        new_si.last_flex4_attack_move = list(self.last_flex4_attack_move)
        new_si.last_pattern4_move = [list(sub) for sub in self.last_pattern4_move]
        new_si.p4_count = [list(sub) for sub in self.p4_count]
        new_si.value_black = self.value_black
        return new_si


class Cell:
    """
    Contains all information for a move cell on board.
    Mirrors Rapfi's Cell struct.
    """
    def __init__(self, piece: Color = Color.EMPTY):
        self.piece: Color = piece
        self.cand: int = 0 # uint8_t in C++, candidate counter

        # pattern4[color_value] -> Pattern4 enum
        self.pattern4: List[Pattern4] = [Pattern4.NONE] * SIDE_NB
        
        # score[color_value] -> int (Score type)
        self.score: List[int] = [0] * SIDE_NB
        
        # valueBlack stores the classical evaluation component for this cell, from Black's perspective
        self.value_black: int = 0 # Corresponds to Value type in C++, stores int

        # pattern2x[direction_index for 4 main directions] -> Pattern2x object
        self.pattern2x: List[Pattern2x] = [Pattern2x() for _ in range(4)] # 4 main directions

    def is_candidate(self) -> bool:
        """Checks if this cell is a move candidate."""
        return self.cand > 0

    def get_line_pattern(self, color: Color, dir_idx: int) -> Pattern:
        """Gets the line pattern for a given color and direction index (0-3)."""
        if not (0 <= dir_idx < 4):
            raise IndexError(f"Direction index {dir_idx} out of range 0-3.")
        return self.pattern2x[dir_idx][color]

    def get_pcode(self, color: Color) -> PatternCode:
        """
        Gets the compressed PatternCode for this cell from the perspective of `color`.
        This involves combining the 4 line patterns for `color`.
        """
        # pattern2x stores Pattern objects for Black and White for each of 4 dirs
        # We need to extract the pattern for the given `color` for each of the 4 directions
        p1 = self.pattern2x[0][color] # Pattern for `color` in direction 0
        p2 = self.pattern2x[1][color] # Pattern for `color` in direction 1
        p3 = self.pattern2x[2][color] # Pattern for `color` in direction 2
        p4 = self.pattern2x[3][color] # Pattern for `color` in direction 3
        
        return get_pcode_from_patterns(p1, p2, p3, p4)

    def update_pattern4_and_score(self, rule: Rule, pcode_black: PatternCode, pcode_white: PatternCode):
        """
        Updates the Pattern4 and score for both Black and White based on their pcodes at this cell.
        Uses engine_config.get_p4score.
        """
        p4score_black = engine_config.get_p4score(rule, Color.BLACK, pcode_black)
        p4score_white = engine_config.get_p4score(rule, Color.WHITE, pcode_white)

        self.pattern4[Color.BLACK.value] = p4score_black.pattern4
        self.pattern4[Color.WHITE.value] = p4score_white.pattern4

        self.score[Color.BLACK.value] = p4score_black.score_self() + p4score_white.score_oppo()
        self.score[Color.WHITE.value] = p4score_white.score_self() + p4score_black.score_oppo()

    def __repr__(self):
        return (f"Cell(piece={self.piece.name}, cand={self.cand}, "
                f"p4B={self.pattern4[Color.BLACK.value].name}, p4W={self.pattern4[Color.WHITE.value].name}, "
                f"sB={self.score[Color.BLACK.value]}, sW={self.score[Color.WHITE.value]}, "
                f"valB={self.value_black})")


if __name__ == '__main__':
    print("--- Board Utils Tests ---")

    # Test CandArea
    ca = CandArea()
    print(f"Initial CandArea: {ca}, is_empty: {ca.is_empty()}")
    assert ca.is_empty()
    ca.expand(Pos(7,7), 15, 2) # Expand around (7,7) on a 15x15 board, dist 2
    print(f"CandArea after expand(Pos(7,7), 15, 2): {ca}")
    # Expected: x0=5, y0=5, x1=9, y1=9
    assert ca.x0 == 5 and ca.y0 == 5 and ca.x1 == 9 and ca.y1 == 9
    ca.expand(Pos(0,0), 15, 1)
    print(f"CandArea after expand(Pos(0,0), 15, 1): {ca}")
    # Expected: x0=0, y0=0, x1=9, y1=9
    assert ca.x0 == 0 and ca.y0 == 0 and ca.x1 == 9 and ca.y1 == 9
    assert not ca.is_empty()

    # Test StateInfo
    si = StateInfo()
    print(f"Initial StateInfo last_move: {si.last_move.x},{si.last_move.y}") # Pos.NONE
    assert si.last_move == Pos.NONE
    print(f"Initial StateInfo p4_count[B][A_FIVE]: {si.p4_count[Color.BLACK.value][Pattern4.A_FIVE.value]}")
    assert si.p4_count[Color.BLACK.value][Pattern4.A_FIVE.value] == 0
    pos_test = Pos(1,1)
    si.last_pattern4_move[Color.BLACK.value][Pattern4.A_FIVE.value - Pattern4.C_BLOCK4_FLEX3.value] = pos_test
    ret_pos = si.get_last_pattern4(Color.BLACK, Pattern4.A_FIVE)
    print(f"Retrieved last_pattern4 for A_FIVE: {ret_pos}")
    assert ret_pos == pos_test
    # Test invalid get_last_pattern4
    ret_invalid = si.get_last_pattern4(Color.BLACK, Pattern4.FORBID)
    assert ret_invalid == Pos.NONE


    # Test Cell
    cell = Cell(Color.BLACK)
    cell.cand = 1
    print(f"Initial Cell: {cell}")
    assert cell.piece == Color.BLACK
    assert cell.is_candidate()

    # Mock pcode and update (Pattern4Score scores are still dummy 0,0 from config)
    # This requires PCODE_LUT and P4SCORES to be somewhat initialized from pattern_utils and config
    # Let's assume pattern_utils.init_pattern_config() has run.
    # We need to ensure engine_config.P4SCORES is populated for the test.
    # For test, let's mock engine_config.get_p4score to return something predictable.
    
    original_get_p4score = engine_config.get_p4score
    def mock_get_p4score(rule, color, pcode):
        if color == Color.BLACK and pcode == 10: # Dummy pcode
            return engine_config.Pattern4Score(Pattern4.B_FLEX4, 100, -10)
        if color == Color.WHITE and pcode == 20: # Dummy pcode
            return engine_config.Pattern4Score(Pattern4.H_FLEX3, 50, -5)
        return engine_config.Pattern4Score() # Default
    engine_config.get_p4score = mock_get_p4score

    cell.update_pattern4_and_score(Rule.FREESTYLE, 10, 20) # pcode_black=10, pcode_white=20
    print(f"Cell after update_pattern4_and_score: {cell}")
    assert cell.pattern4[Color.BLACK.value] == Pattern4.B_FLEX4
    assert cell.pattern4[Color.WHITE.value] == Pattern4.H_FLEX3
    assert cell.score[Color.BLACK.value] == (100 + (-5)) # p4s_B.self + p4s_W.oppo
    assert cell.score[Color.WHITE.value] == (50 + (-10)) # p4s_W.self + p4s_B.oppo
    
    engine_config.get_p4score = original_get_p4score # Restore

    # Test Cell.get_pcode (requires PCODE_LUT from pattern_utils)
    # This is more of an integration test. For now, just call it.
    try:
        # Example: all lines are DEAD pattern
        cell.pattern2x = [Pattern2x(Pattern.DEAD, Pattern.DEAD)] * 4
        pcode_b = cell.get_pcode(Color.BLACK)
        pcode_w = cell.get_pcode(Color.WHITE)
        print(f"PCode for all DEAD lines (Black): {pcode_b}")
        print(f"PCode for all DEAD lines (White): {pcode_w}")
        # Expected: pcode for (DEAD,DEAD,DEAD,DEAD) should be the same for B and W
        # and should be a valid pcode (e.g., 0 if it's the first unique combo)
        assert pcode_b == pcode_w 
        assert pcode_b >= 0 and pcode_b < engine_config.PCODE_NB # PCODE_NB from config
    except Exception as e:
        print(f"Error testing Cell.get_pcode (likely PCODE_LUT not fully ready for standalone test): {e}")


    print("board_utils.py tests completed.")