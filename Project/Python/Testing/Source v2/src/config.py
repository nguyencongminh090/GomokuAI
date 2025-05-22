from .types import Pattern4, Value, Rule, Color, SIDE_NB, PatternCode # Assuming types.py is in the same package
"""
Configuration constants and structures for the Gomoku engine,
based on Rapfi's config.h and config.cpp.
"""
import math
from enum import Enum, IntEnum
from typing import List, Tuple, Callable, Optional, Dict, Any
import sys # For potentially accessing command line args or environment vars later

# Assuming these are in the same package
from .types import (Pattern, Pattern4, Value, Rule, Color, CandidateRange,
                    SIDE_NB, PATTERN_NB, RULE_NB)


# --- Engine Information (added as per search.py usage) ---
ENGINE_NAME: str = "PyRapfi"
ENGINE_AUTHOR: str = "AI Agent & User"


# PCODE_NB and THREAT_NB are defined in this file as per C++ config.h
# UNIQUE_PCODE_COUNT will be determined by pattern_utils.py and used to size P4SCORES dynamically

# --- Constants from config.h ---
# PCODE_NB: Number of unique pattern codes for 4 directions (Pattern Code)
# Rapfi: constexpr uint32_t PCODE_NB = combineNumber(PATTERN_NB, 4);
# This uses the combine_number utility. PATTERN_NB is 14 from types.py
# H(k,n) = C(n+k-1, k) -> H(4, 14) = C(14+4-1, 4) = C(17, 4)
# C(17,4) = (17 * 16 * 15 * 14) / (4 * 3 * 2 * 1) = 17 * 2 * 5 * 7 = 2380
PCODE_NB: int = 2380 # Calculated as combineNumber(PATTERN_NB, 4)

# THREAT_NB: Number of threat mask combinations
# Rapfi: constexpr uint32_t THREAT_NB = power(2, 11);
THREAT_NB: int = 1 << 11 # 2048

class MsgMode(Enum):
    NONE = 0
    BRIEF = 1
    NORMAL = 2
    UCILIKE = 3

class CoordConvertionMode(Enum):
    NONE = 0
    X_FLIPY = 1
    FLIPY_X = 2

class Pattern4Score:
    """
    Represents the Pattern4 type and associated scores.
    Mirrors Rapfi's Pattern4Score struct.
    _pattern4 stores the Pattern4 enum.
    _score_self and _score_oppo are integers.
    """
    def __init__(self, pattern4: Pattern4 = Pattern4.NONE, score_self: int = 0, score_oppo: int = 0):
        self._pattern4: Pattern4 = pattern4
        # Rapfi: _scoreSelf : 14; _scoreOppo : 14; (bitfields for int32_t)
        # Python ints handle arbitrary size, so we don't need bitfields for storage.
        # Clamping might be needed if specific ranges are enforced.
        self._score_self: int = score_self
        self._score_oppo: int = score_oppo

    @property
    def pattern4(self) -> Pattern4:
        return self._pattern4

    @pattern4.setter
    def pattern4(self, value: Pattern4):
        self._pattern4 = value

    def score_self(self) -> int:
        return self._score_self

    def set_score_self(self, score: int):
        self._score_self = score

    def score_oppo(self) -> int:
        return self._score_oppo

    def set_score_oppo(self, score: int):
        self._score_oppo = score
    
    # Rapfi uses operator[] for score access
    def get_score(self, is_opponent_score: bool) -> int:
        return self._score_oppo if is_opponent_score else self._score_self

    def set_score(self, is_opponent_score: bool, score: int):
        if is_opponent_score:
            self._score_oppo = score
        else:
            self._score_self = score

    def __int__(self) -> int: # Allows casting to int, returning Pattern4 enum value
        return self._pattern4.value

    def __eq__(self, other):
        if isinstance(other, Pattern4Score):
            return self._pattern4 == other._pattern4 and \
                   self._score_self == other._score_self and \
                   self._score_oppo == other._score_oppo
        if isinstance(other, Pattern4):
            return self._pattern4 == other
        if isinstance(other, int):
            return self._pattern4.value == other # Compare enum value
        return False
    
    def __repr__(self):
        return f"P4S({self._pattern4.name}, self:{self._score_self}, oppo:{self._score_oppo})"

# --- Global Configuration Variables (mirrors externs in config.h) ---

# Search options
DEFAULT_SEARCHER_NAME: str = "alphabeta"
ASPIRATION_WINDOW: bool = True  # Whether to enable aspiration window logic at all
ASPIRATION_MIN_DEPTH: int = 3   # Minimum depth to start using aspiration windows (if enabled) # ADD THIS LINE
FILTER_SYMMETRY_ROOT_MOVES: bool = True

# Model configs
SCALING_FACTOR: float = 200.0
EVALUATOR_MARGIN_WIN_LOSS_SCALE: float = 1.18
EVALUATOR_MARGIN_WIN_LOSS_EXPONENT: float = 3.07
EVALUATOR_MARGIN_SCALE: float = 395.0
EVALUATOR_DRAW_BLACK_WIN_RATE: float = 0.5
EVALUATOR_DRAW_RATIO: float = 1.0

# Classical evaluation and score tables
# Dimensions: [RULE_NB + 1][PCODE_NB] or [RULE_NB + 1][THREAT_NB]
# RULE_NB + 1 is because Renju has separate tables for Black and White sometimes.
# tableIndex(Rule r, Color c) maps (Rule, Color) to an index 0..RULE_NB
# For Renju, Black maps to Rule.RENJU.value, White maps to Rule.RENJU.value + 1 (if different)
# Or more consistently, table_idx = r.value + (1 if r == Rule.RENJU and c == Color.WHITE else 0)
# Let's use RULE_NB + 1 = 4 as the size for the first dimension of these tables.
# Index 0: Freestyle, 1: Standard, 2: Renju (Black perspective), 3: Renju (White perspective)

EVAL_TABLE_DIM1_SIZE = RULE_NB + 1 # Covers Freestyle, Standard, Renju_Black, Renju_White (if needed)

# Initialize with defaults
EVALS: List[List[int]] = [[0] * PCODE_NB for _ in range(EVAL_TABLE_DIM1_SIZE)]
EVALS_THREAT: List[List[int]] = [[0] * THREAT_NB for _ in range(EVAL_TABLE_DIM1_SIZE)]
P4SCORES: List[List[Pattern4Score]] = [
    [Pattern4Score(Pattern4.NONE, 0, 0) for _ in range(PCODE_NB)] # Initialize with default Pattern4.NONE
    for _ in range(EVAL_TABLE_DIM1_SIZE)
]

# General options
RELOAD_CONFIG_EACH_MOVE: bool = False
CLEAR_HASH_AFTER_CONFIG_LOADED: bool = True
DEFAULT_THREAD_NUM: int = 1 # Python is GIL-limited for CPU-bound tasks, but can be >1 for I/O or multiprocessing
MESSAGE_MODE: MsgMode = MsgMode.BRIEF
IO_COORD_MODE: CoordConvertionMode = CoordConvertionMode.NONE
DEFAULT_CANDIDATE_RANGE: CandidateRange = CandidateRange.SQUARE3_LINE4
MEMORY_RESERVED_MB: List[int] = [0] * RULE_NB # Size per rule
DEFAULT_TT_SIZE_KB: int = 0

# Search options
DEFAULT_SEARCHER_NAME: str = "alphabeta"
ASPIRATION_WINDOW: bool = True
FILTER_SYMMETRY_ROOT_MOVES: bool = True
NUM_ITERATION_AFTER_MATE: int = 6
NUM_ITERATION_AFTER_SINGULAR_ROOT: int = 4
MAX_SEARCH_DEPTH: int = 99

# MCTS specific (mostly placeholders if not implementing MCTS soon)
EXPAND_WHEN_FIRST_EVALUATE: bool = False
MAX_NUM_VISITS_PER_PLAYOUT: int = 100
NODES_TO_PRINT_MCTS_ROOTMOVES: int = 0
TIME_TO_PRINT_MCTS_ROOTMOVES: int = 1000
MAX_NON_PV_ROOTMOVES_TO_PRINT: int = 10
NUM_NODES_AFTER_SINGULAR_ROOT_MCTS: int = 100 # Renamed from NumNodesAfterSingularRoot to avoid clash
NUM_NODE_TABLE_SHARDS_POWER_OF_TWO: int = 10
DRAW_UTILITY_PENALTY_MCTS: float = 0.35 # Renamed

# Time management options
TURN_TIME_RESERVED: int = 30
MATCH_SPACE: float = 22.0
MATCH_SPACE_MIN: float = 7.0
AVERAGE_BRANCH_FACTOR: float = 1.7
ADVANCED_STOP_RATIO: float = 0.9
MOVE_HORIZON: int = 64
TIME_DIVISOR_BIAS: float = 1.25
TIME_DIVISOR_SCALE: float = 0.02
TIME_DIVISOR_DEPTH_POW: float = 1.4
FALLING_FACTOR_SCALE: float = 0.0032
FALLING_FACTOR_BIAS: float = 0.544
BESTMOVE_STABLE_REDUCTION_SCALE: float = 0.0125
BESTMOVE_STABLE_PREV_REDUCTION_POW: float = 0.528

# Database options (placeholders, complex to port fully without DB libraries)
class DatabaseOverwriteRule(Enum): # From Database::OverwriteRule
    BetterValueDepthBound = 0
    BetterDepthBound = 1
    BetterValue = 2
    BetterLabel = 3
    Always = 4
    Disabled = 5

DATABASE_DEFAULT_ENABLED: bool = False
DATABASE_LEGACY_FILE_CODE_PAGE: int = 0 # uint16_t in C++
DATABASE_TYPE: str = ""
DATABASE_URL: str = ""
DATABASE_CACHE_SIZE: int = 4096
DATABASE_RECORD_CACHE_SIZE: int = 32768
DATABASE_MAKER: Optional[Callable[[str], Any]] = None # Placeholder for factory

DATABASE_LIB_BLACK_WIN_MARK: str = 'a'
DATABASE_LIB_WHITE_WIN_MARK: str = 'a'
DATABASE_LIB_BLACK_LOSE_MARK: str = 'c'
DATABASE_LIB_WHITE_LOSE_MARK: str = 'c'
DATABASE_LIB_IGNORE_COMMENT: bool = False
DATABASE_LIB_IGNORE_BOARD_TEXT: bool = False

DATABASE_READONLY_MODE: bool = False
DATABASE_MANDATORY_PARENT_WRITE: bool = True
DATABASE_QUERY_PLY: int = 3
DATABASE_QUERY_PV_ITER_PER_PLY_INCREMENT: int = 1
DATABASE_QUERY_NON_PV_ITER_PER_PLY_INCREMENT: int = 2
DATABASE_PV_WRITE_PLY: int = 1
DATABASE_PV_WRITE_MIN_DEPTH: int = 25
DATABASE_NON_PV_WRITE_PLY: int = 0
DATABASE_NON_PV_WRITE_MIN_DEPTH: int = 25
DATABASE_WRITE_VALUE_RANGE: int = 800
DATABASE_MATE_WRITE_PLY: int = 2
DATABASE_MATE_WRITE_MIN_DEPTH_EXACT: int = 20
DATABASE_MATE_WRITE_MIN_DEPTH_NON_EXACT: int = 40
DATABASE_MATE_WRITE_MIN_STEP: int = 10
DATABASE_EXACT_OVERWRITE_PLY: int = 100
DATABASE_NON_EXACT_OVERWRITE_PLY: int = 0
DATABASE_OVERWRITE_RULE: DatabaseOverwriteRule = DatabaseOverwriteRule.BetterValueDepthBound
DATABASE_OVERWRITE_EXACT_BIAS: int = 3
DATABASE_OVERWRITE_DEPTH_BOUND_BIAS: int = -1
DATABASE_QUERY_RESULT_DEPTH_BOUND_BIAS: int = 0


# --- Helper functions from Config namespace ---
def table_index(r: Rule, c: Color) -> int:
    """
    Calculates the index for EVALS, P4SCORES, etc.
    Freestyle: 0
    Standard: 1
    Renju Black: 2 (Rule.RENJU.value)
    Renju White: 3 (Rule.RENJU.value + 1, if different treatment)
    In Rapfi config.cpp, EVALS[r + BLACK] and EVALS[r + WHITE] are used for Renju.
    P4SCORES[r + BLACK] and P4SCORES[r + WHITE] for Renju.
    Otherwise, just P4SCORES[r].
    Let's map: Index 0 (FS), 1 (STD), 2 (RenjuB), 3 (RenjuW)
    """
    if r == Rule.RENJU:
        return r.value + c.value # RENJU.value=2. BLACK.value=0, WHITE.value=1 -> 2, 3
    return r.value # FS=0, STD=1. Color doesn't change index for these.

def get_value_black(rule: Rule, pcode_black: PatternCode, pcode_white: PatternCode) -> int:
    """
    Lookup eval table for Black's perspective.
    Returns an int.
    """
    idx_black_persp = table_index(rule, Color.BLACK)
    idx_white_persp = table_index(rule, Color.WHITE) # For Renju, this accesses White's specific table
                                                    # For FS/STD, this will be same as idx_black_persp

    eval_for_black_patterns = EVALS[idx_black_persp][pcode_black]
    
    # If white's patterns are evaluated from white's perspective in EVALS[idx_white_persp]
    # then we need to negate it for black's total perspective.
    # If EVALS always stores from "table's color" perspective:
    # EVALS[table_index(rule, WHITE)] stores eval from WHITE's perspective for pcodeWhite
    eval_for_white_patterns_from_white_view = EVALS[idx_white_persp][pcode_white]
    
    return eval_for_black_patterns - eval_for_white_patterns_from_white_view


def get_p4score(rule: Rule, color: Color, pcode: PatternCode) -> Pattern4Score:
    """Lookup Pattern4Score table."""
    idx = table_index(rule, color)
    # Ensure pcode is within bounds of the P4SCORES table for that index
    if 0 <= idx < len(P4SCORES) and 0 <= pcode < len(P4SCORES[idx]):
        return P4SCORES[idx][pcode]
    # print(f"Warning: P4Score lookup out of bounds. Rule: {rule}, Color: {color}, pcode: {pcode}, Index: {idx}")
    return Pattern4Score() # Default empty score

def value_to_win_rate(eval_score: int, strict: bool = True) -> float:
    """Converts evaluation score to winning rate [0, 1]."""
    min_eval = Value.VALUE_MATED_IN_MAX_PLY.value if strict else Value.VALUE_EVAL_MIN.value
    max_eval = Value.VALUE_MATE_IN_MAX_PLY.value if strict else Value.VALUE_EVAL_MAX.value

    if eval_score >= max_eval: return 1.0
    if eval_score <= min_eval: return 0.0
    
    # Sigmoid function
    try:
        return 1.0 / (1.0 + math.exp(-float(eval_score) / SCALING_FACTOR))
    except OverflowError: # exp result too large
        return 1.0 if float(eval_score) > 0 else 0.0


def win_rate_to_value(win_rate: float) -> int:
    """Converts winning rate [0, 1] to an evaluation score."""
    if win_rate >= 1.0: return Value.VALUE_EVAL_MAX.value # Or MATE_IN_MAX_PLY depending on context
    if win_rate <= 0.0: return Value.VALUE_EVAL_MIN.value # Or MATED_IN_MAX_PLY

    # Ensure win_rate is slightly away from 0 and 1 to avoid log(0) or division by zero
    epsilon = 1e-9
    win_rate = max(epsilon, min(win_rate, 1.0 - epsilon))

    value_f32 = SCALING_FACTOR * math.log(win_rate / (1.0 - win_rate))
    
    clamped_value = max(Value.VALUE_EVAL_MIN.value, min(value_f32, Value.VALUE_EVAL_MAX.value))
    return int(round(clamped_value))


# --- Placeholder for config loading functions ---
def load_config_from_toml_stream(stream) -> bool:
    """Placeholder for loading config from a TOML stream."""
    # This would involve a TOML parser and populating the globals above.
    print("Warning: load_config_from_toml_stream is not yet implemented.")
    return False

def load_model_from_stream(stream) -> bool:
    """Placeholder for loading classical eval model from binary stream."""
    # This would involve binary parsing.
    print("Warning: load_model_from_stream is not yet implemented.")
    return False

def export_model_to_stream(stream):
    """Placeholder for exporting classical eval model to binary stream."""
    print("Warning: export_model_to_stream is not yet implemented.")

# --- NEW SECTION: Manual Population of Evaluation Tables ---

def _populate_heuristic_eval_tables():
    """
    Populates EVALS and P4SCORES with simplified heuristic values.
    This function should be called AFTER pattern_utils.init_pattern_config() has run,
    because it relies on engine_config.P4SCORES being filled with the correct Pattern4 types
    by _fill_p4scores_luts_in_config() (which is called by init_pattern_config).
    """
    if not P4SCORES or not P4SCORES[0] or P4SCORES[0][0].pattern4 == Pattern4.NONE and P4SCORES[0][0].score_self() == 0 :
        # This check suggests P4SCORES might not have been fully processed by pattern_utils
        # (i.e., the Pattern4 enum types are not set yet for each pcode).
        # This function relies on P4SCORES[idx][pcode].pattern4 being correctly set.
        # print("Warning: _populate_heuristic_eval_tables called before P4SCORES properly initialized by pattern_utils. Skipping.", file=sys.stderr)
        # For safety, we can try to call the pattern util init here, though it should be on import.
        try:
            from . import pattern_utils # pylint: disable=import-outside-toplevel
            if not pattern_utils.PCODE_LUT: # If pattern_utils hasn't run its init
                 pattern_utils.init_pattern_config()
        except ImportError:
            pass # Cannot do much if it cannot be imported

    # Define base scores for different Pattern4 types (from perspective of the player making the pattern)
    # These are heuristic values and can be tuned.
    # Positive for good patterns, negative for opponent's good patterns seen from our cell.
    pattern4_base_eval_scores: Dict[Pattern4, int] = {
        Pattern4.A_FIVE: 10000,        # Making a five
        Pattern4.B_FLEX4: 5000,         # Making an open four
        Pattern4.C_BLOCK4_FLEX3: 2000,  # Making a B4+F3
        Pattern4.D_BLOCK4_PLUS: 1500,   # B4 + (F2 or B3)
        Pattern4.E_BLOCK4: 1000,        # Making a B4 (closed four)
        Pattern4.F_FLEX3_2X: 800,       # Double open three
        Pattern4.G_FLEX3_PLUS: 600,     # F3 + (F2 or B3)
        Pattern4.H_FLEX3: 500,          # Single open three
        Pattern4.I_BLOCK3_PLUS: 300,    # B3x2 or B3+F2
        Pattern4.J_FLEX2_2X: 150,       # Double F2
        Pattern4.K_BLOCK3: 100,         # Single B3 (closed three)
        Pattern4.L_FLEX2: 50,           # Single F2
        Pattern4.FORBID: -500,          # Occupying a forbidden point (bad for black in renju)
        Pattern4.NONE: 0
    }
    
    # Heuristic scores for P4SCORES (used for move ordering, slightly different scale)
    pattern4_heuristic_scores_self: Dict[Pattern4, int] = {
        Pattern4.A_FIVE: 2000, Pattern4.B_FLEX4: 1000, Pattern4.C_BLOCK4_FLEX3: 800,
        Pattern4.D_BLOCK4_PLUS: 700, Pattern4.E_BLOCK4: 600, Pattern4.F_FLEX3_2X: 500,
        Pattern4.G_FLEX3_PLUS: 450, Pattern4.H_FLEX3: 400, Pattern4.I_BLOCK3_PLUS: 300,
        Pattern4.J_FLEX2_2X: 200, Pattern4.K_BLOCK3: 150, Pattern4.L_FLEX2: 100,
        Pattern4.FORBID: 10, # Forbid itself might be a "good" defensive move sometimes
        Pattern4.NONE: 0
    }
    # Score for opponent if *they* form a pattern at *our* candidate cell
    pattern4_heuristic_scores_oppo: Dict[Pattern4, int] = {
        Pattern4.A_FIVE: -1800, Pattern4.B_FLEX4: -900, Pattern4.C_BLOCK4_FLEX3: -700,
        Pattern4.D_BLOCK4_PLUS: -600, Pattern4.E_BLOCK4: -500, Pattern4.F_FLEX3_2X: -400,
        Pattern4.G_FLEX3_PLUS: -350, Pattern4.H_FLEX3: -300, Pattern4.I_BLOCK3_PLUS: -200,
        Pattern4.J_FLEX2_2X: -100, Pattern4.K_BLOCK3: -80, Pattern4.L_FLEX2: -40,
        Pattern4.FORBID: 0, # No direct opponent score for us occupying their forbid point
        Pattern4.NONE: 0
    }


    for rule_val in range(RULE_NB): # FS, STD, RENJU
        for pcode in range(PCODE_NB):
            # For EVALS table (static evaluation component)
            # This needs to combine effect for Black and White
            # get_p4score(rule, color, pcode) returns Pattern4Score which has .pattern4
            
            # Eval from Black's perspective at this empty cell:
            p4_black_at_cell = get_p4score(Rule(rule_val), Color.BLACK, pcode).pattern4
            p4_white_at_cell = get_p4score(Rule(rule_val), Color.WHITE, pcode).pattern4
            
            eval_black_contrib = pattern4_base_eval_scores.get(p4_black_at_cell, 0)
            eval_white_contrib = pattern4_base_eval_scores.get(p4_white_at_cell, 0)

            # EVALS[table_idx][pcode] stores score from perspective of 'table_idx' player
            # table_idx 0 (FS), 1 (STD) - perspective is effectively BLACK
            # table_idx 2 (RenjuBlack) - perspective is BLACK
            # table_idx 3 (RenjuWhite) - perspective is WHITE

            # For FS and STD (symmetric, effectively Black's view for the table)
            if Rule(rule_val) == Rule.FREESTYLE or Rule(rule_val) == Rule.STANDARD:
                idx = table_index(Rule(rule_val), Color.BLACK) # Should be rule_val
                EVALS[idx][pcode] = eval_black_contrib - eval_white_contrib
            
            # For Renju (asymmetric)
            if Rule(rule_val) == Rule.RENJU:
                # Renju Black's table (idx 2)
                idx_rb = table_index(Rule.RENJU, Color.BLACK)
                # If Black makes p4_black_at_cell, and White would make p4_white_at_cell
                # Score for Black is black_contrib - white_contrib (benefit - opponent's counter-benefit)
                # Special handling for FORBID for black in Renju
                if p4_black_at_cell == Pattern4.FORBID:
                    eval_black_contrib_renju = pattern4_base_eval_scores.get(Pattern4.FORBID,0)
                else:
                    eval_black_contrib_renju = eval_black_contrib
                EVALS[idx_rb][pcode] = eval_black_contrib_renju - eval_white_contrib

                # Renju White's table (idx 3)
                idx_rw = table_index(Rule.RENJU, Color.WHITE)
                # If White makes p4_white_at_cell, and Black would make p4_black_at_cell
                # Score for White is white_contrib - black_contrib
                # Black's FORBID is not a penalty *for white's eval* if white plays there
                EVALS[idx_rw][pcode] = eval_white_contrib - eval_black_contrib


            # For P4SCORES (heuristic scores for move ordering)
            # P4SCORES[table_idx][pcode] stores Pattern4Score object
            # We need to set _scoreSelf and _scoreOppo in it.
            # The .pattern4 enum type itself should have been set by pattern_utils._fill_p4scores_luts_in_config

            for color_val in range(SIDE_NB):
                player_color = Color(color_val)
                opponent_color = ~player_color
                
                current_rule_config_idx = table_index(Rule(rule_val), player_color)

                # Get the Pattern4 this 'player_color' would make at this pcode
                # This P4SCORES[...][pcode].pattern4 was set by pattern_utils
                p4_self = P4SCORES[current_rule_config_idx][pcode].pattern4
                
                # What Pattern4 would opponent make if player_color doesn't play here?
                # This is tricky. The pcode represents patterns for *both* players from this empty square.
                # P4SCORES[table_idx_for_opponent_view][pcode].pattern4 would be opponent's pattern.
                table_idx_for_opponent_view = table_index(Rule(rule_val), opponent_color)
                p4_oppo = P4SCORES[table_idx_for_opponent_view][pcode].pattern4


                score_s = pattern4_heuristic_scores_self.get(p4_self, 0)
                score_o = pattern4_heuristic_scores_oppo.get(p4_oppo, 0) # Opponent's threat if we don't play here
                
                # Renju specific score adjustment for FORBID for Black
                if rule_val == Rule.RENJU and player_color == Color.BLACK and p4_self == Pattern4.FORBID:
                    score_s = pattern4_heuristic_scores_self.get(Pattern4.FORBID, 0) # Usually a deterrent or low score
                
                P4SCORES[current_rule_config_idx][pcode].set_score_self(score_s)
                P4SCORES[current_rule_config_idx][pcode].set_score_oppo(score_o)


    # Threat scores (these are more global, not per-pcode)
    # Example: if opponent has A_FIVE (mask bit 0 set), it's very bad.
    # EVALS_THREAT[table_idx][mask]
    # These are highly heuristic. Example:
    threat_base_scores = {
        0: -8000,  # oppo_five (mask bit 0)
        1: 7000,   # self_flex_four (mask bit 1)
        2: -6000,  # oppo_flex_four (mask bit 2)
        # ... add more for other threat mask bits ...
        6: 500,    # self_three (mask bit 6)
    }
    for rule_val in range(RULE_NB):
        for color_val in range(SIDE_NB):
            player_color = Color(color_val)
            idx = table_index(Rule(rule_val), player_color)
            for i in range(THREAT_NB): # i is the threat_mask
                mask_score = 0
                for bit_pos, base_score in threat_base_scores.items():
                    if (i >> bit_pos) & 1: # If this threat bit is set in the mask
                        # Score sign depends on whether it's self's threat or opponent's threat
                        # Bit positions in make_threat_mask:
                        # 0:oppo_five, 1:self_F4, 2:oppo_F4, 3:self_B4+, 4:self_B4
                        # 5:self_F3+, 6:self_F3, 7:oppo_B4+, 8:oppo_B4, 9:oppo_F3+, 10:oppo_F3
                        is_self_threat_bit = bit_pos in [1,3,4,5,6]
                        if is_self_threat_bit:
                            mask_score += base_score
                        else: # Opponent's threat bit
                            mask_score -= base_score # Subtracting a positive base_score if it's an oppo threat bit
                                                     # Or, if base_score for oppo_five is already negative, add it.
                                                     # Let's simplify: threat_base_scores are absolute impacts.
                                                     # If bit means "opponent has X", then score is negative.
                                                     # If bit means "self has Y", score is positive.
                            # Redo based on this logic:
                            # Let threat_base_scores store positive values for pattern strength
                            # Example: A_FIVE_STRENGTH = 8000, FLEX_FOUR_STRENGTH = 7000 etc.
                            # Then add/subtract based on whose pattern it is.
                            # This is complex; current threat_base_scores is already signed based on who made it.
                            # Bit 0 = oppo_five, score -8000. Bit 1 = self_F4, score +7000.
                            pass # Already handled by sign of base_score
                EVALS_THREAT[idx][i] = mask_score

# No Searcher or DBStorage creation functions for now, as they depend on class definitions.

if __name__ == '__main__':
    print(f"PCODE_NB: {PCODE_NB}")
    print(f"THREAT_NB: {THREAT_NB}")
    print(f"Default Candidate Range: {DEFAULT_CANDIDATE_RANGE.name}")

    # Test table_index
    assert table_index(Rule.FREESTYLE, Color.BLACK) == 0
    assert table_index(Rule.FREESTYLE, Color.WHITE) == 0
    assert table_index(Rule.STANDARD, Color.BLACK) == 1
    assert table_index(Rule.STANDARD, Color.WHITE) == 1
    assert table_index(Rule.RENJU, Color.BLACK) == 2
    assert table_index(Rule.RENJU, Color.WHITE) == 3
    
    # Test P4SCORES and EVALS dimensions (assuming PCODE_NB is final)
    print(f"Size of EVALS[0]: {len(EVALS[0]) if EVALS else 0}")
    assert len(EVALS[0]) == PCODE_NB
    assert len(P4SCORES[0]) == PCODE_NB

    # Test Pattern4Score
    p4s = Pattern4Score(Pattern4.A_FIVE, 1000, -50)
    print(p4s)
    assert p4s.pattern4 == Pattern4.A_FIVE
    assert p4s.score_self() == 1000
    assert p4s == Pattern4.A_FIVE # Test __eq__ with Pattern4
    assert p4s == Pattern4.A_FIVE.value # Test __eq__ with int
    
    # Test value/winrate conversion
    print(f"value_to_win_rate(0): {value_to_win_rate(0)}")
    print(f"value_to_win_rate(200): {value_to_win_rate(200)}") # SCALING_FACTOR is 200
    # expected for 200: 1 / (1 + exp(-1)) = 1 / (1 + 0.3678) = 1 / 1.3678 = ~0.731
    assert abs(value_to_win_rate(200) - 0.731) < 0.001

    print(f"win_rate_to_value(0.5): {win_rate_to_value(0.5)}")
    assert win_rate_to_value(0.5) == 0
    print(f"win_rate_to_value(0.731): {win_rate_to_value(0.7310585786300049)}")
    assert abs(win_rate_to_value(0.7310585786300049) - 200) < 1

    print("config.py basic structure tests passed.")