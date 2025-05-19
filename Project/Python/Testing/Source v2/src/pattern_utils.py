"""
Pattern recognition and lookup table utilities for the Gomoku engine,
based on Rapfi's pattern.h and pattern.cpp.
"""
import sys
from typing import List, Tuple, Dict, cast
from enum import IntEnum
import math # For math.isclose for float comparisons if any, not directly used yet

from .types import (Pattern, Pattern4, Color, Rule, PatternCode, Score, Value,
                    SIDE_NB, PATTERN_NB, RULE_NB)
from . import config as engine_config # To access P4SCORES, getValueBlack, PCODE_NB etc.

# --- PatternConfig Structures and Constants ---

class Pattern2x:
    """
    Compresses two patterns (one for Black, one for White) for a single line.
    Mirrors Rapfi's Pattern2x struct.
    """
    def __init__(self, pat_black: Pattern = Pattern.DEAD, pat_white: Pattern = Pattern.DEAD):
        self.pat_black: Pattern = pat_black
        self.pat_white: Pattern = pat_white

    def __getitem__(self, side: Color) -> Pattern:
        if side == Color.BLACK:
            return self.pat_black
        elif side == Color.WHITE:
            return self.pat_white
        else:
            # Should not happen with Color enum
            raise IndexError("Side for Pattern2x must be BLACK or WHITE")

    def __repr__(self) -> str:
        return f"P2x(B:{self.pat_black.name}, W:{self.pat_white.name})"

    def __eq__(self, other):
        if isinstance(other, Pattern2x):
            return self.pat_black == other.pat_black and self.pat_white == other.pat_white
        return False

# Helper to get HalfLineLen based on Rule
def get_half_line_len(rule: Rule) -> int:
    """Returns the number of context cells on one side of the center for pattern matching."""
    return 4 if rule == Rule.FREESTYLE else 5 # Freestyle: 4, Standard/Renju: 5

def get_key_len_bits(rule: Rule) -> int:
    """Returns number of bits for a line key (HalfLineLen * 2 cells * 2 bits/cell)."""
    return get_half_line_len(rule) * 4 # 2 cells per HLL unit, 2 bits per cell

def get_key_count(rule: Rule) -> int:
    """Total number of unique line keys for a rule."""
    return 1 << get_key_len_bits(rule)

# --- Global LUTs (initialized by init_pattern_config) ---
PATTERN2X_LUTS: List[List[Pattern2x]] = [] # Indexed by [rule.value][key]
PCODE_LUT: List[List[List[List[PatternCode]]]] = [] # Indexed by [p1][p2][p3][p4] (Pattern enum values)
UNIQUE_PCODE_COUNT: int = 0 # Actual number of unique pcodes, set by _fill_pattern_code_lut
DEFENCE_LUTS: List[List[List[int]]] = [] # Indexed by [rule.value][key][attacker_color.value]


# --- Logic from pattern.cpp (anonymous namespace) ---
class ColorFlag(IntEnum):
    """Represents stone state on a line from one player's perspective."""
    SELF = 0
    OPPO = 1 # Opponent's stone or wall
    EMPT = 2 # Empty

LineType = List[ColorFlag] # A line is a list of ColorFlags

# Memoization for _get_pattern_recursive: [rule.value][side.value][line_tuple] -> Pattern
_pattern_memos: List[List[Dict[Tuple[ColorFlag,...], Pattern]]] = []
_pattern_memos_initialized_flag: bool = False

def _initialize_pattern_memos_if_needed():
    global _pattern_memos, _pattern_memos_initialized_flag
    if not _pattern_memos_initialized_flag:
        _pattern_memos = [[{} for _ in range(SIDE_NB)] for _ in range(RULE_NB)]
        _pattern_memos_initialized_flag = True

def _line_from_key(key: int, rule: Rule, current_player_perspective: Color) -> LineType:
    """
    Constructs a Line (list of ColorFlag) from a bit key for pattern analysis.
    The key represents (HalfLineLen * 2) context cells.
    The line constructed is (HalfLineLen * 2 + 1) long, with SELF at the center.
    """
    half_len = get_half_line_len(rule)
    line_len_for_pattern = half_len * 2 + 1
    line: LineType = [ColorFlag.EMPT] * line_len_for_pattern
    
    key_num_cells = half_len * 2 # Number of cells represented in the key

    for i in range(line_len_for_pattern):
        if i == half_len: # Center of the pattern line
            line[i] = ColorFlag.SELF
            continue

        # Determine which cell in the compact `key` this `line[i]` corresponds to
        key_cell_idx = i if i < half_len else i - 1
        
        # Extract the 2 bits for this key_cell_idx from `key`
        # Each cell in `key` is (is_white_present_bit, is_black_present_bit)
        # Rapfi's bitkey: 00 EMPTY, 01 BLACK, 10 WHITE, 11 WALL
        cell_raw_bits = (key >> (2 * key_cell_idx)) & 0b11
        
        is_black_on_raw_cell = (cell_raw_bits == (0x1 + Color.BLACK.value))
        is_white_on_raw_cell = (cell_raw_bits == (0x1 + Color.WHITE.value))
        # is_empty_on_raw_cell = (cell_raw_bits == 0b00)
        # is_wall_on_raw_cell = (cell_raw_bits == 0b11) # Treat WALL as OPPO

        if cell_raw_bits == 0b11: # Wall
            line[i] = ColorFlag.OPPO
        elif current_player_perspective == Color.BLACK:
            if is_black_on_raw_cell: line[i] = ColorFlag.SELF
            elif is_white_on_raw_cell: line[i] = ColorFlag.OPPO
            else: line[i] = ColorFlag.EMPT
        else: # current_player_perspective == Color.WHITE
            if is_white_on_raw_cell: line[i] = ColorFlag.SELF
            elif is_black_on_raw_cell: line[i] = ColorFlag.OPPO
            else: line[i] = ColorFlag.EMPT
    return line

def _count_line_features(line: LineType) -> Tuple[int, int, int, int]:
    """Counts real length, full length, start, and end of SELF stones in a line."""
    line_len = len(line)
    mid_idx = line_len // 2
    if line[mid_idx] != ColorFlag.SELF: # Should always be SELF for lines from _get_pattern_recursive
        # This case can happen if called on an arbitrary line not centered on SELF
        # For simplicity, let's assume it's always called on a line centered on SELF by the caller.
        pass

    real_len = 1 # Starts with the center SELF stone
    full_len = 1
    real_len_inc_left = 1
    real_len_inc_right = 1
    start_scan_idx = mid_idx
    end_scan_idx = mid_idx

    for i in range(mid_idx - 1, -1, -1):
        if line[i] == ColorFlag.SELF:
            real_len += real_len_inc_left
        elif line[i] == ColorFlag.OPPO:
            break
        else: # EMPT
            real_len_inc_left = 0
        full_len += 1
        start_scan_idx = i

    for i in range(mid_idx + 1, line_len):
        if line[i] == ColorFlag.SELF:
            real_len += real_len_inc_right
        elif line[i] == ColorFlag.OPPO:
            break
        else: # EMPT
            real_len_inc_right = 0
        full_len += 1
        end_scan_idx = i
        
    return real_len, full_len, start_scan_idx, end_scan_idx

def _shift_line(line: LineType, original_center_idx_of_new_line: int) -> LineType:
    """
    Shifts the `line` so that the stone at `original_center_idx_of_new_line`
    becomes the new center of the returned line.
    Used when exploring placing a stone at an empty spot.
    """
    line_len = len(line)
    new_mid_idx = line_len // 2
    shifted_line: LineType = [ColorFlag.OPPO] * line_len # Default to OPPO (wall) for out of bounds

    for j in range(line_len):
        # Calculate corresponding index in the original line
        original_line_idx = j + original_center_idx_of_new_line - new_mid_idx
        if 0 <= original_line_idx < line_len:
            shifted_line[j] = line[original_line_idx]
    return shifted_line

def _get_pattern_recursive(rule: Rule, side_for_renju_check: Color, line: LineType) -> Pattern:
    """
    Calculates the pattern for a line using memoized recursion.
    `side_for_renju_check` is the perspective (BLACK or WHITE) for Renju overline checks.
    `line` is assumed to have ColorFlag.SELF at its center.
    """
    _initialize_pattern_memos_if_needed()
    # Memo key must be immutable (tuple)
    line_tuple = tuple(line)
    memo = _pattern_memos[rule.value][side_for_renju_check.value]

    if line_tuple in memo:
        return memo[line_tuple]

    line_len = len(line)
    mid_idx = line_len // 2
    check_overline = (rule == Rule.STANDARD) or \
                     (rule == Rule.RENJU and side_for_renju_check == Color.BLACK)

    real_len, full_len, scan_start, scan_end = _count_line_features(line)
    current_pattern = Pattern.DEAD

    if check_overline and real_len >= 6:
        current_pattern = Pattern.OL
    elif real_len >= 5:
        current_pattern = Pattern.F5
    elif full_len < 5: # Not enough space to make 5
        current_pattern = Pattern.DEAD
    else:
        # Recursive step: try placing a SELF stone in empty spots within the scannable range
        pat_counts = [0] * PATTERN_NB
        f5_indices_in_original_line: List[int] = []

        for i in range(scan_start, scan_end + 1): # Iterate over the "full_len" span
            if line[i] == ColorFlag.EMPT:
                # Simulate placing a stone:
                # 1. Shift the original `line` so that `line[i]` is at the center of `shifted_line`.
                shifted_line = _shift_line(line, i)
                # 2. The center of `shifted_line` (which was `line[i]`, an EMPT) is now SELF.
                #    This `next_line_for_recursion` is what we get the sub_pattern for.
                next_line_for_recursion = list(shifted_line) # mutable copy
                next_line_for_recursion[mid_idx] = ColorFlag.SELF
                
                sub_pattern = _get_pattern_recursive(rule, side_for_renju_check, next_line_for_recursion)
                
                if sub_pattern == Pattern.F5 and len(f5_indices_in_original_line) < 2:
                    f5_indices_in_original_line.append(i)
                pat_counts[sub_pattern.value] += 1
        
        # Determine pattern based on counts of sub_patterns (Rapfi's logic)
        if pat_counts[Pattern.F5.value] >= 2:
            current_pattern = Pattern.F4
            if rule == Rule.RENJU and side_for_renju_check == Color.BLACK:
                if len(f5_indices_in_original_line) >= 2 and \
                   (f5_indices_in_original_line[1] - f5_indices_in_original_line[0]) < 5:
                    current_pattern = Pattern.OL
        elif pat_counts[Pattern.F5.value] == 1:
            current_pattern = Pattern.B4
        elif pat_counts[Pattern.F4.value] >= 2:
            current_pattern = Pattern.F3S
        elif pat_counts[Pattern.F4.value] == 1:
            current_pattern = Pattern.F3
        elif pat_counts[Pattern.B4.value] == 1:
            current_pattern = Pattern.B3
        elif (pat_counts[Pattern.F3S.value] + pat_counts[Pattern.F3.value]) >= 4:
            current_pattern = Pattern.F2B
        elif (pat_counts[Pattern.F3S.value] + pat_counts[Pattern.F3.value]) >= 3:
            current_pattern = Pattern.F2A
        elif (pat_counts[Pattern.F3S.value] + pat_counts[Pattern.F3.value]) >= 1:
            current_pattern = Pattern.F2
        elif pat_counts[Pattern.B3.value] == 1:
            current_pattern = Pattern.B2
        elif (pat_counts[Pattern.F2.value] + pat_counts[Pattern.F2A.value] + pat_counts[Pattern.F2B.value]) >= 1:
            current_pattern = Pattern.F1
        elif pat_counts[Pattern.B2.value] == 1:
            current_pattern = Pattern.B1
        # else: current_pattern remains Pattern.DEAD

    memo[line_tuple] = current_pattern
    return current_pattern

def _fill_pattern2x_luts() -> None:
    """Initializes PATTERN2X_LUTS for all rules."""
    global PATTERN2X_LUTS
    PATTERN2X_LUTS = [[] for _ in range(RULE_NB)] # One list per rule

    _initialize_pattern_memos_if_needed()

    for rule_val in range(RULE_NB):
        rule = Rule(rule_val)
        key_cnt = get_key_count(rule)
        
        current_lut_for_rule = [Pattern2x() for _ in range(key_cnt)]
        PATTERN2X_LUTS[rule_val] = current_lut_for_rule

        for key_value in range(key_cnt):
            # Get line from Black's perspective
            line_for_black = _line_from_key(key_value, rule, Color.BLACK)
            pat_black = _get_pattern_recursive(rule, Color.BLACK, line_for_black)

            # Get line from White's perspective
            line_for_white = _line_from_key(key_value, rule, Color.WHITE)
            pat_white = _get_pattern_recursive(rule, Color.WHITE, line_for_white)
            
            current_lut_for_rule[key_value] = Pattern2x(pat_black, pat_white)

def _get_pattern4(pats: Tuple[Pattern, Pattern, Pattern, Pattern], is_renju_black_rules: bool) -> Pattern4:
    """Determines Pattern4 from four line patterns, considering Renju rules for Black if applicable."""
    n = [0] * PATTERN_NB # Counts for each Pattern type
    for p_enum in pats:
        n[p_enum.value] += 1

    if n[Pattern.F5.value] >= 1: return Pattern4.A_FIVE
    if is_renju_black_rules: # Forbidden checks for Renju Black
        if n[Pattern.OL.value] >= 1: return Pattern4.FORBID
        if (n[Pattern.F4.value] + n[Pattern.B4.value]) >= 2: return Pattern4.FORBID
        if (n[Pattern.F3.value] + n[Pattern.F3S.value]) >= 2: return Pattern4.FORBID

    if n[Pattern.B4.value] >= 2: return Pattern4.B_FLEX4
    if n[Pattern.F4.value] >= 1: return Pattern4.B_FLEX4
    
    if n[Pattern.B4.value] >= 1:
        if n[Pattern.F3.value] >= 1 or n[Pattern.F3S.value] >= 1: return Pattern4.C_BLOCK4_FLEX3
        if n[Pattern.B3.value] >= 1: return Pattern4.D_BLOCK4_PLUS
        if (n[Pattern.F2.value] + n[Pattern.F2A.value] + n[Pattern.F2B.value]) >= 1: return Pattern4.D_BLOCK4_PLUS
        return Pattern4.E_BLOCK4
        
    if n[Pattern.F3.value] >= 1 or n[Pattern.F3S.value] >= 1:
        if (n[Pattern.F3.value] + n[Pattern.F3S.value]) >= 2: return Pattern4.F_FLEX3_2X
        if n[Pattern.B3.value] >= 1: return Pattern4.G_FLEX3_PLUS
        if (n[Pattern.F2.value] + n[Pattern.F2A.value] + n[Pattern.F2B.value]) >= 1: return Pattern4.G_FLEX3_PLUS
        return Pattern4.H_FLEX3

    if n[Pattern.B3.value] >= 1: # Must be checked after F3 types for G_FLEX3_PLUS
        if n[Pattern.B3.value] >= 2: return Pattern4.I_BLOCK3_PLUS
        if (n[Pattern.F2.value] + n[Pattern.F2A.value] + n[Pattern.F2B.value]) >= 1: return Pattern4.I_BLOCK3_PLUS
    
    if (n[Pattern.F2.value] + n[Pattern.F2A.value] + n[Pattern.F2B.value]) >= 2: return Pattern4.J_FLEX2_2X
    
    if n[Pattern.B3.value] >= 1: return Pattern4.K_BLOCK3 # If not I_BLOCK3_PLUS
    if (n[Pattern.F2.value] + n[Pattern.F2A.value] + n[Pattern.F2B.value]) >= 1: return Pattern4.L_FLEX2 # If not J_FLEX2_2X

    return Pattern4.NONE

def _fill_pattern_code_lut() -> None:
    """Initializes PCODE_LUT and sets UNIQUE_PCODE_COUNT."""
    global PCODE_LUT, UNIQUE_PCODE_COUNT
    
    N = PATTERN_NB
    # Using a dictionary to map canonical (sorted) tuples of pattern values to unique codes
    unique_code_map: Dict[Tuple[int, ...], int] = {}
    next_available_code = 0

    # Initialize PCODE_LUT with zeros or a placeholder
    PCODE_LUT = [[[[0]*N for _ in range(N)] for _ in range(N)] for _ in range(N)]
    patterns_enum_list = list(Pattern) # Get all Pattern enum members

    for p1 in patterns_enum_list:
        for p2 in patterns_enum_list:
            for p3 in patterns_enum_list:
                for p4 in patterns_enum_list:
                    # Create a canonical representation (sorted tuple of pattern *values*)
                    canonical_key = tuple(sorted((p1.value, p2.value, p3.value, p4.value)))
                    
                    if canonical_key not in unique_code_map:
                        unique_code_map[canonical_key] = next_available_code
                        next_available_code += 1
                    
                    PCODE_LUT[p1.value][p2.value][p3.value][p4.value] = unique_code_map[canonical_key]

    UNIQUE_PCODE_COUNT = next_available_code
    
    # Ensure PCODE_NB from config matches this calculated unique count
    if engine_config.PCODE_NB != UNIQUE_PCODE_COUNT:
        print(f"Warning: engine_config.PCODE_NB ({engine_config.PCODE_NB}) does not match "
              f"calculated UNIQUE_PCODE_COUNT ({UNIQUE_PCODE_COUNT}). "
              f"Using calculated count for table dimensions.", file=sys.stderr)
        # This implies config.P4SCORES and config.EVALS might need resizing if they used the old PCODE_NB.
        # This is handled in init_pattern_config after this function runs.


def _fill_p4scores_luts_in_config() -> None:
    """Populates engine_config.P4SCORES using PCODE_LUT and _get_pattern4."""
    if not PCODE_LUT:
        # print("Error: PCODE_LUT not initialized before _fill_p4scores_luts_in_config.", file=sys.stderr)
        return
    if UNIQUE_PCODE_COUNT == 0:
        # print("Error: UNIQUE_PCODE_COUNT is 0.", file=sys.stderr)
        return

    # Resize P4SCORES in config if UNIQUE_PCODE_COUNT differs from initial PCODE_NB
    # This must happen BEFORE we try to write to it using pcodes as indices.
    if len(engine_config.P4SCORES[0]) != UNIQUE_PCODE_COUNT:
        engine_config.P4SCORES = [
            [engine_config.Pattern4Score() for _ in range(UNIQUE_PCODE_COUNT)]
            for _ in range(engine_config.EVAL_TABLE_DIM1_SIZE)
        ]

    patterns_enum_list = list(Pattern)
    for p1 in patterns_enum_list:
        for p2 in patterns_enum_list:
            for p3 in patterns_enum_list:
                for p4 in patterns_enum_list:
                    pcode = PCODE_LUT[p1.value][p2.value][p3.value][p4.value]
                    current_patterns_tuple = (p1, p2, p3, p4)

                    # Index for P4SCORES: 0=FS, 1=STD, 2=RenjuBlack, 3=RenjuWhite
                    # Freestyle (idx 0)
                    idx_fs = engine_config.table_index(Rule.FREESTYLE, Color.BLACK) # Or WHITE, same for FS
                    p4_fs = _get_pattern4(current_patterns_tuple, False)
                    engine_config.P4SCORES[idx_fs][pcode] = engine_config.Pattern4Score(p4_fs, 0, 0) # Dummy scores

                    # Standard (idx 1)
                    idx_std = engine_config.table_index(Rule.STANDARD, Color.BLACK)
                    p4_std = _get_pattern4(current_patterns_tuple, False)
                    engine_config.P4SCORES[idx_std][pcode] = engine_config.Pattern4Score(p4_std, 0, 0)

                    # Renju Black (idx 2)
                    idx_renju_b = engine_config.table_index(Rule.RENJU, Color.BLACK)
                    p4_renju_b = _get_pattern4(current_patterns_tuple, True) # is_renju_black = True
                    engine_config.P4SCORES[idx_renju_b][pcode] = engine_config.Pattern4Score(p4_renju_b, 0, 0)

                    # Renju White (idx 3)
                    idx_renju_w = engine_config.table_index(Rule.RENJU, Color.WHITE)
                    p4_renju_w = _get_pattern4(current_patterns_tuple, False) # is_renju_black = False
                    engine_config.P4SCORES[idx_renju_w][pcode] = engine_config.Pattern4Score(p4_renju_w, 0, 0)

def _fill_defence_luts() -> None:
    """Initializes DEFENCE_LUTS for all rules."""
    global DEFENCE_LUTS
    # DEFENCE_LUTS: List[List[List[int]]] = [] # Indexed by [rule.value][key][attacker_color.value]
    DEFENCE_LUTS = [[[0]*SIDE_NB for _ in range(get_key_count(Rule(r)))] for r in range(RULE_NB)]

    if not PATTERN2X_LUTS:
        # print("Error: PATTERN2X_LUTS must be initialized before _fill_defence_luts.", file=sys.stderr)
        return

    for rule_val in range(RULE_NB):
        rule = Rule(rule_val)
        key_cnt = get_key_count(rule)
        half_len = get_half_line_len(rule)
        
        for key_value in range(key_cnt):
            for attacker_val in range(SIDE_NB):
                attacker = Color(attacker_val)
                defence_mask = 0
                
                p2x_original = lookup_pattern_from_luts(rule, key_value) # Use already filled PATTERN2X_LUTS
                attack_pattern_original = p2x_original[attacker]

                if attack_pattern_original.value >= Pattern.F3.value:
                    num_key_cells = half_len * 2
                    for i in range(num_key_cells): # i is index within the compact key's cells
                        # Check if cell 'i' in the original key_value is empty
                        # Raw bits for cell 'i': (key_value >> (2*i)) & 0b11
                        # Empty if raw bits == 0b00
                        if ((key_value >> (2 * i)) & 0b11) == 0b00:
                            # Simulate placing defender's stone at this empty cell 'i'
                            defender = ~attacker
                            # Defender stone value in raw bits: BLACK=01, WHITE=10
                            defender_raw_bits = (0x1 + defender.value)
                            
                            key_with_block = key_value | (defender_raw_bits << (2 * i))
                            
                            p2x_after_block = lookup_pattern_from_luts(rule, key_with_block)
                            if p2x_after_block[attacker].value < Pattern.F3.value: # Threat removed/reduced
                                defence_mask |= (1 << i)
                
                # Center the 8-bit defence mask from the 2*HalfLineLen bit mask
                shift_amount = half_len - 4 # Target 8 cells, source 2*half_len cells
                if shift_amount >= 0:
                    final_mask = (defence_mask >> shift_amount) & 0xFF
                else: # Should not happen if HLL >= 4
                    final_mask = defence_mask & 0xFF
                
                DEFENCE_LUTS[rule_val][key_value][attacker_val] = final_mask

# --- Public API for PatternConfig ---
def fuse_key(rule: Rule, key_segment_from_bitboard: int) -> int:
    """
    Removes the center conceptual piece's bits from a raw line key segment (from bitboard)
    to get the lookup key for PATTERN2X_LUTS.
    The key_segment_from_bitboard has (2*HalfLineLen + 1) cells * 2 bits/cell.
    The LUT key has (2*HalfLineLen) cells * 2 bits/cell.
    """
    half_len = get_half_line_len(rule)
    # Rapfi's non-BMI2 logic:
    # This means the input `key_segment_from_bitboard` here is (2*HLL+1)*2 bit segment.
    # Example: for HLL=4, input is 18 bits. Output is 16 bits.
    # ((key >> 2) & 0xff00) | (key & 0x00ff) -> assumes key is right-aligned after MSB part
    # The bits are: LLLL M RRRR. key_segment: [L4 L3 L2 L1 M R1 R2 R3 R4]
    # For HLL=4, (2*4+1)*2 = 18 bits. Output 16 bits.
    # Left part (L1-L4) is key & ((1<<(2*HL))-1). Right part is key >> (2*(HL+1)).
    # Fused = (RightPart << (2*HL)) | LeftPart
    
    left_mask = (1 << (half_len * 2)) - 1
    left_part = key_segment_from_bitboard & left_mask
    
    # Shift away left part and middle 2 bits
    right_part = key_segment_from_bitboard >> (half_len * 2 + 2) 
    
    fused = (right_part << (half_len * 2)) | left_part
    return fused


def lookup_pattern_from_luts(rule: Rule, fused_key: int) -> Pattern2x:
    """Looks up Pattern2x from the precomputed LUT using a fused_key."""
    rule_val = rule.value
    if not (0 <= rule_val < len(PATTERN2X_LUTS) and PATTERN2X_LUTS[rule_val]):
        # print(f"Warning: PATTERN2X_LUTS not initialized for rule {rule.name} or rule index out of bounds.", file=sys.stderr)
        return Pattern2x()
    if not (0 <= fused_key < len(PATTERN2X_LUTS[rule_val])):
        # print(f"Warning: fused_key {fused_key} out of range for PATTERN2X_LUTS[{rule.name}] (size {len(PATTERN2X_LUTS[rule_val])}).", file=sys.stderr)
        return Pattern2x()
    return PATTERN2X_LUTS[rule_val][fused_key]

def lookup_defence_mask_from_luts(rule: Rule, fused_key: int, attacker_side: Color) -> int:
    """Looks up the 8-bit defence mask from precomputed LUTs."""
    rule_val = rule.value
    attacker_idx = attacker_side.value
    if not (0 <= rule_val < len(DEFENCE_LUTS) and DEFENCE_LUTS[rule_val]):
        # print(f"Warning: DEFENCE_LUTS not initialized for rule {rule.name}.", file=sys.stderr)
        return 0
    if not (0 <= fused_key < len(DEFENCE_LUTS[rule_val])):
        # print(f"Warning: fused_key {fused_key} out of range for DEFENCE_LUTS[{rule.name}].", file=sys.stderr)
        return 0
    # DEFENCE_LUTS[rule_val][fused_key] is a list of size SIDE_NB
    if not (0 <= attacker_idx < len(DEFENCE_LUTS[rule_val][fused_key])):
         # print(f"Warning: attacker_idx {attacker_idx} out of range for DEFENCE_LUTS[{rule.name}][{fused_key}].", file=sys.stderr)
         return 0
    return DEFENCE_LUTS[rule_val][fused_key][attacker_idx]

def get_pcode_from_patterns(p1: Pattern, p2: Pattern, p3: Pattern, p4: Pattern) -> PatternCode:
    """Gets the compressed PatternCode from four line patterns using PCODE_LUT."""
    if not PCODE_LUT: # Should be filled by init
        # print("Warning: PCODE_LUT not initialized! Returning 0.", file=sys.stderr)
        return 0 
    return PCODE_LUT[p1.value][p2.value][p3.value][p4.value]


# --- Initialization ---
def init_pattern_config():
    """Initializes all pattern configuration LUTs."""
    global PATTERN2X_LUTS, PCODE_LUT, UNIQUE_PCODE_COUNT, DEFENCE_LUTS
    
    _fill_pattern_code_lut() # Sets UNIQUE_PCODE_COUNT and PCODE_LUT
    _fill_pattern2x_luts()   # Sets PATTERN2X_LUTS, depends on _get_pattern_recursive
    
    # P4SCORES in config.py needs to be consistent with UNIQUE_PCODE_COUNT
    # This function now assumes config.PCODE_NB was already correct or that
    # P4SCORES can handle the true UNIQUE_PCODE_COUNT.
    # The _fill_p4scores_luts_in_config will resize config.P4SCORES if needed.
    _fill_p4scores_luts_in_config() # Populates engine_config.P4SCORES

    _fill_defence_luts()     # Sets DEFENCE_LUTS, depends on PATTERN2X_LUTS

# Call initialization when module is loaded
init_pattern_config()


if __name__ == '__main__':
    print("--- Pattern Utils Tests ---")
    if not PATTERN2X_LUTS or not PCODE_LUT or not DEFENCE_LUTS or UNIQUE_PCODE_COUNT == 0:
        print("ERROR: LUTs not initialized properly!", file=sys.stderr)
        # Try to re-init for isolated test run if necessary
        # init_pattern_config()
    else:
        print(f"PATTERN2X_LUTS[FS] size: {len(PATTERN2X_LUTS[Rule.FREESTYLE.value])} (expected {get_key_count(Rule.FREESTYLE)})")
        assert len(PATTERN2X_LUTS[Rule.FREESTYLE.value]) == get_key_count(Rule.FREESTYLE)
        
        print(f"PATTERN2X_LUTS[RENJU] size: {len(PATTERN2X_LUTS[Rule.RENJU.value])} (expected {get_key_count(Rule.RENJU)})")
        assert len(PATTERN2X_LUTS[Rule.RENJU.value]) == get_key_count(Rule.RENJU)

        print(f"Unique PCODE_COUNT: {UNIQUE_PCODE_COUNT}")
        # Max H(14,4) = 2380
        assert 0 < UNIQUE_PCODE_COUNT <= 2380

        print(f"engine_config.P4SCORES[0] size: {len(engine_config.P4SCORES[0]) if engine_config.P4SCORES else 'Not Init'}")
        assert len(engine_config.P4SCORES[0]) == UNIQUE_PCODE_COUNT
        
        print(f"DEFENCE_LUTS[FS] size: {len(DEFENCE_LUTS[Rule.FREESTYLE.value])}")
        assert len(DEFENCE_LUTS[Rule.FREESTYLE.value]) == get_key_count(Rule.FREESTYLE)
        if get_key_count(Rule.FREESTYLE) > 0:
             assert len(DEFENCE_LUTS[Rule.FREESTYLE.value][0]) == SIDE_NB


        # Test fuse_key
        # A key segment from bitboard representing 9 cells (HLL=4 for FS) -> 18 bits
        # LLLL M RRRR. Example: M is Black (01), L1 is White (10), rest Empty (00)
        # Key from bitboard (example, bits for R4,R3,R2,R1, M, L1,L2,L3,L4 - LSB first):
        # R4 R3 R2 R1  M  L1 L2 L3 L4
        # 00 00 00 00  01 10 00 00 00  (binary)
        raw_key_fs = 0b000000000110000000
        fused_fs = fuse_key(Rule.FREESTYLE, raw_key_fs)
        # Expected: LLLL RRRR -> L1=White (10), rest empty (00)
        # L4L3L2L1 R4R3R2R1
        # 00000010 00000000 -> 0b0000001000000000 = 2 * 256 = 512
        # My fuse_key logic: left_part = raw & ( (1<<(4*2))-1 ) = raw & 0xFF = 0b10000000 (L1..L4)
        # right_part = raw >> (4*2 + 2) = raw >> 10 = 0b00000000 (R1..R4)
        # fused = (right_part << 8) | left_part = 0b10000000 = 128
        print(f"Raw key FS: {bin(raw_key_fs)}, Fused FS: {bin(fused_fs)} (decimal {fused_fs})")
        # This assertion needs careful validation of bit order in key segment and fuse_key.
        # Rapfi's non-BMI2 fuse_key: ((key >> 2) & 0xff00) | (key & 0x00ff)
        # If raw_key_fs is 0b000000000110000000 = 0x00180
        # (0x180 >> 2) & 0xFF00 = (0x60 & 0xFF00) = 0x0000
        # (0x180 & 0x00FF)      = 0x0080
        # Result = 0x80 = 128. This matches my Python fuse_key.
        assert fused_fs == 0b10000000 # L1 is White, others Empty in the fused key

        # Test lookup_pattern_from_luts
        # Key 0 for FS (all context cells empty). Fused key is also 0.
        p2x_key0_fs = lookup_pattern_from_luts(Rule.FREESTYLE, 0)
        print(f"Pattern2x for fused key 0 (Freestyle): {p2x_key0_fs}")
        assert p2x_key0_fs.pat_black == Pattern.F1
        assert p2x_key0_fs.pat_white == Pattern.F1

        # Test _get_pattern_recursive with a simple line
        test_line_f1 = [ColorFlag.EMPT]*4 + [ColorFlag.SELF] + [ColorFlag.EMPT]*4 # LineLen=9 for HLL=4
        pat_f1_b = _get_pattern_recursive(Rule.FREESTYLE, Color.BLACK, test_line_f1)
        print(f"Pattern for EEEE S EEEE (FS, B): {pat_f1_b.name}")
        assert pat_f1_b == Pattern.F1

        # Test PCODE_LUT access
        pcode_all_dead = get_pcode_from_patterns(Pattern.DEAD, Pattern.DEAD, Pattern.DEAD, Pattern.DEAD)
        print(f"PCode for (DEAD,DEAD,DEAD,DEAD): {pcode_all_dead}")
        assert pcode_all_dead >= 0 and pcode_all_dead < UNIQUE_PCODE_COUNT

        # Test DEFENCE_LUTS
        # For an empty line (key 0), defence mask should be 0
        def_mask_empty_fs_b = lookup_defence_mask_from_luts(Rule.FREESTYLE, 0, Color.BLACK)
        print(f"Defence mask for empty line (FS, attacker B): {def_mask_empty_fs_b}")
        assert def_mask_empty_fs_b == 0

        print("Pattern utils tests completed.")