"""
Move generation functions for the Gomoku engine.
Based on Rapfi's movegen.h and movegen.cpp.
"""
from __future__ import annotations # For type hinting Board within ScoredMove/functions before full def
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
from enum import IntFlag, Enum # IntFlag for GenType

from .types import Pattern, Pattern4, Color, Rule, Score, Value, CandidateRange
from .pos import Pos, Direction, DIRECTIONS # Assuming Pos, Direction, DIRECTIONS are defined
from .board_utils import Cell # Assuming Cell is defined
# Import Board later with a type checking guard if needed, or rely on forward hints
if TYPE_CHECKING:
    from .board import Board # Full import for type checkers

# GenType: Specifies which pattern types are allowed to generate.
# Using IntFlag for bitwise operations.
class GenType(IntFlag):
    # Rule part (values 0-3, mutually exclusive in practice for this part)
    # These are not flags in the sense of combining, but distinct modes.
    # We'll handle the rule part separately if needed, or assume rule is passed explicitly.
    # For now, focus on the bitmask part.
    # RULE_ALL       = 0 # Implicitly, if no specific rule type is set as a flag
    # RULE_FREESTYLE = 1 # Not treated as a flag here for combination
    # RULE_STANDARD  = 2
    # RULE_RENJU     = 3

    NONE           = 0 # Helper for no flags
    COMB           = 1 << 2  # Move must be a combination pattern type (4)
    WINNING        = 1 << 3  # (8)
    VCF            = 1 << 4  # (16)
    VCT            = 1 << 5  # (32)
    VC2            = 1 << 6  # (64)
    TRIVIAL        = 1 << 7  # (128) - Generate all other moves if this is set

    DEFEND_FIVE    = 1 << 8  # (256) - Only generate A_FIVE defence move
    DEFEND_FOUR    = 1 << 9  # (512) - Only generate B_FLEX4 defence move
    DEFEND_B4F3    = 1 << 10 # (1024) - Only generate C_BLOCK4_FLEX3 defence move
    
    DEFEND         = DEFEND_FIVE | DEFEND_FOUR | DEFEND_B4F3
    ALL            = WINNING | VCF | VCT | VC2 | TRIVIAL # Generate all moves, no matter whether needs to defend

# Max distance to find a pos in a line (from movegen.cpp anonymous namespace)
MAX_FIND_DIST = 4

class ScoredMove:
    """
    Contains a position (Pos) and its score(s) for move ordering.
    Mirrors Rapfi's ScoredMove struct.
    """
    def __init__(self, pos: Pos = Pos.NONE, score: Score = 0, raw_score: Score = 0):
        self.pos: Pos = pos
        self.score: Score = score # Score with history/heuristics for sorting
        
        # Union in C++ for rawScore/policy. Python doesn't have unions directly.
        # We can use a property or just have both if needed.
        # For classical evaluation focus, raw_score is primary.
        self.raw_score: Score = raw_score # Raw score from table or evaluator
        self.policy: float = 0.0 # Normalized policy score (for NN, placeholder here)

    def __lt__(self, other: ScoredMove) -> bool: # For sorting (descending score)
        return self.score > other.score # Higher score is "less than" for min-heap (or sort reverse)

    def __repr__(self) -> str:
        return f"ScoredMove({self.pos!r}, score={self.score}, raw={self.raw_score})"

    # Operator Pos() const -> Pythonic equivalent is not direct.
    # Usually access move.pos.
    # def __pos__(self) -> Pos: return self.pos # Not standard

    # void operator=(Pos p) -> Pythonic: move.pos = p
    
    @staticmethod
    def score_comparator(a: ScoredMove, b: ScoredMove) -> bool:
        """Sorts by score descending."""
        return a.score > b.score

    @staticmethod
    def policy_comparator(a: ScoredMove, b: ScoredMove) -> bool:
        """Sorts by policy descending."""
        return a.policy > b.policy

# --- Helper functions from anonymous namespace in movegen.cpp ---

def _basic_pattern_filter(board: 'Board', pos: Pos, side: Color, gen_type: GenType, rule: Rule) -> bool:
    """Filters move by Pattern4 according to GenType."""
    cell = board.get_cell(pos) # Use board's method to get cell
    p4 = cell.pattern4[side.value] # Pattern4 for the 'side' considering this move

    is_renju_rule = (rule == Rule.RENJU)

    if gen_type & GenType.WINNING:
        if p4.value >= Pattern4.B_FLEX4.value: # B_FLEX4 or A_FIVE
            return True
    
    if gen_type & GenType.VCF:
        if gen_type & GenType.COMB: # VCF + COMB
            if p4.value >= Pattern4.D_BLOCK4_PLUS.value:
                return True
        elif is_renju_rule: # VCF + RENJU (implicit)
            # FORBID for BLACK, or E_BLOCK4+
            # Also check if FORBID is due to B4 lines
            if p4 == Pattern4.FORBID and side == Color.BLACK:
                # Check if any of the 4 line patterns are B4 or F4 (winning if not for forbid)
                is_b4_forbidden = False
                for dir_idx in range(4):
                    line_pat = cell.get_line_pattern(side, dir_idx)
                    if line_pat == Pattern.B4 or line_pat == Pattern.F4:
                        is_b4_forbidden = True
                        break
                if is_b4_forbidden:
                    return True
            elif p4.value >= Pattern4.E_BLOCK4.value:
                return True
        else: # VCF (Freestyle/Standard)
            if p4.value >= Pattern4.E_BLOCK4.value:
                return True

    if gen_type & GenType.VCT: # VCT (and not already covered by VCF if VCF is also set)
        if p4.value < Pattern4.E_BLOCK4.value: # Only if not already a VCF move
            if gen_type & GenType.COMB:
                if p4.value >= Pattern4.G_FLEX3_PLUS.value:
                    return True
            else:
                if p4.value >= Pattern4.H_FLEX3.value:
                    return True
    
    if gen_type & GenType.VC2: # VC2 (and not VCF/VCT)
        if p4.value < Pattern4.H_FLEX3.value:
            if gen_type & GenType.COMB:
                if p4.value >= Pattern4.J_FLEX2_2X.value:
                    return True
            else:
                if p4.value >= Pattern4.L_FLEX2.value:
                    return True
    
    return bool(gen_type & GenType.TRIVIAL) # If TRIVIAL is set, pass if no other category matched

def _pre_check_filter(board: 'Board', side: Color, gen_type: GenType) -> bool:
    """A fast check to skip unnecessary move generation if certain patterns don't exist."""
    if gen_type & GenType.VCF:
        # Get p4_count from current board state
        p4_counts_for_side = board.get_current_state_info().p4_count[side.value]
        if gen_type & GenType.COMB:
            if (p4_counts_for_side[Pattern4.D_BLOCK4_PLUS.value] + \
                p4_counts_for_side[Pattern4.C_BLOCK4_FLEX3.value]) == 0:
                return False # No D_BLOCK4_PLUS or C_BLOCK4_FLEX3, so no VCF-COMB moves
        else: # VCF without COMB
            if (p4_counts_for_side[Pattern4.E_BLOCK4.value] + \
                p4_counts_for_side[Pattern4.D_BLOCK4_PLUS.value] + \
                p4_counts_for_side[Pattern4.C_BLOCK4_FLEX3.value]) == 0:
                return False
    return True # Continue generation

def _find_first_pattern4_pos(board: 'Board', side: Color, p4_target: Pattern4) -> Pos:
    """Gets the first found pos that has the given pattern4."""
    for pos in board.iter_candidate_moves(): # Uses board's candidate iterator
        if board.get_cell(pos).pattern4[side.value] == p4_target:
            return pos
    # print(f"Warning: Could not find pattern4 {p4_target.name} for side {side.name} when expected.", file=sys.stderr)
    return Pos.NONE # Should be found if p4Count > 0

# --- More complex defence move generation logic (ported from movegen.cpp) ---
# These are quite involved and depend heavily on precise board state and pattern interaction.
# They will be placeholders or simplified initially.

def _find_all_pseudo_four_defend_pos(board: 'Board', side_with_four: Color, move_list_ref: List[ScoredMove]):
    """Finds all positions that defend against a FOUR by `side_with_four`."""
    # Appends ScoredMove(pos) to move_list_ref
    for pos in board.iter_candidate_moves():
        cell = board.get_cell(pos)
        # If placing a stone at `pos` blocks `side_with_four`'s threat
        if cell.pattern4[side_with_four.value].value >= Pattern4.E_BLOCK4.value:
            move_list_ref.append(ScoredMove(pos))
        elif cell.pattern4[side_with_four.value] == Pattern4.FORBID and side_with_four == Color.BLACK:
            # Check if FORBID is due to B4/F4 lines
            is_b4_forbidden = any(
                cell.get_line_pattern(Color.BLACK, dir_idx) in (Pattern.B4, Pattern.F4)
                for dir_idx in range(4)
            )
            if is_b4_forbidden:
                move_list_ref.append(ScoredMove(pos))

def _find_four_defence(board: 'Board', include_losing_moves: bool, move_list_ref: List[ScoredMove]):
    """
    Finds exact defense positions against opponent's B_FLEX4.
    Simplified version. Rapfi's is very detailed, checking specific line patterns.
    """
    # This is a highly complex function in Rapfi.
    # For a Python version, a full port of its line-tracing logic is a large sub-project.
    # Simplified: find all moves that block any F4 line of the opponent or occupy the flex4 point.
    opponent = ~board.side_to_move()
    
    # Heuristic: If opponent has B_FLEX4, there's usually a key point(s) forming it.
    # The primary defense is to occupy one of these key points.
    # A more robust way is to find the actual B_FLEX4 location(s).
    
    # Placeholder: add all positions that are B_FLEX4 for the opponent
    # This is what _find_all_pseudo_four_defend_pos does for E_BLOCK4+
    _find_all_pseudo_four_defend_pos(board, opponent, move_list_ref)
    # TODO: Implement the detailed line analysis from Rapfi if this simplification isn't enough.


def _find_b4f3_defence(board: 'Board', rule: Rule, move_list_ref: List[ScoredMove]):
    """
    Finds exact defense positions against opponent's C_BLOCK4_FLEX3.
    Simplified version.
    """
    opponent = ~board.side_to_move()
    # Heuristic: Main defense is the C_BLOCK4_FLEX3 point itself.
    # Other defenses involve blocking the B4 or F3 components.
    
    # Add the C_BLOCK4_FLEX3 point itself.
    c_block4_f3_pos = board.get_current_state_info().get_last_pattern4(opponent, Pattern4.C_BLOCK4_FLEX3)
    if c_block4_f3_pos != Pos.NONE and board.is_empty(c_block4_f3_pos): # Check if it's still a valid target
         # Verify it's still C_BLOCK4_FLEX3 for opponent at that pos
        if board.get_cell(c_block4_f3_pos).pattern4[opponent.value] == Pattern4.C_BLOCK4_FLEX3:
            move_list_ref.append(ScoredMove(c_block4_f3_pos))
            
    # TODO: Add logic from Rapfi's findF3LineDefence, findB4InLine, findAllB3CounterDefence.
    # This is very complex due to line tracing and temporary board modifications.

# --- Main Generation Functions ---

def generate_moves(board: 'Board', rule: Rule, gen_type: GenType) -> List[ScoredMove]:
    """
    Generates moves satisfying GenType. Scores are initially 0.
    Returns a new list of ScoredMove.
    """
    moves: List[ScoredMove] = []
    current_player = board.side_to_move()

    if not _pre_check_filter(board, current_player, gen_type):
        return moves # Skip generation based on pre-check

    for pos in board.iter_candidate_moves():
        if _basic_pattern_filter(board, pos, current_player, gen_type, rule):
            moves.append(ScoredMove(pos)) # Score is 0 by default
    return moves

def generate_neighbor_moves(board: 'Board', rule: Rule, gen_type: GenType,
                            center: Pos,
                            neighbor_offsets: Tuple[int, ...]) -> List[ScoredMove]:
    """Generates moves from neighbor_offsets around center, filtered by GenType."""
    moves: List[ScoredMove] = []
    current_player = board.side_to_move()

    if not _pre_check_filter(board, current_player, gen_type):
        return moves

    for offset_val in neighbor_offsets:
        neighbor_pos = center + offset_val # Pos + int (offset)
        if board.is_on_board(neighbor_pos) and \
           board.is_empty(neighbor_pos) and \
           board.get_cell(neighbor_pos).is_candidate():
            if _basic_pattern_filter(board, neighbor_pos, current_player, gen_type, rule):
                moves.append(ScoredMove(neighbor_pos))
    return moves

# --- Specialized Generators ---

def generate_winning_moves(board: 'Board', rule: Rule) -> List[ScoredMove]:
    """Generates direct winning moves (A_FIVE or B_FLEX4)."""
    moves: List[ScoredMove] = []
    current_player = board.side_to_move()
    current_state = board.get_current_state_info()

    if current_state.p4_count[current_player.value][Pattern4.A_FIVE.value] > 0:
        # Find the A_FIVE pos
        pos = current_state.get_last_pattern4(current_player, Pattern4.A_FIVE)
        if pos != Pos.NONE and board.get_cell(pos).pattern4[current_player.value] == Pattern4.A_FIVE:
             moves.append(ScoredMove(pos))
        else: # Fallback scan if state info is not precise enough or stale
            pos_scan = _find_first_pattern4_pos(board, current_player, Pattern4.A_FIVE)
            if pos_scan != Pos.NONE: moves.append(ScoredMove(pos_scan))

    elif current_state.p4_count[current_player.value][Pattern4.B_FLEX4.value] > 0:
        pos = current_state.get_last_pattern4(current_player, Pattern4.B_FLEX4)
        if pos != Pos.NONE and board.get_cell(pos).pattern4[current_player.value] == Pattern4.B_FLEX4:
            moves.append(ScoredMove(pos))
        else:
            pos_scan = _find_first_pattern4_pos(board, current_player, Pattern4.B_FLEX4)
            if pos_scan != Pos.NONE: moves.append(ScoredMove(pos_scan))
    # else: Rapfi asserts false if no winning moves found but this generator was called.
    return moves

def generate_defend_five_moves(board: 'Board', rule: Rule) -> List[ScoredMove]:
    """Generates the single defense move against opponent's A_FIVE."""
    moves: List[ScoredMove] = []
    opponent = ~board.side_to_move()
    current_state = board.get_current_state_info()

    if current_state.p4_count[opponent.value][Pattern4.A_FIVE.value] > 0:
        pos = current_state.get_last_pattern4(opponent, Pattern4.A_FIVE)
        # Validate the pos from state info
        if pos != Pos.NONE and board.is_empty(pos) and \
           board.get_cell(pos).pattern4[opponent.value] == Pattern4.A_FIVE:
            moves.append(ScoredMove(pos))
        else: # Fallback scan
            pos_scan = _find_first_pattern4_pos(board, opponent, Pattern4.A_FIVE)
            if pos_scan != Pos.NONE: moves.append(ScoredMove(pos_scan))
    return moves

def generate_defend_four_moves(board: 'Board', rule: Rule, include_losing: bool) -> List[ScoredMove]:
    """Generates defense moves for opponent's B_FLEX4."""
    moves_ref: List[ScoredMove] = []
    _find_four_defence(board, include_losing, moves_ref)
    
    # Sort and unique, then remove self-VCF moves (Rapfi logic)
    moves_ref.sort(key=lambda sm: sm.pos._pos) # Sort by raw pos value
    unique_moves: List[ScoredMove] = []
    if moves_ref:
        unique_moves.append(moves_ref[0])
        for i in range(1, len(moves_ref)):
            if moves_ref[i].pos != moves_ref[i-1].pos:
                unique_moves.append(moves_ref[i])
    
    final_moves = [
        sm for sm in unique_moves 
        if board.get_cell(sm.pos).pattern4[board.side_to_move().value].value < Pattern4.E_BLOCK4.value
    ]
    return final_moves

def generate_defend_b4f3_moves(board: 'Board', rule: Rule) -> List[ScoredMove]:
    """Generates defense moves for opponent's C_BLOCK4_FLEX3."""
    moves_ref: List[ScoredMove] = []
    _find_b4f3_defence(board, rule, moves_ref)

    if not moves_ref: # No direct defense needed (e.g., we have a B4 counter)
        return []

    moves_ref.sort(key=lambda sm: sm.pos._pos)
    unique_moves: List[ScoredMove] = []
    if moves_ref:
        unique_moves.append(moves_ref[0])
        for i in range(1, len(moves_ref)):
            if moves_ref[i].pos != moves_ref[i-1].pos:
                unique_moves.append(moves_ref[i])
    
    final_moves = [
        sm for sm in unique_moves
        if board.get_cell(sm.pos).pattern4[board.side_to_move().value].value < Pattern4.E_BLOCK4.value
    ]
    return final_moves

# Dispatcher or main generate function
def generate_all_moves(board: 'Board', rule: Rule, gen_type_flags: GenType) -> List[ScoredMove]:
    """
    Master move generator that dispatches to specialized generators or generic one.
    """
    # Handle specific DEFEND types first as they are very targeted
    if gen_type_flags & GenType.DEFEND_FIVE:
        return generate_defend_five_moves(board, rule)
    if gen_type_flags & GenType.DEFEND_FOUR:
        # Rapfi template: generate<DEFEND_FOUR | ALL> or generate<DEFEND_FOUR>
        include_losing = bool(gen_type_flags & GenType.ALL) # Bit 'ALL' in C++ GenType is ambiguous here
                                                            # Rapfi uses specific template specializations.
                                                            # Let's assume if GenType.TRIVIAL is part of a general "ALL" it means include losing.
        return generate_defend_four_moves(board, rule, include_losing)
    if gen_type_flags & GenType.DEFEND_B4F3:
        return generate_defend_b4f3_moves(board, rule)
    
    # Handle WINNING type
    if gen_type_flags & GenType.WINNING: # If ONLY winning is requested, or as part of broader
        # If other flags like VCF, VCT are also set, Rapfi might generate those too.
        # The C++ generate<WINNING> is a template specialization that *only* returns winning moves.
        # If gen_type is exactly GenType.WINNING
        if gen_type_flags == GenType.WINNING:
             return generate_winning_moves(board, rule)
        # If WINNING is part of a larger set (e.g., GenType.ALL), the generic generate_moves
        # below will pick up winning moves via _basic_pattern_filter.

    # Generic generation based on flags
    return generate_moves(board, rule, gen_type_flags)


if __name__ == '__main__':
    from .board import Board # Full import for testing
    
    print("--- Movegen Tests ---")
    board = Board(15, CandidateRange.SQUARE2)
    board.new_game(Rule.FREESTYLE)

    print("Generating ALL moves on empty board (FS):")
    all_moves_empty = generate_all_moves(board, Rule.FREESTYLE, GenType.ALL)
    # On empty board, candidates are usually around center or defined by cand_area_expand_dist
    # _basic_pattern_filter with TRIVIAL should pick them up.
    # For a fresh board with new_game, cand_area is empty unless full board.
    # Let's expand candidate area for testing.
    board.expand_cand_area(board.center_pos(), 2, 2) # Expand 5x5 around center
    
    all_moves_empty_expanded = generate_all_moves(board, Rule.FREESTYLE, GenType.ALL)
    print(f"Found {len(all_moves_empty_expanded)} moves (ALL type) after cand expand.")
    # Expected number depends on how many cells are marked candidate by expand_cand_area
    # A 5x5 area has 25 cells.
    assert len(all_moves_empty_expanded) <= 25 
    if all_moves_empty_expanded:
        print(f"Sample: {all_moves_empty_expanded[0]}")

    # Simulate a situation for WINNING moves
    # Manually set up a B_FLEX4 for black at a hypothetical pos (e.g. by mocking cell patterns)
    # This is hard without making actual moves or deep board manipulation.
    # For now, test that the call doesn't crash.
    
    # Make a few moves to create some patterns
    board.make_move(Rule.FREESTYLE, Pos(7,7)) # B
    board.make_move(Rule.FREESTYLE, Pos(7,8)) # W
    board.make_move(Rule.FREESTYLE, Pos(6,7)) # B
    board.make_move(Rule.FREESTYLE, Pos(6,8)) # W
    board.make_move(Rule.FREESTYLE, Pos(5,7)) # B
    board.make_move(Rule.FREESTYLE, Pos(5,8)) # W
    board.make_move(Rule.FREESTYLE, Pos(4,7)) # B - Black has a line of 4 (X.XXXX) -> B_FLEX4 at (3,7) or (8,7)
                                            # or A_FIVE at (3,7) if (3,7) completes it.

    print("\nBoard state for winning/defend tests:")
    print(board.to_string())
    print(f"Ply: {board.ply()}, Side to move: {board.current_side.name}") # White to move (opponent of Black's B4)

    winning_for_black_hypothetical = []
    if board.get_current_state_info().p4_count[Color.BLACK.value][Pattern4.B_FLEX4.value] > 0:
         # If black were to move, what winning moves?
        # We'd need to temporarily flip side to test this path of generate_winning_moves
        print(f"Black has B_FLEX4 count: {board.get_current_state_info().p4_count[Color.BLACK.value][Pattern4.B_FLEX4.value]}")
        # winning_for_black_hypothetical = generate_winning_moves(board_temp_black_turn, Rule.FREESTYLE)
        # print(f"Hypothetical winning moves for Black: {winning_for_black_hypothetical}")


    # Current side is White. Check if Black (opponent) has A_FIVE for DEFEND_FIVE test
    if board.get_current_state_info().p4_count[Color.BLACK.value][Pattern4.A_FIVE.value] > 0:
        defend_five_w = generate_all_moves(board, Rule.FREESTYLE, GenType.DEFEND_FIVE)
        print(f"DEFEND_FIVE moves for White: {defend_five_w}")
        assert len(defend_five_w) > 0

    # Check if Black (opponent) has B_FLEX4 for DEFEND_FOUR test
    if board.get_current_state_info().p4_count[Color.BLACK.value][Pattern4.B_FLEX4.value] > 0:
        defend_four_w = generate_all_moves(board, Rule.FREESTYLE, GenType.DEFEND_FOUR)
        print(f"DEFEND_FOUR moves for White: {defend_four_w}")
        # This will depend on the simplified _find_four_defence.
        # It should find moves that would block Black's flex four.
        # If black made X.XXXX at (4,7)-(7,7), then (3,7) and (8,7) are flex points.
        # White (current player) should generate moves at (3,7) and (8,7).
        # The current simplified _find_four_defence might just return these points if they are B_FLEX4 for opponent.
        
        # Assertions here are tricky without knowing the exact outcome of simplified defense logic.
        # For a line X _ X X X _ X (B to move, assuming _ are empty, X are B)
        # Rapfi's B_FLEX4 means opponent (W) has to defend one of the _ to prevent immediate loss.
        # The example moves made: B at (77)(67)(57)(47).
        # This is B B B B. Empty cells around it are candidates for A_FIVE for Black.
        # (3,7) and (8,7) are potential winning spots for Black.
        # White (current) must play at (3,7) or (8,7).
        # So, DEFEND_FOUR for White should ideally generate ScoredMove(Pos(3,7)) and ScoredMove(Pos(8,7)).
        found_3_7 = any(m.pos == Pos(3,7) for m in defend_four_w)
        found_8_7 = any(m.pos == Pos(8,7) for m in defend_four_w)
        if not (found_3_7 and found_8_7) and len(defend_four_w) > 0: # Len >0 if any defense found
            print(f"Note: DEFEND_FOUR did not find both (3,7) and (8,7). Found: {defend_four_w}")
            print("This might be due to simplified defense logic or actual board state.")

    print("Movegen tests completed (basic calls).")