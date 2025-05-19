"""
Quick win/loss checking functions for the Gomoku engine.
Based on Rapfi's wincheck.h.
"""
from typing import TYPE_CHECKING

from .types import Value, Color, Rule, Pattern4, mate_in, mated_in
from .pos import Pos
# Import Board and StateInfo for type checking
if TYPE_CHECKING:
    from .board import Board
    from .board_utils import StateInfo


def quick_win_check(board: 'Board', rule: Rule, ply: int, beta: int = Value.VALUE_INFINITE.value) -> int:
    """
    Quickly checks if the current position is an immediate win or loss.
    Args:
        board: The current board state.
        rule: The game rule being played.
        ply: The current search ply (depth in the search tree).
        beta: Current beta value (used to relax some checks in Renju).
    Returns:
        A Value enum member (e.g., mate_in(X), mated_in(X), VALUE_ZERO).
    """
    current_player: Color = board.side_to_move()
    opponent: Color = ~current_player
    current_state: 'StateInfo' = board.get_current_state_info() # Get actual StateInfo

    # 1. Current player makes A_FIVE (Win in 1 ply from current state, so ply + 1 total from root of this sub-search)
    if current_state.p4_count[current_player.value][Pattern4.A_FIVE.value] > 0:
        return mate_in(ply + 1)

    # 2. Opponent has A_FIVE threats
    opponent_a_five_count = current_state.p4_count[opponent.value][Pattern4.A_FIVE.value]
    if opponent_a_five_count > 0:
        if opponent_a_five_count == 1:
            # Single A_FIVE for opponent means we might be able to defend. Not an immediate loss here.
            return Value.VALUE_ZERO 
        else: # More than one A_FIVE for opponent (double five, etc.)
            # Loss in 2 plies (opponent makes their winning move next)
            return mated_in(ply + 2) 

    # 3. Current player has B_FLEX4 (Straight/Flex Four)
    # This leads to a win in 3 plies (current move creates B_FLEX4, next move makes A_FIVE)
    if current_state.p4_count[current_player.value][Pattern4.B_FLEX4.value] > 0:
        return mate_in(ply + 3)

    # Helper to count block fours for a side
    def count_block_fours(side: Color) -> int:
        return (current_state.p4_count[side.value][Pattern4.C_BLOCK4_FLEX3.value] +
                current_state.p4_count[side.value][Pattern4.D_BLOCK4_PLUS.value] +
                current_state.p4_count[side.value][Pattern4.E_BLOCK4.value])

    # Rapfi's check: if (Rule != Rule::RENJU || mate_in(ply + 5) >= beta)
    # This means for Renju, these deeper checks are only done if the potential win is good enough.
    perform_deeper_checks = True
    if rule == Rule.RENJU: # This means for both Black and White playing Renju
        if not (mate_in(ply + 5) >= beta): # This is equivalent to mate_in(ply+5) < beta
            perform_deeper_checks = False

    if perform_deeper_checks:
        self_c_block4_flex3_count = current_state.p4_count[current_player.value][Pattern4.C_BLOCK4_FLEX3.value]
        if self_c_block4_flex3_count > 0:
            can_win_by_c_type = True
            if rule == Rule.RENJU and current_player == Color.BLACK:
                can_win_by_c_type = False
            
            opponent_block_fours = count_block_fours(opponent) # Get the count

            if can_win_by_c_type and opponent_block_fours == 0: # Use the fetched count
                return mate_in(ply + 5)            
            pass

        # 5. Current player has F_FLEX3_2X (Double Flex Three)
        # Can lead to a win in 5 plies if opponent has no B4 or other strong blocks.
        if current_state.p4_count[current_player.value][Pattern4.F_FLEX3_2X.value] > 0:
            if current_state.p4_count[opponent.value][Pattern4.B_FLEX4.value] == 0 and \
               count_block_fours(opponent) == 0:
                return mate_in(ply + 5)

    return Value.VALUE_ZERO # No quick win/loss found

# Dynamic dispatch version (not strictly needed if calling typed version directly)
# def quick_win_check_dispatch(board: 'Board', rule: Rule, ply: int, beta: Value = Value.VALUE_INFINITE) -> Value:
#     return quick_win_check(board, rule, ply, beta)


if __name__ == '__main__':
    from .board import Board 
    from .board_utils import StateInfo 
    from .pos import Pos
    # from .types import Rule, Color, Value, Pattern4, mate_in, mated_in # Already imported at top
    from . import config as engine_config 

    print("--- Wincheck Tests ---")

    board = Board(15) 
    board.new_game(Rule.FREESTYLE) 
    current_ply = 0 

    # Helper to get a "name" for an integer score if it matches a Value enum member
    def get_score_name(score_int: int) -> str:
        try:
            return Value(score_int).name # Try to convert to Value enum to get its name
        except ValueError:
            # If it's not a direct enum member (e.g., mate_in(100) = 29900)
            if score_int > Value.VALUE_EVAL_MAX.value: # Heuristic for mate
                 # Find closest mate_in value if possible, or just describe
                for i in range(500): # Check a range of plies
                    if score_int == (Value.VALUE_MATE.value - i):
                        return f"mate_in({i})"
                return f"STRONG_WIN({score_int})"
            elif score_int < Value.VALUE_EVAL_MIN.value: # Heuristic for mated
                for i in range(500):
                    if score_int == (-Value.VALUE_MATE.value + i):
                        return f"mated_in({i})"
                return f"STRONG_LOSS({score_int})"
            return f"SCORE({score_int})"


    # Test 1: Current player (Black) has A_FIVE
    print("\nTest 1: Black has A_FIVE")
    si = board.get_current_state_info()
    si.p4_count[Color.BLACK.value][Pattern4.A_FIVE.value] = 1
    result_int = quick_win_check(board, Rule.FREESTYLE, current_ply)
    expected_int = mate_in(current_ply + 1)
    print(f"Result: {get_score_name(result_int)} ({result_int}), Expected: {get_score_name(expected_int)} ({expected_int})")
    assert result_int == expected_int
    si.p4_count[Color.BLACK.value][Pattern4.A_FIVE.value] = 0

    # Test 2: Opponent (White) has one A_FIVE
    print("\nTest 2: White has one A_FIVE (Black to move)")
    si.p4_count[Color.WHITE.value][Pattern4.A_FIVE.value] = 1
    result_int = quick_win_check(board, Rule.FREESTYLE, current_ply)
    expected_int = Value.VALUE_ZERO.value # VALUE_ZERO is an enum member, so .value is its int value
    print(f"Result: {get_score_name(result_int)} ({result_int}), Expected: {get_score_name(expected_int)} ({expected_int})")
    assert result_int == expected_int
    si.p4_count[Color.WHITE.value][Pattern4.A_FIVE.value] = 0

    # Test 3: Opponent (White) has two A_FIVEs
    print("\nTest 3: White has two A_FIVEs (Black to move)")
    si.p4_count[Color.WHITE.value][Pattern4.A_FIVE.value] = 2
    result_int = quick_win_check(board, Rule.FREESTYLE, current_ply)
    expected_int = mated_in(current_ply + 2)
    print(f"Result: {get_score_name(result_int)} ({result_int}), Expected: {get_score_name(expected_int)} ({expected_int})")
    assert result_int == expected_int
    si.p4_count[Color.WHITE.value][Pattern4.A_FIVE.value] = 0

    # Test 4: Current player (Black) has B_FLEX4
    print("\nTest 4: Black has B_FLEX4")
    si.p4_count[Color.BLACK.value][Pattern4.B_FLEX4.value] = 1
    result_int = quick_win_check(board, Rule.FREESTYLE, current_ply)
    expected_int = mate_in(current_ply + 3)
    print(f"Result: {get_score_name(result_int)} ({result_int}), Expected: {get_score_name(expected_int)} ({expected_int})")
    assert result_int == expected_int
    si.p4_count[Color.BLACK.value][Pattern4.B_FLEX4.value] = 0

    # Test 5: Deeper checks (C_BLOCK4_FLEX3 for Black, Freestyle, no oppo block)
    print("\nTest 5: Black C_BLOCK4_FLEX3, FS, no oppo block")
    si.p4_count[Color.BLACK.value][Pattern4.C_BLOCK4_FLEX3.value] = 1
    for p4_enum_val in range(Pattern4.C_BLOCK4_FLEX3.value, Pattern4.E_BLOCK4.value + 1):
        si.p4_count[Color.WHITE.value][p4_enum_val] = 0
    result_int = quick_win_check(board, Rule.FREESTYLE, current_ply)
    expected_int = mate_in(current_ply + 5)
    print(f"Result: {get_score_name(result_int)} ({result_int}), Expected: {get_score_name(expected_int)} ({expected_int})")
    assert result_int == expected_int
    si.p4_count[Color.BLACK.value][Pattern4.C_BLOCK4_FLEX3.value] = 0

    # Test 6: Deeper checks (F_FLEX3_2X for Black, FS, no oppo block)
    print("\nTest 6: Black F_FLEX3_2X, FS, no oppo block")
    si.p4_count[Color.BLACK.value][Pattern4.F_FLEX3_2X.value] = 1
    si.p4_count[Color.WHITE.value][Pattern4.B_FLEX4.value] = 0
    for p4_enum_val in range(Pattern4.C_BLOCK4_FLEX3.value, Pattern4.E_BLOCK4.value + 1):
        si.p4_count[Color.WHITE.value][p4_enum_val] = 0
    result_int = quick_win_check(board, Rule.FREESTYLE, current_ply)
    expected_int = mate_in(current_ply + 5)
    print(f"Result: {get_score_name(result_int)} ({result_int}), Expected: {get_score_name(expected_int)} ({expected_int})")
    assert result_int == expected_int
    si.p4_count[Color.BLACK.value][Pattern4.F_FLEX3_2X.value] = 0

    # Test 7: Renju rule, deeper checks skipped if beta is too low
    print("\nTest 7: Renju, Black C_BLOCK4_FLEX3, low beta")
    si.p4_count[Color.BLACK.value][Pattern4.C_BLOCK4_FLEX3.value] = 1
    # beta is an int. Value.VALUE_INFINITE.value is an int.
    low_beta_int = mate_in(current_ply + 6) 
    result_int = quick_win_check(board, Rule.RENJU, current_ply, low_beta_int)
    expected_int = Value.VALUE_ZERO.value
    print(f"Result: {get_score_name(result_int)} ({result_int}), Expected: {get_score_name(expected_int)} ({expected_int})")
    assert result_int == expected_int 
    si.p4_count[Color.BLACK.value][Pattern4.C_BLOCK4_FLEX3.value] = 0
    
    # Test 8: Renju rule, deeper checks performed if beta is high enough
    print("\nTest 8: Renju, White C_BLOCK4_FLEX3, TIGHTER beta, no oppo block")
    board.current_side = Color.WHITE 
    si_white_turn = board.get_current_state_info() 
    
    # Reset relevant p4_counts from previous tests on this shared si_white_turn object
    for p4_val in Pattern4:
        si_white_turn.p4_count[Color.WHITE.value][p4_val.value] = 0
        si_white_turn.p4_count[Color.BLACK.value][p4_val.value] = 0

    si_white_turn.p4_count[Color.WHITE.value][Pattern4.C_BLOCK4_FLEX3.value] = 1
    # Ensure no Black (opponent) block4s that would prevent the C-type win
    # (count_block_fours(opponent) checks C_BLOCK4_FLEX3, D_BLOCK4_PLUS, E_BLOCK4 for opponent)
    # These are already zeroed out by the loop above.
    
    # Set beta to exactly mate_in(ply+5) to ensure deeper checks are performed
    tight_beta_int = mate_in(current_ply + 5) # 29995
    
    result_int = quick_win_check(board, Rule.RENJU, current_ply, tight_beta_int)
    expected_int = mate_in(current_ply + 5) # Expecting White to win
    print(f"Result: {get_score_name(result_int)} ({result_int}), Expected: {get_score_name(expected_int)} ({expected_int})")
    assert result_int == expected_int 
    
    # Cleanup for this test case
    si_white_turn.p4_count[Color.WHITE.value][Pattern4.C_BLOCK4_FLEX3.value] = 0
    board.current_side = Color.BLACK

    print("\nWincheck tests completed.")