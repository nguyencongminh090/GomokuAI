"""
Classical evaluation functions for the Gomoku engine,
based on Rapfi's eval.h and eval.cpp.
"""

from .types import Value, Color, Rule, Pattern4, SIDE_NB, PATTERN4_NB
from .config import PCODE_NB, CandidateRange # PCODE_NB for EvalInfo
from .pos import Pos

# Import actual Board and StateInfo from board.py and board_utils.py
from .board import Board # Assuming board.py defines Board
from .board_utils import StateInfo # Assuming board_utils.py defines StateInfo

from . import config as engine_config

# THREAT_NB is defined in engine_config
# THREAT_CONDITION_COUNT = 11 # Not strictly needed if THREAT_NB from config is used

def make_threat_mask(st: StateInfo, current_player: Color) -> int: # Takes actual StateInfo
    """Makes threat mask according current pattern4 counts on board."""
    opponent = ~current_player

    # Accessing p4_count from the actual StateInfo object
    oppo_five = st.p4_count[opponent.value][Pattern4.A_FIVE.value] > 0
    self_flex_four = st.p4_count[current_player.value][Pattern4.B_FLEX4.value] > 0
    oppo_flex_four = st.p4_count[opponent.value][Pattern4.B_FLEX4.value] > 0
    self_four_plus = (st.p4_count[current_player.value][Pattern4.D_BLOCK4_PLUS.value] +
                      st.p4_count[current_player.value][Pattern4.C_BLOCK4_FLEX3.value]) > 0
    self_four = st.p4_count[current_player.value][Pattern4.E_BLOCK4.value] > 0
    self_three_plus = (st.p4_count[current_player.value][Pattern4.G_FLEX3_PLUS.value] +
                       st.p4_count[current_player.value][Pattern4.F_FLEX3_2X.value]) > 0
    self_three = st.p4_count[current_player.value][Pattern4.H_FLEX3.value] > 0
    oppo_four_plus = (st.p4_count[opponent.value][Pattern4.D_BLOCK4_PLUS.value] +
                      st.p4_count[opponent.value][Pattern4.C_BLOCK4_FLEX3.value]) > 0
    oppo_four = st.p4_count[opponent.value][Pattern4.E_BLOCK4.value] > 0
    oppo_three_plus = (st.p4_count[opponent.value][Pattern4.G_FLEX3_PLUS.value] +
                       st.p4_count[opponent.value][Pattern4.F_FLEX3_2X.value]) > 0
    oppo_three = st.p4_count[opponent.value][Pattern4.H_FLEX3.value] > 0

    mask = 0
    if oppo_five:        mask |= (1 << 0)
    if self_flex_four:   mask |= (1 << 1)
    if oppo_flex_four:   mask |= (1 << 2)
    if self_four_plus:   mask |= (1 << 3)
    if self_four:        mask |= (1 << 4)
    if self_three_plus:  mask |= (1 << 5)
    if self_three:       mask |= (1 << 6)
    if oppo_four_plus:   mask |= (1 << 7)
    if oppo_four:        mask |= (1 << 8)
    if oppo_three_plus:  mask |= (1 << 9)
    if oppo_three:       mask |= (1 << 10)
    assert 0 <= mask < engine_config.THREAT_NB, f"Threat mask {mask} out of range {engine_config.THREAT_NB}"
    return mask

def _get_threat_eval_table_for_rule(rule: Rule, current_player: Color) -> list[int]:
    """Helper to get the threat evaluation table from config."""
    idx = engine_config.table_index(rule, current_player)

    if 0 <= idx < len(engine_config.EVALS_THREAT):
        table = engine_config.EVALS_THREAT[idx]
        # Ensure table has correct size and content type (int)
        if len(table) == engine_config.THREAT_NB:
            if table and isinstance(table[0], Value): # Should not happen if config is correct
                return [int(v) for v in table]
            return table
    return [0] * engine_config.THREAT_NB


def evaluate_threat(st: StateInfo, current_player: Color, rule: Rule) -> int: # Takes actual StateInfo
    """Evaluates threats."""
    threat_table = _get_threat_eval_table_for_rule(rule, current_player)
    mask = make_threat_mask(st, current_player) # st is now actual StateInfo
    if mask >= len(threat_table):
        return 0
    return threat_table[mask]

def evaluate_basic(st: StateInfo, current_player: Color) -> int: # Takes actual StateInfo
    """Evaluates basic patterns on board from current_player's perspective."""
    # st.value_black from actual StateInfo should already be an int
    val_black = st.value_black
    return val_black if current_player == Color.BLACK else -val_black

def evaluate(board: Board, rule: Rule, # Takes actual Board
             alpha: int = Value.VALUE_EVAL_MIN.value,
             beta: int = Value.VALUE_EVAL_MAX.value
            ) -> int:
    """Calculates the final classical evaluation of a board."""
    current_player = board.side_to_move() # Uses actual board method
    eval_score_intermediate: int

    if board.ply() <= 0:
        st0 = board.get_current_state_info() # Uses actual board method
        basic_eval = evaluate_basic(st0, current_player)
        threat_eval_val = evaluate_threat(st0, current_player, rule)
        eval_score_intermediate = basic_eval + threat_eval_val
    else:
        # board.state_infos is the list, get_current_state_info() is state_infos[ply]
        # Rapfi: st0 = board.stateInfo(); st1 = board.stateInfo(1);
        # In Python Board, state_infos[ply] is current, state_infos[ply-1] is previous.
        # The `board.stateInfo(1)` from C++ is state *before* opponent's last move.
        # If current ply is P, current state is state_infos[P].
        # State before opponent's last move (which led to state_infos[P]) is state_infos[P-1].
        # So, for our Python Board:
        # st0 is self.state_infos[self.move_count]
        # st1 is self.state_infos[self.move_count - 1] (if move_count > 0)
        st0 = board.get_current_state_info()
        if board.ply() > 0 :
             st1 = board.state_infos[board.ply() -1] # Direct access for simplicity here
        else: # Should not happen due to outer if, but for safety
            st1 = st0 

        basic_eval_st0 = evaluate_basic(st0, current_player)
        basic_eval_st1 = evaluate_basic(st1, current_player) # Evaluate previous state from current player's view
        
        avg_basic_eval = (basic_eval_st0 + basic_eval_st1) // 2
        threat_eval_val = evaluate_threat(st0, current_player, rule) # Threat based on current state
        eval_score_intermediate = avg_basic_eval + threat_eval_val
    
    clamped_eval = max(Value.VALUE_EVAL_MIN.value, min(eval_score_intermediate, Value.VALUE_EVAL_MAX.value))

    # Rapfi's evaluator() check and classicalEvalMargin logic is skipped for now
    # if board.evaluator_instance: (using the name from our Python board.py)
    #     # ... logic involving EvaluatorMargin etc. ...
    #     pass

    return clamped_eval


class EvalInfo: # Used for more detailed evaluation (e.g. for training/tuning)
    """EvalInfo struct contains all information needed to evaluate a position."""
    def __init__(self, board: Board, rule: Rule): # Takes actual Board
        self.current_player: Color = board.side_to_move()
        st0 = board.get_current_state_info()
        self.threat_mask: int = make_threat_mask(st0, self.current_player)

        # Uses PCODE_NB from .config (which is the compressed pattern code count)
        self.ply_back_pcode_counts: list[list[list[int]]] = [
            [[0] * PCODE_NB for _ in range(SIDE_NB)] for _ in range(2)
        ]
        
        # Populating pcode_counts requires iterating over the board cells and
        # getting their pattern codes. This needs the actual Board and Cell.
        # Rapfi's EvalInfo constructor does temporary undos to get previous state's pcodes.
        
        # board_clone = board.clone() # Assuming a clone method exists or make one for temp ops
        # For now, this part remains complex to implement without a board clone or safe temp undo.
        
        # Example of populating for current state (ply_back_idx = 0)
        current_board_state_pcodes = self.ply_back_pcode_counts[0]
        for pos_iter in board.iter_empty_positions(): # Iterate actual empty positions
            cell = board.get_cell(pos_iter) # Get actual cell
            # get_pcode is a method of Cell in board_utils.py
            current_board_state_pcodes[Color.BLACK.value][cell.get_pcode(Color.BLACK)] +=1
            current_board_state_pcodes[Color.WHITE.value][cell.get_pcode(Color.WHITE)] +=1
            
        # Populating for previous state (ply_back_idx = 1) would require:
        if board.ply() > 0:
            # 1. Access to the previous board state (e.g. by temporary undo or history)
            # This is non-trivial if the board object itself is modified.
            # If Board stores history of Cell states or pcodes, it's easier.
            # Rapfi: Board &b = const_cast<Board &>(board); ... b.undo(rule); ... b.move(rule, ...);
            # This implies EvalInfo is created with a board that can be temporarily modified.
            # For now, we can't easily do the temporary undo here.
            # A board.clone() method would be ideal for this.
            pass


if __name__ == '__main__':
    from .board import Board # For testing with actual Board
    from .board_utils import StateInfo # For direct manipulation in tests if needed

    print("--- Evaluation Tests (with actual Board/StateInfo) ---")
    
    # Setup dummy EVALS_THREAT in engine_config for testing
    idx_fs_b = engine_config.table_index(Rule.FREESTYLE, Color.BLACK)
    idx_std_b = engine_config.table_index(Rule.STANDARD, Color.BLACK)
    idx_renju_b = engine_config.table_index(Rule.RENJU, Color.BLACK)
    idx_renju_w = engine_config.table_index(Rule.RENJU, Color.WHITE)

    engine_config.EVALS_THREAT[idx_fs_b] = [i * 10 for i in range(engine_config.THREAT_NB)]
    engine_config.EVALS_THREAT[idx_std_b] = [i * 8 for i in range(engine_config.THREAT_NB)]
    engine_config.EVALS_THREAT[idx_renju_b] = [i * 12 for i in range(engine_config.THREAT_NB)]
    engine_config.EVALS_THREAT[idx_renju_w] = [i * 6 for i in range(engine_config.THREAT_NB)]

    # Create a board instance for testing
    test_board = Board(15, CandidateRange.SQUARE2) # Default cand range
    test_board.new_game(Rule.FREESTYLE) # Initialize with a rule

    # --- Test make_threat_mask with actual StateInfo from board ---
    # Get current state (empty board, Black to move)
    current_si = test_board.get_current_state_info()
    current_si.p4_count[Color.WHITE.value][Pattern4.A_FIVE.value] = 1 # Hypothetical opponent five
    current_si.p4_count[Color.BLACK.value][Pattern4.B_FLEX4.value] = 1 # Hypothetical self flex four
    
    mask = make_threat_mask(current_si, Color.BLACK)
    print(f"Calculated threat mask for BLACK on board: {mask:011b}")
    expected_mask = (1 << 0) | (1 << 1) # oppo_five | self_flex_four
    assert mask == expected_mask

    threat_val_fs_b = evaluate_threat(current_si, Color.BLACK, Rule.FREESTYLE)
    print(f"Threat value for mask {mask} (FS, B): {threat_val_fs_b}")
    assert threat_val_fs_b == engine_config.EVALS_THREAT[idx_fs_b][mask]
    
    # Reset p4_counts for next test if needed, or use a fresh StateInfo/Board
    current_si.p4_count[Color.WHITE.value][Pattern4.A_FIVE.value] = 0
    current_si.p4_count[Color.BLACK.value][Pattern4.B_FLEX4.value] = 0


    # --- Test evaluate_basic with actual StateInfo ---
    current_si.value_black = 100 # Set a base eval for Black's view
    
    eval_b = evaluate_basic(current_si, Color.BLACK)
    eval_w = evaluate_basic(current_si, Color.WHITE)
    print(f"Basic eval for BLACK: {eval_b}, for WHITE: {eval_w}")
    assert eval_b == 100
    assert eval_w == -100
    current_si.value_black = 0 # Reset

    # --- Test full evaluate function with actual Board ---
    # Board is at ply 0, BLACK to move. new_game has calculated initial value_black and p4_counts.
    print(f"\nInitial board (ply 0, B to move) value_black: {test_board.get_current_state_info().value_black}")
    eval_val_ply0 = evaluate(test_board, Rule.FREESTYLE)
    print(f"Evaluation at ply 0 for BLACK (FS): {eval_val_ply0}")
    
    # Expected: basic_eval from state_info(0).value_black
    # + threat_val from state_info(0).p4_counts for Black.
    st0_ply0 = test_board.get_current_state_info()
    expected_basic_ply0 = evaluate_basic(st0_ply0, Color.BLACK)
    expected_threat_ply0 = evaluate_threat(st0_ply0, Color.BLACK, Rule.FREESTYLE)
    expected_eval_ply0_calc = expected_basic_ply0 + expected_threat_ply0
    clamped_expected_eval_ply0 = max(Value.VALUE_EVAL_MIN.value, min(expected_eval_ply0_calc, Value.VALUE_EVAL_MAX.value))
    assert eval_val_ply0 == clamped_expected_eval_ply0
    print(f"  (Components: basic={expected_basic_ply0}, threat={expected_threat_ply0})")


    # Make a move: Black plays at center
    center_pos = Pos(7,7)
    test_board.make_move(Rule.FREESTYLE, center_pos) # Now ply 1, White to move

    print(f"\nBoard after B plays at {center_pos} (ply 1, W to move)")
    print(f"  Current player: {test_board.side_to_move().name}")
    print(f"  StateInfo value_black: {test_board.get_current_state_info().value_black}") # Eval of board from B's view

    eval_val_ply1_white = evaluate(test_board, Rule.FREESTYLE)
    print(f"Evaluation at ply 1 for WHITE (FS): {eval_val_ply1_white}")

    # Expected for ply 1 (White to move):
    # st0 is state_infos[1] (after Black's move)
    # st1 is state_infos[0] (empty board state)
    st0_ply1 = test_board.get_current_state_info() # State after Black's move at (7,7)
    st1_ply1 = test_board.state_infos[0]          # Initial empty board state

    basic_st0_white = evaluate_basic(st0_ply1, Color.WHITE) # -st0_ply1.value_black
    basic_st1_white = evaluate_basic(st1_ply1, Color.WHITE) # -st1_ply1.value_black
    avg_basic_white = (basic_st0_white + basic_st1_white) // 2
    
    threat_st0_white = evaluate_threat(st0_ply1, Color.WHITE, Rule.FREESTYLE) # Threat for White on current board
    
    expected_eval_ply1_white_calc = avg_basic_white + threat_st0_white
    clamped_expected_eval_ply1_white = max(Value.VALUE_EVAL_MIN.value, min(expected_eval_ply1_white_calc, Value.VALUE_EVAL_MAX.value))
    assert eval_val_ply1_white == clamped_expected_eval_ply1_white
    print(f"  (Components for White: basic_st0={basic_st0_white}, basic_st1={basic_st1_white}, avg_basic={avg_basic_white}, threat_st0={threat_st0_white})")


    # Test EvalInfo instantiation (partially, pcode_counts for previous ply is complex)
    try:
        eval_info_test = EvalInfo(test_board, Rule.FREESTYLE)
        print(f"\nEvalInfo created for current board state (ply {test_board.ply()})")
        print(f"  EvalInfo threat_mask: {eval_info_test.threat_mask:011b}")
        # Check if pcode_counts for current state (idx 0) got populated
        pcode_sum_b = sum(eval_info_test.ply_back_pcode_counts[0][Color.BLACK.value])
        pcode_sum_w = sum(eval_info_test.ply_back_pcode_counts[0][Color.WHITE.value])
        print(f"  EvalInfo pcode_counts[0] sums: Black={pcode_sum_b}, White={pcode_sum_w}")
        # Number of empty cells on a 15x15 board after 1 move is 225-1 = 224.
        # Each empty cell contributes one pcode for B and one for W.
        assert pcode_sum_b == (test_board.board_size * test_board.board_size - test_board.ply())
        assert pcode_sum_w == (test_board.board_size * test_board.board_size - test_board.ply())

    except Exception as e:
        print(f"Error during EvalInfo test: {e}")


    print("\nEvaluation tests (with actual Board/StateInfo) completed.")