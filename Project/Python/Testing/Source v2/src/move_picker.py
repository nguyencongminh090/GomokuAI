"""
Move picker for ordering moves during search.
Based on Rapfi's movepick.h.
"""
from __future__ import annotations 
from typing import List, Optional, Tuple, TYPE_CHECKING, Callable, Any
from enum import Enum, IntFlag 

from .types import Color, Rule, Score, Value, Pattern4
from .pos import Pos
from .board_utils import Cell 

# Import the movegen module itself to call its functions qualified
from . import movegen as movegen_module 
# Import specific classes/enums from movegen if they are used as types or directly
from .movegen import ScoredMove, GenType 

from .history import MainHistory, CounterMoveHistory, CounterMovePair # Assuming CounterMovePair is defined in history
from .search_stack import StackEntry 

if TYPE_CHECKING:
    from .board import Board


class PickStage(Enum):
    UNINITIALIZED = 0
    TT_MOVE = 1
    GENERATE_WINNING_FORCING = 2 
    WINNING_MOVES = 3            
    DEFEND_OPPO_A5 = 4           
    DEFEND_OPPO_B4 = 5           
    KILLER_MOVES = 6
    COUNTER_MOVE = 7
    GENERATE_REMAINING = 8       
    SORTED_QUIET_MOVES = 9       
    LOSING_CAPTURES_EVASIONS = 10 
    DONE = 11

class MovePicker:
    def __init__(self, board: 'Board', rule: Rule,
                 tt_move: Pos, 
                 main_history: MainHistory,
                 counter_move_history: CounterMoveHistory,
                 stack_entry: StackEntry, 
                 stack_entry_minus_1: Optional[StackEntry] = None, 
                 stack_entry_minus_2: Optional[StackEntry] = None, 
                 is_pv_node: bool = False,
                 skip_quiet_threshold: Optional[Score] = None):

        self.board: 'Board' = board
        self.rule: Rule = rule
        self.tt_move: Pos = tt_move
        self.main_history: MainHistory = main_history
        self.counter_move_history: CounterMoveHistory = counter_move_history
        self.ss: StackEntry = stack_entry 
        self.ss_m1: Optional[StackEntry] = stack_entry_minus_1 
        self.ss_m2: Optional[StackEntry] = stack_entry_minus_2 
        
        self.current_player: Color = board.side_to_move()
        self.opponent: Color = ~self.current_player

        self.stage: PickStage = PickStage.TT_MOVE
        self.current_idx: int = 0 
        
        self._winning_moves: List[ScoredMove] = []
        self._defend_a5_moves: List[ScoredMove] = []
        self._defend_b4_moves: List[ScoredMove] = []
        self._quiet_moves: List[ScoredMove] = [] 

        self.skip_quiet_threshold: Optional[Score] = skip_quiet_threshold

    def _score_quiet_moves(self):
        """Scores quiet moves using main_history and sorts them."""
        for sm in self._quiet_moves:
            sm.score = self.main_history.get_score(self.current_player, sm.pos)
        
        # Sorts descending by score attribute
        self._quiet_moves.sort(key=lambda sm: sm.score, reverse=True)

    def _is_already_processed(self, move: Pos, stage_limit: PickStage) -> bool:
        if move == Pos.NONE: return True
        if stage_limit.value > PickStage.TT_MOVE.value and move == self.tt_move: return True
        if stage_limit.value > PickStage.WINNING_MOVES.value:
            if any(sm.pos == move for sm in self._winning_moves): return True
        if stage_limit.value > PickStage.DEFEND_OPPO_A5.value:
            if any(sm.pos == move for sm in self._defend_a5_moves): return True
        if stage_limit.value > PickStage.DEFEND_OPPO_B4.value:
             if any(sm.pos == move for sm in self._defend_b4_moves): return True
        if stage_limit.value > PickStage.KILLER_MOVES.value:
            if self.ss.is_killer(move): return True
        if stage_limit.value > PickStage.COUNTER_MOVE.value and self.ss_m1:
            opp_prev_move = self.ss_m1.current_move
            if opp_prev_move != Pos.NONE:
                cm_pair = self.counter_move_history.get_counter_moves(opp_prev_move)
                if move == cm_pair[0] or move == cm_pair[1]: return True
        return False

    def __iter__(self) -> MovePicker:
        return self

    def __next__(self) -> Pos:
        while True: 
            # print(f"DEBUG: Picker stage: {self.stage.name}, current_idx: {self.current_idx}")
            if self.stage == PickStage.TT_MOVE:
                self.stage = PickStage.GENERATE_WINNING_FORCING
                if self.tt_move != Pos.NONE and self.board.is_legal(self.tt_move):
                    return self.tt_move

            elif self.stage == PickStage.GENERATE_WINNING_FORCING:
                self._winning_moves = movegen_module.generate_all_moves(self.board, self.rule, GenType.WINNING)
                # print(f"Debug: GENERATE_WINNING_FORCING, _winning_moves={self._winning_moves}")
                
                if self.board.get_current_state_info().p4_count[self.opponent.value][Pattern4.A_FIVE.value] > 0:
                    self._defend_a5_moves = movegen_module.generate_all_moves(self.board, self.rule, GenType.DEFEND_FIVE)
                
                if self.board.get_current_state_info().p4_count[self.opponent.value][Pattern4.B_FLEX4.value] > 0:
                    self._defend_b4_moves = movegen_module.generate_all_moves(self.board, self.rule, GenType.DEFEND_FOUR)
                
                self.stage = PickStage.WINNING_MOVES
                self.current_idx = 0

            elif self.stage == PickStage.WINNING_MOVES:
                # print(f"Debug: Stage WINNING_MOVES, current_idx={self.current_idx}, len={len(self._winning_moves)}")
                if self.current_idx < len(self._winning_moves):
                    sm = self._winning_moves[self.current_idx]; self.current_idx += 1
                    if not self._is_already_processed(sm.pos, PickStage.WINNING_MOVES): return sm.pos
                else: self.stage = PickStage.DEFEND_OPPO_A5; self.current_idx = 0
            
            elif self.stage == PickStage.DEFEND_OPPO_A5:
                if self.current_idx < len(self._defend_a5_moves):
                    sm = self._defend_a5_moves[self.current_idx]; self.current_idx += 1
                    if not self._is_already_processed(sm.pos, PickStage.DEFEND_OPPO_A5): return sm.pos
                else: self.stage = PickStage.DEFEND_OPPO_B4; self.current_idx = 0

            elif self.stage == PickStage.DEFEND_OPPO_B4:
                if self.current_idx < len(self._defend_b4_moves):
                    sm = self._defend_b4_moves[self.current_idx]; self.current_idx += 1
                    if not self._is_already_processed(sm.pos, PickStage.DEFEND_OPPO_B4): return sm.pos
                else: self.stage = PickStage.KILLER_MOVES; self.current_idx = 0

            elif self.stage == PickStage.KILLER_MOVES:
                if self.current_idx < len(self.ss.killers):
                    killer_move = self.ss.killers[self.current_idx]; self.current_idx += 1
                    if killer_move != Pos.NONE and \
                       not self._is_already_processed(killer_move, PickStage.KILLER_MOVES) and \
                       self.board.is_legal(killer_move):
                        return killer_move
                else: self.stage = PickStage.COUNTER_MOVE

            elif self.stage == PickStage.COUNTER_MOVE:
                self.stage = PickStage.GENERATE_REMAINING 
                if self.ss_m1 and self.ss_m1.current_move != Pos.NONE:
                    opp_prev_move = self.ss_m1.current_move
                    cm_pair = self.counter_move_history.get_counter_moves(opp_prev_move)
                    for cm_idx, cm in enumerate(cm_pair): # Iterate primary then secondary
                        # Need to ensure countermove itself is not already processed by earlier stages
                        # The _is_already_processed check should use PickStage.COUNTER_MOVE
                        if cm != Pos.NONE and \
                           not self._is_already_processed(cm, PickStage.COUNTER_MOVE) and \
                           self.board.is_legal(cm):
                            # Yield one countermove per call to __next__ if this stage is active
                            # This requires restructuring how __next__ handles stages that can yield multiple items
                            # A simpler way: return the first valid one found.
                            # The current logic implies it will iterate through stages, and once COUNTER_MOVE is entered,
                            # it finds the first valid CM and returns, then next call to __next__ goes to GENERATE_REMAINING.
                            # This is fine.
                            return cm 
            
            elif self.stage == PickStage.GENERATE_REMAINING:
                all_other_moves_scored = movegen_module.generate_all_moves(self.board, self.rule, GenType.ALL)
                
                self._quiet_moves = []
                for sm in all_other_moves_scored:
                    if not self._is_already_processed(sm.pos, PickStage.GENERATE_REMAINING):
                        self._quiet_moves.append(sm)
                
                self._score_quiet_moves() 
                self.stage = PickStage.SORTED_QUIET_MOVES
                self.current_idx = 0

            elif self.stage == PickStage.SORTED_QUIET_MOVES:
                if self.current_idx < len(self._quiet_moves):
                    sm = self._quiet_moves[self.current_idx]
                    self.current_idx += 1
                    if self.skip_quiet_threshold is not None and sm.score < self.skip_quiet_threshold:
                        continue 
                    return sm.pos
                else:
                    self.stage = PickStage.DONE 
            
            elif self.stage == PickStage.DONE:
                raise StopIteration
            
            else: 
                raise Exception(f"Unknown MovePicker stage: {self.stage}")


if __name__ == '__main__':
    from unittest.mock import patch 
    from .board import Board # Assuming board.py also imports movegen correctly now for its own use if any
    from .config import DEFAULT_CANDIDATE_RANGE
    from .types import Rule, Color, Pattern4, Score # GenType from .types if movegen only has ScoredMove
    from .movegen import GenType # Already imported by move_picker module scope
    from .pos import Pos 
    from .search_stack import StackEntry
    from .history import MainHistory, CounterMoveHistory
    
    print("--- Move Picker Tests (Extended - Corrected Mocking) ---")
    
    main_hist_table = MainHistory()
    cm_hist_table = CounterMoveHistory()
    
    ss0_default = StackEntry(); ss0_default.ply = 0
    ss_minus_1_default = StackEntry(); ss_minus_1_default.ply = -1; ss_minus_1_default.current_move = Pos(0,0)

    def collect_moves(picker: MovePicker, max_moves_to_collect: int = 10) -> List[Pos]:
        moves = []
        try:
            for _ in range(max_moves_to_collect):
                moves.append(next(picker))
        except StopIteration:
            pass
        return moves

    # Test Case 1: TT Move available
    print("\nTest 1: TT Move available")
    board_t1 = Board(15, DEFAULT_CANDIDATE_RANGE); board_t1.new_game(Rule.FREESTYLE)
    tt_hit_move_t1 = Pos(7,7) 
    picker_t1 = MovePicker(board_t1, Rule.FREESTYLE, tt_hit_move_t1, main_hist_table, cm_hist_table, ss0_default, ss_minus_1_default)
    move_t1 = next(picker_t1)
    print(f"Picked: {move_t1}")
    assert move_t1 == tt_hit_move_t1

    # Test Case 2: No TT move, but winning move for self
    print("\nTest 2: No TT move, self has winning move")
    board_t2 = Board(15, DEFAULT_CANDIDATE_RANGE); board_t2.new_game(Rule.FREESTYLE)
    
    def mock_logic_for_test2_local(b, r, gt): 
        # print(f"DEBUG MOCK T2: CALLED with GenType: {gt}") # Keep this for verification
        if gt == GenType.WINNING:
            # print("DEBUG MOCK T2: RETURNING winning move for WINNING")
            return [ScoredMove(Pos(3,7))]
        return [] 

    # Patch 'generate_all_moves' in the 'src.movegen' module.
    with patch('src.movegen.generate_all_moves', side_effect=mock_logic_for_test2_local) as mocked_gen_moves_t2:
        picker2 = MovePicker(board_t2, Rule.FREESTYLE, Pos.NONE, main_hist_table, cm_hist_table, ss0_default, ss_minus_1_default)
        picked_moves_test2 = collect_moves(picker2, 3) # Collect up to 3 moves
        print(f"Picked (Test 2): {picked_moves_test2}")
        if not Pos(3,7) in picked_moves_test2:
            print(f"DEBUG T2: _winning_moves in picker2: {getattr(picker2, '_winning_moves', 'N/A')}")
            print(f"DEBUG T2: mocked_gen_moves_t2 call count: {mocked_gen_moves_t2.call_count}")
            if mocked_gen_moves_t2.call_args_list:
                print(f"DEBUG T2: mocked_gen_moves_t2 calls: {mocked_gen_moves_t2.call_args_list}")
        assert Pos(3,7) in picked_moves_test2
        assert picked_moves_test2[0] == Pos(3,7) # Should be the first non-TT move
    
    # Test Case 3: Opponent has A_FIVE, defense should be picked
    print("\nTest 3: Opponent A_FIVE threat")
    board_t3 = Board(15, DEFAULT_CANDIDATE_RANGE); board_t3.new_game(Rule.FREESTYLE)
    # Simulate opponent (White) having an A_FIVE threat.
    # The actual p4_count on board_t3.state_infos[0] needs to reflect this.
    board_t3.get_current_state_info().p4_count[Color.WHITE.value][Pattern4.A_FIVE.value] = 1 
    
    def mock_logic_for_test3_local(b, r, gt):
        # print(f"DEBUG MOCK T3: CALLED with GenType: {gt}")
        if gt == GenType.DEFEND_FIVE:
            # print("DEBUG MOCK T3: RETURNING A5 defense move")
            return [ScoredMove(Pos(8,8))]
        elif gt == GenType.WINNING: # Ensure self-winning doesn't take precedence if it's empty
             return [] 
        return [] 

    with patch('src.movegen.generate_all_moves', side_effect=mock_logic_for_test3_local) as mocked_gen_moves_t3:
        picker3 = MovePicker(board_t3, Rule.FREESTYLE, Pos.NONE, main_hist_table, cm_hist_table, ss0_default, ss_minus_1_default)
        picked_moves_test3 = collect_moves(picker3, 3) # Collect up to 3
        print(f"Picked (Test 3): {picked_moves_test3}")
        if not Pos(8,8) in picked_moves_test3:
            print(f"DEBUG T3: _defend_a5_moves in picker3: {getattr(picker3, '_defend_a5_moves', 'N/A')}")
            print(f"DEBUG T3: _winning_moves in picker3: {getattr(picker3, '_winning_moves', 'N/A')}")
            print(f"DEBUG T3: mocked_gen_moves_t3 call count: {mocked_gen_moves_t3.call_count}")
            if mocked_gen_moves_t3.call_args_list:
                 print(f"DEBUG T3: mocked_gen_moves_t3 calls: {mocked_gen_moves_t3.call_args_list}")
        assert Pos(8,8) in picked_moves_test3
        assert picked_moves_test3[0] == Pos(8,8) # Should be first after TT (empty) and self-winning (empty mock)
    board_t3.get_current_state_info().p4_count[Color.WHITE.value][Pattern4.A_FIVE.value] = 0 

    # Test Cases 4-10 (from previous extended tests)
    # ... (These should now work correctly with the import change in MovePicker) ...
    # Re-paste Test 6 for order check (TT > Win > Defend)
    print("\nTest 6: Order: TT > Winning > Defense")
    board_t6 = Board(15, DEFAULT_CANDIDATE_RANGE); board_t6.new_game(Rule.FREESTYLE)
    tt_move_t6 = Pos(1,1)
    winning_move_t6 = Pos(2,2)
    defense_move_t6 = Pos(3,3) 
    board_t6.get_current_state_info().p4_count[Color.WHITE.value][Pattern4.A_FIVE.value] = 1 
    def mock_logic_t6(b, r, gt):
        if gt == GenType.WINNING: return [ScoredMove(winning_move_t6)]
        if gt == GenType.DEFEND_FIVE: return [ScoredMove(defense_move_t6)]
        return []
    with patch('src.movegen.generate_all_moves', side_effect=mock_logic_t6):
        picker_t6 = MovePicker(board_t6, Rule.FREESTYLE, tt_move_t6, main_hist_table, cm_hist_table, ss0_default, ss_minus_1_default)
        picked_t6 = collect_moves(picker_t6, 3)
        print(f"Picked (Test 6): {picked_t6}")
        assert picked_t6 == [tt_move_t6, winning_move_t6, defense_move_t6]
    board_t6.get_current_state_info().p4_count[Color.WHITE.value][Pattern4.A_FIVE.value] = 0

    # Test Case 10: Quiet moves sorted by history
    print("\nTest 10: Quiet Moves Sorted by History")
    board_t10 = Board(15, DEFAULT_CANDIDATE_RANGE); board_t10.new_game(Rule.FREESTYLE)
    q1 = Pos(1,1); q2 = Pos(2,2); q3 = Pos(3,3) # Quiet moves

    main_hist_t10 = MainHistory()
    main_hist_t10.update_score(board_t10.side_to_move(), q1, 100) # q1 medium
    main_hist_t10.update_score(board_t10.side_to_move(), q2, 300) # q2 best
    main_hist_t10.update_score(board_t10.side_to_move(), q3, 50)  # q3 worst

    def mock_logic_t10(b, r, gt):
        if gt == GenType.ALL: # GENERATE_REMAINING stage
            # Return in non-history order
            return [ScoredMove(q1), ScoredMove(q3), ScoredMove(q2)] 
        return [] # No other types of moves

    with patch('src.movegen.generate_all_moves', side_effect=mock_logic_t10):
        picker_t10 = MovePicker(board_t10, Rule.FREESTYLE, Pos.NONE, main_hist_t10, cm_hist_table, ss0_default, ss_minus_1_default)
        picked_t10 = collect_moves(picker_t10, 3)
        print(f"Picked (Test 10): {picked_t10}")
        assert picked_t10 == [q2, q1, q3] # Expected order: q2 (300), q1 (100), q3 (50)

    print("\nMove Picker extended tests completed.")

    print("\nMove Picker extended tests (corrected mocking) completed.")