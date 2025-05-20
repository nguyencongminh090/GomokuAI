"""
Data structures used by the Search algorithms,
such as RootMove, SearchOptions, Balance2Move, and ABSearchData.
Based on concepts from Rapfi's searchcommon.h, searcher.h, and searchthread.h.
"""
import math
import sys 
from typing import List, Optional, Any, NamedTuple, Callable 
from enum import Enum, IntFlag 

from .types import (Value, ActionType, Rule, Color, GameRule, OpeningRule, Score,
                    Pattern, Pattern4, SIDE_NB, PATTERN_NB, RULE_NB) 
from .pos import Pos, MAX_MOVES 
from .history import MainHistory, CounterMoveHistory 
from .utils import PRNG 
from .search_abc import SearchDataBase # <--- IMPORT THE ABC

# ... (RootMove, Balance2Move, SearchOptions, comparators, get_draw_value remain the same) ...
def balanced_value(value: int, bias: int) -> int:
    return -abs(value - bias) + bias
class Balance2Move(NamedTuple):
    move1: Pos; move2: Pos
class RootMove:
    def __init__(self, move_or_pair: Pos | Balance2Move):
        self.value: int = Value.VALUE_NONE.value
        self.previous_value: int = Value.VALUE_NONE.value
        self.sel_depth: int = 0
        self.win_rate: float = float('nan')
        self.draw_rate: float = float('nan')
        self.policy_prior: float = float('nan')
        self.utility_stdev: float = float('nan')
        self.lcb_value: float = float('nan') 
        self.selection_value: float = float('nan') 
        self.nodes_searched: int = 0 
        self.pv: List[Pos]
        if isinstance(move_or_pair, Pos): self.pv = [move_or_pair]
        elif isinstance(move_or_pair, Balance2Move): self.pv = [move_or_pair.move1, move_or_pair.move2]
        else: raise TypeError("RootMove must be initialized with Pos or Balance2Move")
        self.previous_pv: List[Pos] = list(self.pv)
    @property
    def move(self) -> Pos: return self.pv[0] if self.pv else Pos.NONE
    def __repr__(self) -> str:
        pv_str = " ".join(f"({p.x},{p.y})" if p!=Pos.NONE else "None" for p in self.pv if p != Pos.NONE)
        move_repr = repr(self.move) if self.pv else "Pos.NONE"
        return f"RootMove({move_repr}, val={self.value}, sel_d={self.sel_depth}, PV=[{pv_str}])"
    def __eq__(self, other: Any) -> bool:
        if not self.pv: return False 
        if isinstance(other, RootMove):
            if not other.pv: return False
            if len(self.pv) >= 2 and len(other.pv) >= 2 and self.pv[0] != self.pv[1] and other.pv[0] != other.pv[1]:
                 return self.pv[0] == other.pv[0] and self.pv[1] == other.pv[1]
            return self.pv[0] == other.pv[0]
        if isinstance(other, Pos): return self.pv[0] == other
        if isinstance(other, Balance2Move): return len(self.pv) >= 2 and self.pv[0] == other.move1 and self.pv[1] == other.move2
        return False
def root_move_value_comparator(rm_a: RootMove, rm_b: RootMove) -> bool:
    if rm_a.value != rm_b.value: return rm_a.value > rm_b.value
    return rm_a.previous_value > rm_b.previous_value
class BalanceMoveValueComparator:
    def __init__(self, bias: int = 0): self.bias: int = bias
    def compare(self, rm_a: RootMove, rm_b: RootMove) -> bool:
        bal_a = balanced_value(rm_a.value, self.bias); bal_b = balanced_value(rm_b.value, self.bias)
        if bal_a != bal_b: return bal_a > bal_b
        bal_prev_a = balanced_value(rm_a.previous_value, self.bias); bal_prev_b = balanced_value(rm_b.previous_value, self.bias)
        return bal_prev_a > bal_prev_b
class SearchOptions:
    class InfoMode(IntFlag): INFO_NONE=0; INFO_REALTIME=1<<0; INFO_DETAIL=1<<1; INFO_REALTIME_AND_DETAIL=INFO_REALTIME|INFO_DETAIL
    class BalanceMode(Enum): BALANCE_NONE=0; BALANCE_ONE=1; BALANCE_TWO=2
    class DrawResult(Enum): RES_DRAW=0; RES_BLACK_WIN=1; RES_WHITE_WIN=2
    def __init__(self):
        self.game_rule: GameRule = GameRule(Rule.FREESTYLE, OpeningRule.FREEOPEN)
        self.swapable: bool = False; self.disable_opening_query: bool = False; self.pondering: bool = False 
        self.time_limit_is_active: bool = False; self.info_mode: SearchOptions.InfoMode = SearchOptions.InfoMode.INFO_NONE
        self.turn_time: int = 0; self.match_time: int = 0; self.time_left: int = 0 
        self.inc_time: int = 0; self.moves_to_go: int = 0; self.max_nodes: int = 0  
        self.max_depth: int = 99; self.start_depth: int = 2; self.multi_pv: int = 1
        self.strength_level: int = 100; self.balance_mode: SearchOptions.BalanceMode = SearchOptions.BalanceMode.BALANCE_NONE
        self.balance_bias: int = 0; self.max_moves_in_game: int = sys.maxsize 
        self.draw_result_at_max_moves: SearchOptions.DrawResult = SearchOptions.DrawResult.RES_DRAW
        self.block_moves: List[Pos] = [] 
    def is_analysis_mode(self) -> bool: return not self.time_limit_is_active and self.max_nodes == 0
    def set_time_control(self, turn_time: int, match_time: int, time_left: int, inc_time: int, moves_to_go: int):
        self.turn_time = turn_time; self.match_time = match_time; self.time_left = time_left
        self.inc_time = inc_time; self.moves_to_go = moves_to_go
        if self.turn_time <= 0 and self.match_time <= 0: self.time_limit_is_active = False
        elif self.turn_time > 0 and self.match_time <= 0: self.time_limit_is_active = True
        elif self.turn_time <= 0 and self.match_time > 0: 
            self.time_limit_is_active = True
            if self.time_left <= 0 : self.time_left = self.match_time 
        else: 
            self.time_limit_is_active = True
            if self.time_left <= 0 : self.time_left = self.match_time
def get_draw_value(board_non_pass_move_count: int, side_to_move: Color, options: SearchOptions, current_search_ply: int) -> int:
    from .types import mate_in, mated_in # Local import for direct execution if needed
    plies_until_max_ply = max(options.max_moves_in_game - board_non_pass_move_count, 0)
    effective_mate_ply = current_search_ply + plies_until_max_ply
    if options.draw_result_at_max_moves == SearchOptions.DrawResult.RES_DRAW: return Value.VALUE_DRAW.value
    elif options.draw_result_at_max_moves == SearchOptions.DrawResult.RES_BLACK_WIN:
        return mate_in(effective_mate_ply) if side_to_move == Color.BLACK else mated_in(effective_mate_ply)
    elif options.draw_result_at_max_moves == SearchOptions.DrawResult.RES_WHITE_WIN:
        return mate_in(effective_mate_ply) if side_to_move == Color.WHITE else mated_in(effective_mate_ply)
    return Value.VALUE_DRAW.value

class ABSearchData(SearchDataBase): # Inherit from the ABC
    def __init__(self, options: Optional[SearchOptions] = None):
        super().__init__() # Call ABC constructor
        self.multi_pv_count: int = options.multi_pv if options else 1
        self.current_pv_index: int = 0     
        self.current_search_depth: int = 0 
        self.iter_completed_depth: int = 0 
        self.best_move_changes_this_iter: int = 0
        self.is_singular_root: bool = False 

        self.main_history: MainHistory = MainHistory()
        self.counter_move_history: CounterMoveHistory = CounterMoveHistory()

        self.root_moves: List[RootMove] = [] 
        self.best_move_root: Pos = Pos.NONE  
        self.result_action: ActionType = ActionType.MOVE 

        self.sel_depth_max: int = 0 
        self.root_alpha: int = Value.VALUE_NONE.value
        self.root_beta: int = Value.VALUE_NONE.value 
        self.root_delta: int = 0 

        self.previous_pv_line: List[Pos] = []
        self.previous_best_move_at_root: Pos = Pos.NONE

    def clear_data(self, search_thread_instance: Any): # Implements abstract method
        current_options = search_thread_instance.options()
        
        self.multi_pv_count = current_options.multi_pv
        self.current_pv_index = 0
        self.current_search_depth = 0  
        self.iter_completed_depth = 0 
        self.best_move_changes_this_iter = 0 
        self.is_singular_root = False 
        
        self.main_history.init(Score(0)) 
        self.counter_move_history.init((Pos.NONE, Pos.NONE))
        
        self.root_moves = []
        self.best_move_root = Pos.NONE
        self.result_action = ActionType.MOVE
        self.sel_depth_max = 0 
        self.root_alpha = Value.VALUE_NONE.value
        self.root_beta = Value.VALUE_NONE.value
        self.root_delta = 0
        self.previous_pv_line = []
        self.previous_best_move_at_root = Pos.NONE


if __name__ == '__main__':
    print("--- Search Datastructures Tests (Extended) ---")
    # ... (tests remain the same, should pass now) ...
    rm1 = RootMove(Pos(7,7)); rm1.value = 100; rm1.sel_depth = 10
    rm1.pv = [Pos(7,7), Pos(8,8)]; 
    print(f"RootMove 1: {rm1}")

    b2m = Balance2Move(Pos(1,1), Pos(1,2))
    rm_b2 = RootMove(b2m)
    print(f"RootMove B2: {rm_b2}")

    opts = SearchOptions()
    opts.multi_pv = 2
    opts.game_rule = GameRule(Rule.RENJU, OpeningRule.SWAP1)
    print(f"SearchOptions: multi_pv={opts.multi_pv}, rule={opts.game_rule.rule.name}")

    search_data = ABSearchData(options=opts) 
    search_data.root_moves.append(rm1)
    search_data.iter_completed_depth = 4 
    print(f"Initial ABSearchData: multi_pv={search_data.multi_pv_count}")
    assert search_data.multi_pv_count == 2
    
    class MockSearchThread: 
        def __init__(self, options_obj: SearchOptions):
            self._current_options = options_obj
            self.num_nodes = 0 
        def options(self) -> SearchOptions: 
            return self._current_options

    new_mock_options = SearchOptions() 
    new_mock_options.multi_pv = 1 
    mock_thread = MockSearchThread(new_mock_options)

    search_data.clear_data(mock_thread) 
    print(f"ABSearchData after clear_data: multi_pv={search_data.multi_pv_count}")
    assert search_data.multi_pv_count == 1 
    assert not search_data.root_moves

    from .types import mate_in, mated_in 
    draw_opts = SearchOptions()
    draw_opts.max_moves_in_game = 60
    draw_opts.draw_result_at_max_moves = SearchOptions.DrawResult.RES_BLACK_WIN
    val_draw = get_draw_value(board_non_pass_move_count=58, side_to_move=Color.BLACK, 
                              options=draw_opts, current_search_ply=10)
    print(f"Draw value (Black wins, BTM, 2 moves to limit): {val_draw} (expected mate_in(12)={mate_in(12)})")
    assert val_draw == mate_in(12)

    print("Search datastructures tests (extended) completed.")