"""
Alpha-Beta search algorithm implementation.
Based on Rapfi's search.cpp and ab/searcher.h.
"""
from __future__ import annotations # For type hinting
from typing import List, Optional, TYPE_CHECKING, Any
import time # For basic timing if needed outside TimeControl

from .types import Value, Rule, Depth, ActionType, GameRule, OpeningRule, Bound
from .pos import MAX_MOVES, Pos # For reduction table sizing, etc.
from .utils import now # For timing
from .config import ENGINE_NAME, ENGINE_AUTHOR # Using config for these
from . import config as engine_config # For various search parameters

from .search_abc import SearcherBase, SearchDataBase # Abstract base classes
from .search_datastructures import (SearchOptions, RootMove, ABSearchData,
                                    root_move_value_comparator, BalanceMoveValueComparator,
                                    get_draw_value)
from .time_control import TimeControl, TimeInfo, StopConditions
from .hashtable import TranspositionTable
from .search_stack import SearchStackManager, StackEntry
from .move_picker import MovePicker, PickStage # For type hints or direct use
from .history import MainHistory, CounterMoveHistory
from .evaluation import evaluate as evaluate_board # Renamed to avoid clash if a local 'evaluate' is defined
from .wincheck import quick_win_check
from .search_output import SearchPrinter
# from .opening import Opening # If we integrate opening book later
# from .skill import SkillMovePicker # If we integrate skill levels later

if TYPE_CHECKING:
    from .board import Board
    # ThreadPool and MainSearchThread types are complex for single-threaded Python start
    # Using Any for now for those parameters in abstract methods if we were to implement them fully.
    # For our single-threaded approach, search_main and search will be called differently.
    SimplifiedThreadPool = Any
    SimplifiedMainSearchThread = Any
    SimplifiedSearchThread = Any


class ABSearcher(SearcherBase):
    """
    Implements the Alpha-Beta search algorithm with iterative deepening.
    """
    def __init__(self):
        super().__init__()
        self.tt: TranspositionTable = TranspositionTable()
        self.time_control: TimeControl = TimeControl()
        self.printer: SearchPrinter = SearchPrinter() # For outputting search info

        # Search-specific data, managed per top-level "go" command
        self.search_data: Optional[ABSearchData] = None # Created by make_search_data for a search instance

        # Reduction tables (LUTs for Late Move Reduction, etc.)
        # reductions[rule_val][depth_or_move_count] -> reduction_amount (Depth)
        # Rapfi: std::array<Depth, MAX_MOVES + 1> reductions[RULE_NB];
        # Max depth or move count for indexing. MAX_MOVES is board_area + 1.
        # Let's use a simple list of lists for now.
        self.reduction_tables: List[List[Depth]] = [
            [0.0] * (MAX_MOVES + 1) for _ in range(engine_config.RULE_NB)
        ]
        self._init_reduction_lut() # Initialize with default/Rapfi values

        # Per-game state (like in Rapfi's ABSearcher)
        self.previous_search_best_value: int = Value.VALUE_NONE.value
        self.previous_time_reduction_factor: float = 1.0


    def _init_reduction_lut(self):
        """Initializes reduction tables. Placeholder for now."""
        # TODO: Populate with actual reduction values based on Rapfi's
        # Search::AB::initReductionLUT or similar logic.
        # For example, reductions based on depth and move number.
        # For now, they are all 0.0.
        for r_idx in range(engine_config.RULE_NB):
            for i in range(MAX_MOVES + 1):
                # Example: Simple reduction that increases with depth and move_count
                # This is NOT Rapfi's actual logic, just a placeholder.
                # self.reduction_tables[r_idx][i] = math.log(1 + i * 0.1) / 2.0
                self.reduction_tables[r_idx][i] = 0.0 # Start with no reductions

    # --- Implementation of SearcherBase abstract methods ---

    def make_search_data(self, search_thread_instance: SimplifiedSearchThread) -> ABSearchData:
        """
        Creates an instance of ABSearchData for a search.
        In single-threaded, search_thread_instance might be 'self' or a simple context.
        """
        # The 'options' for ABSearchData should come from the current search command.
        # Assuming search_thread_instance (or a SearchContext passed to think()) has options.
        current_options = getattr(search_thread_instance, 'options_obj', SearchOptions())
        return ABSearchData(options=current_options)

    def set_memory_limit(self, memory_size_kb: int) -> None:
        """Sets the memory size limit for the TT."""
        self.tt.resize(memory_size_kb)

    def get_memory_limit(self) -> int:
        """Gets the current memory size limit of the TT in KiB."""
        return self.tt.hash_size_kb()

    def clear(self, thread_pool_instance: Optional[SimplifiedThreadPool] = None, clear_all_memory: bool = False) -> None:
        """
        Clears searcher states between different games.
        `thread_pool_instance` is less relevant for single-threaded.
        """
        self.previous_search_best_value = Value.VALUE_NONE.value
        self.previous_time_reduction_factor = 1.0
        # Re-initialize reduction LUTs if they depend on game-specific settings (unlikely for static ones)
        # self._init_reduction_lut()

        if clear_all_memory:
            self.tt.clear()
        # History tables are part of ABSearchData, cleared per new search call.

    def search_main(self, main_search_context: SimplifiedMainSearchThread) -> None:
        """
        Main entry point for starting a search (orchestrates iterative deepening).
        `main_search_context` would hold the board, search options, and store results.
        In Rapfi, this is called on the MainSearchThread.
        """
        # 1. Setup: Get board, options from context.
        #    Initialize self.search_data.
        #    Handle opening book, immediate wins/forced moves if any.
        #    Setup TimeControl.
        #    Start iterative deepening.
        #    Store best move in context.
        
        # This will be the main iterative deepening loop orchestrator.
        # For now, a placeholder.
        
        # Assume main_search_context has:
        # - board_to_search: Board
        # - search_options: SearchOptions
        # - And attributes to store results: best_move_found, result_action_type

        current_board: Board = main_search_context.board_to_search
        options: SearchOptions = main_search_context.search_options
        
        self.search_data = self.make_search_data(main_search_context) # Create/reset search data
        self.search_data.root_moves = self._generate_root_moves(current_board, options)

        if not self.search_data.root_moves:
            # print("No moves to search at root.", file=sys.stderr)
            main_search_context.best_move_found = Pos.NONE # Or PASS if allowed
            main_search_context.result_action_type = ActionType.MOVE
            return

        # TODO: Opening book probe
        # TODO: Check for immediate mate at root / single legal move

        self.time_control.init(
            turn_time_ms=options.turn_time,
            match_time_ms=options.match_time,
            time_left_ms=options.time_left,
            time_info=TimeInfo(ply=current_board.ply(), moves_left_in_game=options.moves_to_go or 30), # Estimate moves
            inc_time_ms=options.inc_time,
            moves_to_go=options.moves_to_go
        )
        self.tt.inc_generation() # New search, new generation for TT entries

        self.printer.print_search_starts(main_search_context, self.time_control)

        self._iterative_deepening_loop(current_board, options, main_search_context)

        # After loop, select final best move from self.search_data.root_moves
        if self.search_data.root_moves:
            # Sort root_moves by score if not already sorted by last iteration
            self.search_data.root_moves.sort(key=lambda rm: rm.value, reverse=True) # Higher score first
            main_search_context.best_move_found = self.search_data.root_moves[0].move
            main_search_context.result_action_type = self.search_data.result_action # May be set by opening/swap logic
        else:
            main_search_context.best_move_found = Pos.NONE
        
        # TODO: Pondering logic if applicable

        # Save per-game state for next search
        if self.search_data.root_moves:
            self.previous_search_best_value = self.search_data.root_moves[0].value
        # self.previous_time_reduction_factor is updated by time_control.check_stop

        # Output final search stats using printer
        # This needs a "best_thread_context", which in single thread is just main_search_context
        # Also need final completed depth.
        self.printer.print_search_ends(main_search_context, self.time_control,
                                       self.search_data.completed_search_depth,
                                       main_search_context, # As best_thread_context
                                       getattr(main_search_context, 'total_nodes_accumulated', 0))


    def search(self, search_thread_context: SimplifiedSearchThread) -> None:
        """
        The main search function called for a specific depth iteration.
        In Rapfi, this is called by each thread. In our single-thread model,
        this will be called by the iterative deepening loop.
        `search_thread_context` provides the board, options, and where to store results for this iteration.
        """
        # This will contain the call to the root aspiration search / root alpha-beta.
        # It's the entry point for a single iteration of iterative deepening.
        # For now, placeholder. The logic is mostly in _iterative_deepening_loop and below.
        # In Rapfi, ABSearcher::search calls the aspiration search loop.
        pass # Actual search per iteration happens in _iterative_deepening_loop for now


    def check_timeup_condition(self) -> bool:
        """Checks if the search time limit has been reached."""
        return self.time_control.is_time_up(check_optimal=False) # Check against maximum_time

    # --- Helper methods for search ---
    def _generate_root_moves(self, board: Board, options: SearchOptions) -> List[RootMove]:
        """Generates and possibly filters initial root moves."""
        # Basic generation - a real engine might use a dedicated root movegen type
        # or specialized move picker for the root.
        # For now, use a general move picker and convert to RootMove objects.
        
        # In Rapfi, this is complex, involving MovePicker::ExtraArgs<MovePicker::ROOT>
        # and potential filtering (symmetry, block_moves).
        
        # Simplified: get all candidate moves, then filter
        temp_stack_entry = StackEntry() # Dummy stack for root move generation
        # No TT move for initial generation, no specific history tables for root move gen picker.
        # This is a simplification. Rapfi's root move picker might use some context.
        picker = MovePicker(board, options.game_rule.rule, Pos.NONE,
                            MainHistory(), CounterMoveHistory(), # Fresh/dummy histories
                            temp_stack_entry)
        
        initial_moves: List[RootMove] = []
        try:
            while True:
                move = next(picker)
                if move not in options.block_moves:
                    initial_moves.append(RootMove(move))
        except StopIteration:
            pass
        
        # TODO: Symmetry filtering (complex) - Opening::filterSymmetryMoves
        # For now, return all valid generated moves.
        if not initial_moves and board.ply() == 0: # First move, no legal moves? (Should not happen)
            return [RootMove(board.center_pos())] # Fallback to center
        return initial_moves

    def _iterative_deepening_loop(self, board: Board, options: SearchOptions,
                                  main_search_context: SimplifiedMainSearchThread):
        """Manages the iterative deepening process."""
        if self.search_data is None: 
            return 

        current_board_eval = evaluate_board(board, options.game_rule.rule)
        
        for rm in self.search_data.root_moves:
            rm.value = current_board_eval 

        max_iter_depth = min(options.max_depth, engine_config.MAX_SEARCH_DEPTH)
        start_iter_depth = min(options.start_depth, max_iter_depth)

        for depth_iter in range(start_iter_depth, max_iter_depth + 1):
            self.search_data.current_search_depth = depth_iter
            
            aspiration_center_score = current_board_eval 
            if depth_iter > start_iter_depth and self.search_data.root_moves:
                aspiration_center_score = self.search_data.root_moves[0].value

            for rm in self.search_data.root_moves:
                rm.previous_value = rm.value
                rm.previous_pv = list(rm.pv) 

            if hasattr(main_search_context, 'previous_ply_best_move'): # Check attribute existence
                if self.search_data.root_moves:
                    main_search_context.previous_ply_best_move = self.search_data.root_moves[0].move


            for pv_idx in range(self.search_data.multi_pv_count):
                if pv_idx >= len(self.search_data.root_moves): break 

                self.search_data.current_pv_index = pv_idx
                
                alpha = Value.VALUE_MATED_IN_MAX_PLY.value 
                beta = Value.VALUE_MATE_IN_MAX_PLY.value   
                
                # --- Mocking search result for now ---
                mock_score = current_board_eval - pv_idx * 10 
                if pv_idx == 0 and depth_iter % 2 == 0 : mock_score += 20 
                self.search_data.root_moves[pv_idx].value = mock_score
                self.search_data.root_moves[pv_idx].sel_depth = depth_iter 
                if self.search_data.root_moves[pv_idx].pv: 
                    mock_next_move = Pos(self.search_data.root_moves[pv_idx].pv[0].x + 1, self.search_data.root_moves[pv_idx].pv[0].y)
                    if mock_next_move.is_on_board(board.board_size, board.board_size): # Use board from parameters
                         self.search_data.root_moves[pv_idx].pv = [self.search_data.root_moves[pv_idx].pv[0], mock_next_move]
                    else:
                         self.search_data.root_moves[pv_idx].pv = [self.search_data.root_moves[pv_idx].pv[0]]
                # --- End Mocking search result ---

                if options.balance_mode != SearchOptions.BalanceMode.BALANCE_NONE:
                    # Sorting with BalanceMoveValueComparator needs a proper key function
                    # For now, using a simplified sort based on the regular value for testing flow
                    self.search_data.root_moves.sort(key=lambda rm_sort: rm_sort.value, reverse=True)
                else:
                    self.search_data.root_moves.sort(key=lambda rm_sort: (rm_sort.value, rm_sort.previous_value), reverse=True)

                self.printer.print_pv_completes(main_search_context, self.time_control,
                                                depth_iter, pv_idx, self.search_data.multi_pv_count,
                                                self.search_data.root_moves[pv_idx], 
                                                getattr(main_search_context, 'total_nodes_accumulated', 0) 
                                                )

                if self.check_timeup_condition(): break 
            
            if not self.check_timeup_condition():
                self.search_data.completed_search_depth = depth_iter
                if self.search_data.root_moves: # Ensure root_moves is not empty
                    self.printer.print_depth_completes(main_search_context, self.time_control,
                                                      depth_iter, self.search_data.root_moves[0])
            
            stop_conditions = StopConditions(
                current_search_depth=depth_iter,
                # ***** CORRECTED ATTRIBUTE NAME HERE *****
                last_best_move_change_depth=depth_iter - self.search_data.best_move_changes_this_iter, 
                current_best_value=self.search_data.root_moves[0].value if self.search_data.root_moves else Value.VALUE_NONE.value,
                previous_search_best_value=self.previous_search_best_value,
                previous_time_reduction_factor=self.previous_time_reduction_factor,
                avg_best_move_changes_this_iter=float(self.search_data.best_move_changes_this_iter) 
            )
            should_stop, new_reduction_factor = self.time_control.check_stop(stop_conditions, self.previous_time_reduction_factor)
            self.previous_time_reduction_factor = new_reduction_factor 

            if should_stop or self.check_timeup_condition():
                if hasattr(main_search_context, 'mark_pondering_available'):
                    main_search_context.mark_pondering_available()
                break  # Exit iterative deepening loop
            
            # Prepare for next iteration (e.g. reset best_move_changes_current_iter)
            # self.search_data.clear_for_next_iteration() # Resets some per-iteration stats


    def _recursive_search(self, board: Board, options: SearchOptions,
                          alpha: int, beta: int, depth: int, # Remaining depth
                          current_ply_stack: StackEntry, # ss for current ply
                          is_pv_node: bool, is_root_node: bool = False,
                          # Other flags like cut_node, null_move_allowed etc.
                          ) -> int:
        """
        The core recursive Alpha-Beta (Negamax style) search function.
        Returns the evaluation of the position.
        """
        # This is the main workhorse. Placeholder for now.
        # It will involve:
        # 1. Increment node count (self.search_data.total_nodes_searched or thread-local)
        # 2. Check for draw by repetition/ply limit.
        # 3. Mate distance pruning (adjust alpha/beta based on mate_in(ply), mated_in(ply)).
        # 4. TT Lookup:
        #    - If hit and entry is good enough (depth, bound), return TT score.
        #    - Use TT move for move ordering.
        # 5. Base case: depth <= 0 or terminal node:
        #    - Call quiescence search (e.g., VCF search) or static evaluation.
        #    - `quickWinCheck` for immediate tactical wins/losses.
        #    - `evaluate_board` for static eval.
        # 6. Null Move Pruning (if conditions met):
        #    - Make a null move, recurse with reduced depth.
        #    - If result causes cutoff, return beta.
        # 7. Initialize best_score = -VALUE_INFINITE, best_move = Pos.NONE.
        # 8. Create MovePicker for current node.
        # 9. Loop through moves from MovePicker:
        #    - Pruning before making move (e.g., futility pruning if applicable).
        #    - Make move on board.
        #    - Late Move Reduction (LMR): if move is late and not critical, search with reduced depth.
        #    - Recursive call: score = -_recursive_search(..., -beta, -alpha, new_depth, ss+1, ...).
        #    - Undo move.
        #    - Update alpha, best_score, best_move.
        #    - If score >= beta (fail-high):
        #        - Store TT entry (LOWER_BOUND).
        #        - Update killer moves, history heuristics.
        #        - Return beta.
        # 10. After loop, store TT entry (EXACT_BOUND if alpha changed, UPPER_BOUND otherwise).
        # 11. Return best_score (which is alpha if it was raised).
        
        # Placeholder implementation:
        if depth <= 0:
            # In a real search, this would be quiescence search / VCF.
            # For now, just static evaluation.
            q_score = quick_win_check(board, options.game_rule.rule, current_ply_stack.ply, beta)
            if q_score != Value.VALUE_ZERO.value:
                return q_score
            return evaluate_board(board, options.game_rule.rule, alpha, beta)

        # TODO: TT Probe, Null move, etc.
        
        best_value = -Value.VALUE_INFINITE.value # or alpha
        num_moves_searched = 0

        # Placeholder: get moves (a real picker would be used)
        # This needs the full context for the picker.
        # For this placeholder, let's assume we have a way to get some moves.
        # move_list = self._generate_some_moves_for_node(board, options) # Placeholder

        # For now, just return a dummy value
        return evaluate_board(board, options.game_rule.rule, alpha, beta)


if __name__ == '__main__':
    from .board import Board # For full Board object

    print("--- ABSearcher Basic Initialization Test ---")
    searcher = ABSearcher()
    assert searcher.tt is not None
    assert searcher.time_control is not None
    print(f"Default TT size: {searcher.get_memory_limit()} KB")
    searcher.set_memory_limit(2048) # 2MB
    print(f"Set TT size: {searcher.get_memory_limit()} KB")
    # Note: Python TT size estimate is rough, so it might not be exactly 2048.
    # Actual number of buckets depends on floor_power_of_two.
    # For 2048KB and ~256B/bucket -> 8192 buckets. floor_power_of_two(8192) = 8192.
    # So expected size should be (8192 * 256) / 1024 = 2048 KB.
    assert searcher.get_memory_limit() >= 1024 # Check if it's reasonably large

    print("\n--- Mocking a Search Main Call (Simplified) ---")
    # Mock a MainSearchThread context
    class MockMainSearchContext:
        def __init__(self, board_obj: Board, search_opts: SearchOptions):
            self.board_to_search: Board = board_obj
            self.search_options: SearchOptions = search_opts
            self.best_move_found: Pos = Pos.NONE
            self.result_action_type: ActionType = ActionType.MOVE
            self.total_nodes_accumulated: int = 0 
            self.in_ponder: bool = False
            self.board_size_for_output = board_obj.board_size 
            self.root_moves: List[RootMove] = [] # Add this to store the final root moves list
            self.best_root_move_obj: Optional[RootMove] = None # Explicitly for printer

        def options(self) -> SearchOptions: 
            return self.search_options
        
        def mark_pondering_available(self): pass


    test_board = Board(15)
    test_board.new_game(Rule.FREESTYLE)
    
    test_options = SearchOptions()
    test_options.max_depth = 3 # Shallow search for test
    test_options.start_depth = 1
    test_options.time_limit_is_active = False # No time limits for this basic test
    test_options.multi_pv = 1

    main_ctx = MockMainSearchContext(test_board, test_options)
    
    searcher.clear(clear_all_memory=True) # Clear searcher state
    searcher.search_main(main_ctx) # Call the main search orchestrator

    print(f"Search main completed. Best move found: {main_ctx.best_move_found}")
    print(f"Final search data completed depth: {searcher.search_data.completed_search_depth if searcher.search_data else -1}")
    
    # We expect it to run up to depth 3.
    # The current _iterative_deepening_loop has mock scoring, so a best move will be picked.
    assert main_ctx.best_move_found != Pos.NONE
    if searcher.search_data:
         assert searcher.search_data.completed_search_depth == test_options.max_depth

    print("ABSearcher basic structure and search_main call test completed.")