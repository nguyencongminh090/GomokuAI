"""
Alpha-Beta search algorithm implementation.
Based on Rapfi's search.cpp and ab/searcher.h.
"""
from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING, Any, cast
import time 
import math 

from .types import (Value, Rule, Depth, ActionType, GameRule, OpeningRule, Bound, 
                    mate_in, mated_in) # Ensure mate_in, mated_in are imported
from .pos import MAX_MOVES, Pos 
from .utils import now 
from .config import ENGINE_NAME, ENGINE_AUTHOR 
from . import config as engine_config 

from .search_abc import SearcherBase, SearchDataBase 
from .search_datastructures import (SearchOptions, RootMove, ABSearchData,
                                    root_move_value_comparator, BalanceMoveValueComparator,
                                    get_draw_value, balanced_value) # Added balanced_value
from .time_control import TimeControl, TimeInfo, StopConditions
from .hashtable import TranspositionTable # TTEntry not explicitly needed for type hints here
from .search_stack import SearchStackManager, StackEntry
from .move_picker import MovePicker, PickStage 
from .history import MainHistory, CounterMoveHistory
from .evaluation import evaluate as evaluate_board 
from .wincheck import quick_win_check
from .search_output import SearchPrinter

if TYPE_CHECKING:
    from .board import Board
    SimplifiedThreadPool = Any
    SimplifiedMainSearchThread = Any
    SimplifiedSearchThread = Any


class ABSearcher(SearcherBase):
    def __init__(self):
        super().__init__()
        self.tt: TranspositionTable = TranspositionTable()
        self.time_control: TimeControl = TimeControl()
        self.printer: SearchPrinter = SearchPrinter()
        self.search_data: Optional[ABSearchData] = None 
        self.reduction_tables: List[List[Depth]] = [
            [0.0] * (MAX_MOVES + 1) for _ in range(engine_config.RULE_NB)
        ]
        self._init_reduction_lut() 
        self.previous_search_best_value: int = Value.VALUE_NONE.value
        self.previous_time_reduction_factor: float = 1.0

    def _init_reduction_lut(self):
        for r_idx in range(engine_config.RULE_NB):
            for i in range(MAX_MOVES + 1):
                self.reduction_tables[r_idx][i] = 0.0 

    def make_search_data(self, search_thread_instance: SimplifiedSearchThread) -> ABSearchData:
        current_options = getattr(search_thread_instance, 'search_options', None)
        if current_options is None and hasattr(search_thread_instance, 'options'):
            current_options = search_thread_instance.options()
        if not isinstance(current_options, SearchOptions):
            current_options = SearchOptions()
        ab_data = ABSearchData(options=current_options)
        # Ensure search_stack_manager is initialized here if not in ABSearchData constructor by default
        if not hasattr(ab_data, 'search_stack_manager'):
             ab_data.search_stack_manager = SearchStackManager(
                 max_depth=current_options.max_depth + 10, # Or from engine_config.MAX_SEARCH_DEPTH
                 initial_static_eval=Value.VALUE_ZERO.value # Placeholder, will be set in ID loop
             )
        return ab_data

    def set_memory_limit(self, memory_size_kb: int) -> None:
        self.tt.resize(memory_size_kb)

    def get_memory_limit(self) -> int:
        return self.tt.hash_size_kb()

    def clear(self, thread_pool_instance: Optional[SimplifiedThreadPool] = None, clear_all_memory: bool = False) -> None:
        self.previous_search_best_value = Value.VALUE_NONE.value
        self.previous_time_reduction_factor = 1.0
        if clear_all_memory:
            self.tt.clear()

    def search_main(self, main_search_context: SimplifiedMainSearchThread) -> None:
        current_board: Board = main_search_context.board_to_search
        options: SearchOptions = main_search_context.search_options
        
        self.search_data = self.make_search_data(main_search_context)
        if not self.search_data: return

        self.search_data.root_moves = self._generate_root_moves(current_board, options)

        if not self.search_data.root_moves:
            if hasattr(main_search_context, 'best_move_found'): main_search_context.best_move_found = Pos.NONE 
            if hasattr(main_search_context, 'result_action_type'): main_search_context.result_action_type = ActionType.MOVE
            return

        self.time_control.init(
            turn_time_ms=options.turn_time, match_time_ms=options.match_time,
            time_left_ms=options.time_left,
            time_info=TimeInfo(ply=current_board.ply(), moves_left_in_game=options.moves_to_go or 30),
            inc_time_ms=options.inc_time, moves_to_go=options.moves_to_go
        )
        self.tt.inc_generation() 

        if hasattr(main_search_context, 'in_ponder'): main_search_context.in_ponder = getattr(options, 'pondering', False)
        if hasattr(main_search_context, 'board_size_for_output'): main_search_context.board_size_for_output = current_board.board_size

        self.printer.print_search_starts(main_search_context, self.time_control)
        self._iterative_deepening_loop(current_board, options, main_search_context)

        final_best_root_move_obj: Optional[RootMove] = None
        if self.search_data.root_moves:
            if options.balance_mode != SearchOptions.BalanceMode.BALANCE_NONE:
                comp = BalanceMoveValueComparator(options.balance_bias)
                self.search_data.root_moves.sort(key=lambda rm: balanced_value(rm.value, comp.bias), reverse=True)
            else:
                self.search_data.root_moves.sort(key=lambda rm: (rm.value, rm.previous_value), reverse=True) 
            final_best_root_move_obj = self.search_data.root_moves[0]
            if hasattr(main_search_context, 'best_move_found'): main_search_context.best_move_found = final_best_root_move_obj.move
            if hasattr(main_search_context, 'result_action_type'): main_search_context.result_action_type = self.search_data.result_action 
        else:
            if hasattr(main_search_context, 'best_move_found'): main_search_context.best_move_found = Pos.NONE
        
        if hasattr(main_search_context, 'best_root_move_obj'): main_search_context.best_root_move_obj = final_best_root_move_obj
        if hasattr(main_search_context, 'root_moves'): main_search_context.root_moves = self.search_data.root_moves

        if self.search_data.root_moves: 
            self.previous_search_best_value = self.search_data.root_moves[0].value
        
        self.printer.print_search_ends(main_search_context, self.time_control,
                                       self.search_data.completed_search_depth,
                                       main_search_context, 
                                       self.search_data.nodes_this_search # Use nodes from search_data
                                       )

    def search(self, search_thread_context: SimplifiedSearchThread) -> None:
        # Called by iterative deepening for each root move's search process
        # This is where aspiration search for a specific root move would happen
        # For now, _iterative_deepening_loop directly calls _recursive_search for children
        pass

    def check_timeup_condition(self) -> bool:
        return self.time_control.is_time_up(check_optimal=False)

    def _generate_root_moves(self, board: Board, options: SearchOptions) -> List[RootMove]:
        if not self.search_data: # Should exist if called from search_main
            # print("Warning: _generate_root_moves called without search_data", file=sys.stderr)
            return [RootMove(board.center_pos())] if board.ply() == 0 else []

        temp_stack_entry = StackEntry() 
        picker = MovePicker(board, options.game_rule.rule, Pos.NONE,
                            self.search_data.main_history, 
                            self.search_data.counter_move_history, 
                            temp_stack_entry)
        initial_moves: List[RootMove] = []
        try:
            while True:
                move = next(picker)
                if move not in options.block_moves:
                    initial_moves.append(RootMove(move))
        except StopIteration: pass
        if not initial_moves and board.ply() == 0: return [RootMove(board.center_pos())] 
        return initial_moves

    def _iterative_deepening_loop(self, board: Board, options: SearchOptions,
                                  main_search_context: SimplifiedMainSearchThread):
        if self.search_data is None: return 
        
        self.search_data.search_stack_manager = SearchStackManager(
            max_depth=options.max_depth + 10, 
            initial_static_eval=evaluate_board(board, options.game_rule.rule)
        )
        self.search_data.nodes_this_search = 0 

        current_board_eval = self.search_data.search_stack_manager.root_stack_entry().static_eval
        
        for rm in self.search_data.root_moves:
            rm.value = rm.previous_value if rm.previous_value != Value.VALUE_NONE.value else current_board_eval

        max_iter_depth = min(options.max_depth, engine_config.MAX_SEARCH_DEPTH)
        start_iter_depth = min(options.start_depth, max_iter_depth)
        
        for depth_iter in range(start_iter_depth, max_iter_depth + 1):
            if self.search_data is None: break 
            self.search_data.current_search_depth = depth_iter
            self.search_data.best_move_changes_this_iter = 0 
            # self.search_data.sel_depth_max = 0 # Reset per iteration overall max

            for rm in self.search_data.root_moves:
                rm.previous_value = rm.value
                rm.previous_pv = list(rm.pv) 

            if hasattr(main_search_context, 'previous_ply_best_move'):
                if self.search_data.root_moves:
                    main_search_context.previous_ply_best_move = self.search_data.root_moves[0].move
            
            aspiration_center = current_board_eval
            if depth_iter > start_iter_depth and self.search_data.root_moves:
                aspiration_center = self.search_data.root_moves[0].value 

            root_aspiration_alpha = Value.VALUE_MATED_IN_MAX_PLY.value
            root_aspiration_beta = Value.VALUE_MATE_IN_MAX_PLY.value
            
            if engine_config.ASPIRATION_WINDOW and depth_iter >= engine_config.ASPIRATION_MIN_DEPTH:
                aspiration_delta_val = 50 + depth_iter * 5 
                root_aspiration_alpha = max(Value.VALUE_MATED_IN_MAX_PLY.value, aspiration_center - aspiration_delta_val)
                root_aspiration_beta = min(Value.VALUE_MATE_IN_MAX_PLY.value, aspiration_center + aspiration_delta_val)

            for pv_idx in range(self.search_data.multi_pv_count):
                if pv_idx >= len(self.search_data.root_moves): break 
                if self.search_data is None: break 

                self.search_data.current_pv_index = pv_idx
                current_root_move_obj = self.search_data.root_moves[pv_idx]
                
                # Reset sel_depth_max for this specific PV line's search
                self.search_data.sel_depth_max = 0 

                board.make_move(options.game_rule.rule, current_root_move_obj.move)
                
                ss_for_root_child = self.search_data.search_stack_manager.get_entry(board.ply())
                ss_for_root_child.reset() 
                ss_for_root_child.static_eval = evaluate_board(board, options.game_rule.rule)

                child_alpha = -root_aspiration_beta 
                child_beta = -root_aspiration_alpha  
                
                found_value_for_child = self._recursive_search(
                    board, options, child_alpha, child_beta, 
                    depth_iter - 1, ss_for_root_child, 
                    is_pv_node=(pv_idx == 0), is_root_call=False 
                )
                score_for_root_move = -found_value_for_child 
                
                board.undo_move(options.game_rule.rule)

                current_root_move_obj.value = score_for_root_move
                current_root_move_obj.pv = [current_root_move_obj.move] + [p for p in ss_for_root_child.pv if p != Pos.NONE]
                current_root_move_obj.sel_depth = self.search_data.sel_depth_max 

                # Sort only processed root moves for printing the current best for this iteration
                # This ensures root_moves[0] is always the best found *so far in this iteration* for print_pv_completes
                temp_root_moves_processed_this_iter = self.search_data.root_moves[:pv_idx + 1]
                if options.balance_mode != SearchOptions.BalanceMode.BALANCE_NONE:
                    comp = BalanceMoveValueComparator(options.balance_bias)
                    temp_sorted_moves = sorted(temp_root_moves_processed_this_iter, key=lambda rm: balanced_value(rm.value, comp.bias), reverse=True)
                else:
                    temp_sorted_moves = sorted(temp_root_moves_processed_this_iter, key=lambda rm: (rm.value, rm.previous_value), reverse=True)
                for i in range(len(temp_sorted_moves)): self.search_data.root_moves[i] = temp_sorted_moves[i]
                
                self.printer.print_pv_completes(main_search_context, self.time_control,
                                                depth_iter, pv_idx, self.search_data.multi_pv_count,
                                                self.search_data.root_moves[0], # Print current iteration's best
                                                self.search_data.nodes_this_search
                                                )
                if self.check_timeup_condition(): break 
            
            if self.search_data and self.search_data.root_moves: # Final sort of all root moves for this depth
                if options.balance_mode != SearchOptions.BalanceMode.BALANCE_NONE:
                    comp = BalanceMoveValueComparator(options.balance_bias)
                    self.search_data.root_moves.sort(key=lambda rm_sort: balanced_value(rm_sort.value, comp.bias), reverse=True)
                else:
                    self.search_data.root_moves.sort(key=lambda rm_sort: (rm_sort.value, rm_sort.previous_value), reverse=True)

            if self.check_timeup_condition(): 
                if hasattr(main_search_context, 'mark_pondering_available'): main_search_context.mark_pondering_available()
                break

            self.search_data.completed_search_depth = depth_iter
            if self.search_data.root_moves: 
                self.printer.print_depth_completes(main_search_context, self.time_control,
                                                  depth_iter, self.search_data.root_moves[0])
            
            stop_conditions = StopConditions(
                current_search_depth=depth_iter,
                last_best_move_change_depth=depth_iter - self.search_data.best_move_changes_this_iter, 
                current_best_value=self.search_data.root_moves[0].value if self.search_data.root_moves else Value.VALUE_NONE.value,
                previous_search_best_value=self.previous_search_best_value,
                previous_time_reduction_factor=self.previous_time_reduction_factor,
                avg_best_move_changes_this_iter=float(self.search_data.best_move_changes_this_iter) 
            )
            should_stop, new_reduction_factor = self.time_control.check_stop(stop_conditions, self.previous_time_reduction_factor)
            self.previous_time_reduction_factor = new_reduction_factor 

            if should_stop:
                if hasattr(main_search_context, 'mark_pondering_available'): main_search_context.mark_pondering_available()
                break 
            
            if self.search_data.root_moves:
                self.previous_search_best_value = self.search_data.root_moves[0].value


    def _recursive_search(self, board: Board, options: SearchOptions,
                          alpha: int, beta: int, depth: int, 
                          ss: StackEntry, 
                          is_pv_node: bool, is_root_call: bool = False # is_root_call True if this node is a root move's child
                         ) -> int:
        """ Core recursive Alpha-Beta (Negamax style) search function. """
        if self.search_data is None: return Value.VALUE_NONE.value # Should have search_data
        
        self.search_data.nodes_this_search += 1 

        # Update selective depth for PV display if this is a PV path
        if is_pv_node and self.search_data.sel_depth_max < ss.ply:
            self.search_data.sel_depth_max = ss.ply
        
        # --- 1. Termination checks (Draws, Max Depth in Search Tree) ---
        if board.ply() >= options.max_moves_in_game : 
             return get_draw_value(board.ply() - board.pass_count[0]-board.pass_count[1], board.side_to_move(), options, ss.ply)
        # Practical search depth limit (ss.ply is 0-indexed from actual board root)
        if ss.ply >= engine_config.MAX_SEARCH_DEPTH + 5 : 
             return evaluate_board(board, options.game_rule.rule, alpha, beta)

        # --- 2. Check for immediate win/loss (not for root's children direct call, but for deeper nodes) ---
        # In Rapfi, this is done after TT lookup for non-PV nodes.
        # And not done if depth <= 0 (quiescence handles it).
        # `is_root_call` here means this node is a direct child of one of the ID loop's root moves.
        # For these, quickWinCheck might be too aggressive or redundant with root move generation logic.
        # Let's defer this to after TT lookup or if not is_root_call and depth > 0.

        # --- 3. Mate Distance Pruning ---
        # Values from perspective of current player at this node
        effective_alpha = max(alpha, mated_in(ss.ply)) # We must at least achieve better than being mated
        effective_beta = min(beta, mate_in(ss.ply + 1))   # No point searching if we can mate sooner
        if effective_alpha >= effective_beta:
            return effective_alpha # This means current window is already in a mate/mated state

        # --- 4. Transposition Table Lookup ---
        current_zobrist_key = board.zobrist_key() 
        tt_hit, tt_value, tt_eval_from_tt, tt_is_pv_entry, tt_bound, tt_move_from_tt, tt_depth_from_tt = \
           self.tt.probe(current_zobrist_key, ss.ply)

        if tt_hit and tt_depth_from_tt >= depth: # Use entry if depth is sufficient
            # Apply TT cutoffs based on bound type
            if tt_bound == Bound.BOUND_EXACT:
                if tt_move_from_tt != Pos.NONE : ss.update_pv(tt_move_from_tt, StackEntry()) # Minimal PV
                return tt_value
            if tt_bound == Bound.BOUND_LOWER and tt_value >= effective_beta:
                # TODO: Update counter/killer for tt_move_from_tt if it causes cutoff
                if tt_move_from_tt != Pos.NONE: ss.add_killer(tt_move_from_tt) # Basic killer update
                return tt_value 
            if tt_bound == Bound.BOUND_UPPER and tt_value <= effective_alpha:
                return tt_value 
        
        # Initialize static_eval for current node (ss)
        # If TT hit provided an eval, use it. Otherwise, compute if not already set (e.g. by parent)
        if tt_hit and tt_eval_from_tt != Value.VALUE_NONE.value:
            ss.static_eval = tt_eval_from_tt
        elif ss.static_eval == Value.VALUE_ZERO.value and ss.ply > 0 : # If not root and not set
             ss.static_eval = evaluate_board(board, options.game_rule.rule, effective_alpha, effective_beta)
        # Note: ss.static_eval for ply 0 (root of search) is set by _iterative_deepening_loop.
        # For children (ss_for_root_child, ply=1), it's set before calling _recursive_search.

        # --- QuickWinCheck again, now that static_eval might be set (for quiescence) ---
        # This is usually done if depth <= 0, or as part of quiescence.
        # Rapfi does it *before* depth <= 0 if not root call. Let's stick to that.
        if not is_root_call and depth > 0: # Check only for non-root, non-quiescence nodes
            immediate_mate_score = quick_win_check(board, options.game_rule.rule, ss.ply, effective_beta)
            if immediate_mate_score != Value.VALUE_ZERO.value:
                return immediate_mate_score

        # --- 5. Base case for recursion: Depth <= 0 (Quiescence/VCF or Static Eval) ---
        if depth <= 0:
            # TODO: Implement Quiescence Search (VCF search in Rapfi) which would use a MovePicker
            # with GenType.VCF or similar, and search only forcing moves.
            # For now, return the static evaluation (potentially already improved by quickWinCheck if it returned 0).
            # q_score = quick_win_check(board, options.game_rule.rule, ss.ply, effective_beta) # Already did for non-root
            # if q_score != Value.VALUE_ZERO.value: return q_score
            return ss.static_eval 

        # --- (Skipping Null Move, Razoring, Futility for now: these are advanced pruning) ---

        # --- 6. Initialize for move iteration ---
        original_alpha_for_tt_store = effective_alpha 
        best_value_at_node = -Value.VALUE_INFINITE.value # Scores are relative to current player
        best_move_at_node = Pos.NONE # TT move could be a candidate if not used for cutoff
        
        # --- 7. Move Generation & Loop ---
        if self.search_data is None: return Value.VALUE_NONE.value 
        
        # Determine if this node is considered PV for child searches.
        # It's a PV node if the parent was PV AND this is the first move tried from parent OR first move raising alpha.
        # `is_pv_node` parameter already tells us if parent considers us PV.
        
        move_picker = MovePicker(board, options.game_rule.rule, tt_move_from_tt, 
                                 self.search_data.main_history,
                                 self.search_data.counter_move_history,
                                 ss, # Current ply's stack entry
                                 self.search_data.search_stack_manager.get_entry(ss.ply - 1) if ss.ply > 0 else None, # ss-1
                                 self.search_data.search_stack_manager.get_entry(ss.ply - 2) if ss.ply > 1 else None  # ss-2
                                )
        
        if ss.ply == 1: # Only print for direct children of root
            print(f"DEBUG RS (ply={ss.ply}, depth={depth}): MovePicker created. Board state for picker (player {board.side_to_move().name}):")
            print(board.to_string()) 
            
            temp_move_list_for_debug = []
            # Create a temporary picker to see what it yields without consuming the main one
            temp_picker_for_debug = MovePicker(board, options.game_rule.rule, tt_move_from_tt, 
                                 self.search_data.main_history, self.search_data.counter_move_history,
                                 ss, 
                                 self.search_data.search_stack_manager.get_entry(ss.ply - 1) if ss.ply > 0 else None,
                                 self.search_data.search_stack_manager.get_entry(ss.ply - 2) if ss.ply > 1 else None)
            try:
                for i in range(25): # Try to get more moves for debugging
                    m = next(temp_picker_for_debug)
                    if m == Pos.NONE: break 
                    temp_move_list_for_debug.append(m)
            except StopIteration:
                pass 
            print(f"DEBUG RS (ply={ss.ply}, depth={depth}): Initial moves from DEBUG picker: {temp_move_list_for_debug}")
        # ***** END DEBUG PRINT *****
        
        moves_searched_this_node = 0
        for current_move in move_picker: 
            if current_move == Pos.NONE: continue # Should not be yielded by picker

            # Make move
            board.make_move(options.game_rule.rule, current_move)
            moves_searched_this_node +=1
            
            # Prepare child stack entry
            child_ss = self.search_data.search_stack_manager.get_entry(ss.ply + 1)
            child_ss.reset() 
            child_ss.ply = board.ply() # Ply after move
            # Static eval for child node can be computed here or by child call
            child_ss.static_eval = evaluate_board(board, options.game_rule.rule) 
            child_ss.current_move = current_move # For debugging/history context

            # Determine if child is a PV node. True if current is PV and this is the first move improving alpha.
            # A simpler rule: first move of a PV node search is also PV.
            child_is_pv = is_pv_node and (moves_searched_this_node == 1) 

            # TODO: Reductions (LMR) would adjust new_depth here
            new_depth = depth - 1
            
            # --- PVS Search (Principal Variation Search) Logic ---
            # If this is a PV node and not the first move, try a null-window search first.
            # if is_pv_node and moves_searched_this_node > 1:
            #    score = -self._recursive_search(board, options, -effective_alpha - 1, -effective_alpha, new_depth, child_ss, False, False)
            #    if score > effective_alpha and score < effective_beta: # Re-search if it failed high for null window
            #        score = -self._recursive_search(board, options, -effective_beta, -effective_alpha, new_depth, child_ss, True, False)
            # else: # Full window search for first move or non-PV nodes

            score = -self._recursive_search(board, options, -effective_beta, -effective_alpha, new_depth, 
                                            child_ss, child_is_pv, False) 
            
            board.undo_move(options.game_rule.rule)

            if score > best_value_at_node:
                best_value_at_node = score
                best_move_at_node = current_move
                
                ss.update_pv(best_move_at_node, child_ss) # Update PV for current node (ss)

                if score > effective_alpha: 
                    effective_alpha = score # Raise alpha
                    if effective_alpha >= effective_beta: # Beta cutoff (fail-high)
                        self.tt.store(current_zobrist_key, effective_beta, ss.static_eval, is_pv_node,
                                      Bound.BOUND_LOWER, best_move_at_node, depth, ss.ply)
                        if best_move_at_node != Pos.NONE: ss.add_killer(best_move_at_node)
                        # TODO: Update history heuristics for best_move_at_node
                        return effective_beta # Return beta as the score for fail-high
        
        # --- 8. After all moves searched ---
        if moves_searched_this_node == 0: # No legal moves from this position
            return mated_in(ss.ply) # Current player is mated at this ply

        # Store TT entry based on final alpha
        bound_type_to_store = Bound.BOUND_EXACT if effective_alpha > original_alpha_for_tt_store else Bound.BOUND_UPPER
        self.tt.store(current_zobrist_key, best_value_at_node, ss.static_eval, is_pv_node,
                      bound_type_to_store, best_move_at_node, depth, ss.ply)
                      
        return best_value_at_node # This is original_alpha if no improvement, or new alpha.


if __name__ == '__main__':
    from .board import Board 
    from .search_datastructures import SearchOptions, RootMove, ActionType 
    from .types import Rule, Value
    from . import config as engine_config 

    print("--- Search Main Call Test (ABSearcher Full Structure) ---")
    
    searcher = ABSearcher() 

    class MockMainSearchContext:
        def __init__(self, board_obj: Board, search_opts: SearchOptions):
            self.board_to_search: Board = board_obj
            self.search_options: SearchOptions = search_opts
            self.best_move_found: Pos = Pos.NONE
            self.result_action_type: ActionType = ActionType.MOVE
            self.total_nodes_accumulated: int = 0 
            self.in_ponder: bool = False
            self.board_size_for_output = board_obj.board_size 
            self.root_moves: List[RootMove] = [] 
            self.best_root_move_obj: Optional[RootMove] = None 
        def mark_pondering_available(self): pass

    test_board = Board(15, engine_config.DEFAULT_CANDIDATE_RANGE) 
    test_board.new_game(Rule.FREESTYLE)
    
    test_options = SearchOptions()
    test_options.max_depth = 3 
    test_options.start_depth = 1
    test_options.time_limit_is_active = False 
    test_options.multi_pv = 1

    main_ctx = MockMainSearchContext(test_board, test_options)
    
    searcher.clear(clear_all_memory=True) 
    
    print("LOG: === CALLING search_main ONCE ===")
    searcher.search_main(main_ctx) 
    print("LOG: === RETURNED FROM search_main ONCE ===")

    print(f"FINAL: Search main completed. Best move found: {main_ctx.best_move_found}")
    final_completed_depth = -1
    if searcher.search_data: 
        final_completed_depth = searcher.search_data.completed_search_depth
    print(f"FINAL: Final search data completed depth: {final_completed_depth}")
    
    if searcher.search_data and searcher.search_data.root_moves: 
        assert main_ctx.best_move_found != Pos.NONE
        assert final_completed_depth == test_options.max_depth
    else:
        print("FINAL: No root moves were generated or retained by the search.")

    print("Search.py test finished.")