"""
Alpha-Beta search algorithm implementation.
Based on Rapfi's search.cpp and ab/searcher.h.
"""
from __future__ import annotations
from typing import List, Optional, TYPE_CHECKING, Any, cast
import time 
import math 
from . import movegen as movegen_module 
from .movegen import GenType, ScoredMove

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
                          is_pv_node: bool, is_root_call: bool = False 
                         ) -> int:
        # ... (node count, sel_depth, termination checks, mate distance pruning, TT lookup as before) ...
        if self.search_data is None: return Value.VALUE_NONE.value
        self.search_data.nodes_this_search += 1
        if is_pv_node and self.search_data.sel_depth_max < ss.ply: self.search_data.sel_depth_max = ss.ply
        if board.ply() >= options.max_moves_in_game: return get_draw_value(board.ply() - board.pass_count[0]-board.pass_count[1], board.side_to_move(), options, ss.ply)
        if ss.ply >= engine_config.MAX_SEARCH_DEPTH + 5: return evaluate_board(board, options.game_rule.rule, alpha, beta)
        
        # Check for immediate win/loss (moved before MDP to match Rapfi search.cpp structure more closely for non-root)
        if not is_root_call:
            immediate_mate_score = quick_win_check(board, options.game_rule.rule, ss.ply, beta)
            if immediate_mate_score != Value.VALUE_ZERO.value:
                return immediate_mate_score
        
        effective_alpha = max(alpha, mated_in(ss.ply))
        effective_beta = min(beta, mate_in(ss.ply + 1))
        if effective_alpha >= effective_beta:
            return effective_alpha

        current_zobrist_key = board.zobrist_key() 
        tt_hit, tt_value, tt_eval_from_tt, tt_is_pv_entry, tt_bound, tt_move_from_tt, tt_depth_from_tt = \
           self.tt.probe(current_zobrist_key, ss.ply)

        if tt_hit and tt_depth_from_tt >= depth:
            if tt_bound == Bound.BOUND_EXACT:
                if tt_move_from_tt != Pos.NONE : ss.update_pv(tt_move_from_tt, StackEntry()) 
                return tt_value
            if tt_bound == Bound.BOUND_LOWER and tt_value >= effective_beta:
                if tt_move_from_tt != Pos.NONE: ss.add_killer(tt_move_from_tt)
                return tt_value 
            if tt_bound == Bound.BOUND_UPPER and tt_value <= effective_alpha:
                return tt_value 
        
        if tt_hit and tt_eval_from_tt != Value.VALUE_NONE.value:
            ss.static_eval = tt_eval_from_tt
        # Else, if static_eval is not set (e.g. for root children, it's set by _iterative_deepening_loop)
        # For deeper nodes, if not from TT, it should be computed if it's zero.
        elif ss.static_eval == Value.VALUE_ZERO.value and ss.ply > 0 : 
             ss.static_eval = evaluate_board(board, options.game_rule.rule, effective_alpha, effective_beta)
        
        # --- Base case for recursion: Depth <= 0 -> CALL QUIESCENCE SEARCH ---
        if depth <= 0:
            return self._quiescence_search(board, options, effective_alpha, effective_beta, ss)

        # ... (Null Move Pruning placeholder) ...
        # ... (Razoring, Futility Pruning placeholders) ...

        original_alpha_for_tt_store = effective_alpha 
        best_value_at_node = -Value.VALUE_INFINITE.value 
        best_move_at_node = Pos.NONE
        
        if self.search_data is None: return Value.VALUE_NONE.value # Should be caught earlier
        
        move_picker = MovePicker(board, options.game_rule.rule, tt_move_from_tt, 
                                 self.search_data.main_history, self.search_data.counter_move_history,
                                 ss, 
                                 self.search_data.search_stack_manager.get_entry(ss.ply - 1) if ss.ply > 0 else None,
                                 self.search_data.search_stack_manager.get_entry(ss.ply - 2) if ss.ply > 1 else None
                                )
        
        moves_searched_this_node = 0
        for current_move in move_picker: 
            if current_move == Pos.NONE: continue 

            # Use Board.MoveType.NORMAL for main search moves to update all states including internal eval
            board.make_move(options.game_rule.rule, current_move, Board.MoveType.NORMAL)
            moves_searched_this_node +=1
            
            child_ss = self.search_data.search_stack_manager.get_entry(ss.ply + 1)
            child_ss.reset() 
            child_ss.ply = board.ply() 
            # The static_eval for child_ss is now set by board.make_move if it updates st.value_black
            # and if child_ss.static_eval takes it from -board.current_state_info().value_black
            # Or, it's computed before the recursive call if not updated by make_move's side effects.
            # For now, let's assume make_move(NORMAL) makes board state (and its eval) ready.
            # We can pass the eval of the child position directly.
            child_ss.static_eval = evaluate_board(board, options.game_rule.rule) 

            child_ss.current_move = current_move 
            
            child_is_pv = is_pv_node and (moves_searched_this_node == 1) 
            new_depth = depth - 1
            
            # TODO: LMR, PVS search window logic

            score = -self._recursive_search(board, options, -effective_beta, -effective_alpha, new_depth, 
                                            child_ss, child_is_pv, False) 
            
            board.undo_move(options.game_rule.rule, Board.MoveType.NORMAL)

            if score > best_value_at_node:
                best_value_at_node = score
                best_move_at_node = current_move
                ss.update_pv(best_move_at_node, child_ss) 

                if score > effective_alpha: 
                    effective_alpha = score
                    if effective_alpha >= effective_beta: 
                        self.tt.store(current_zobrist_key, effective_beta, ss.static_eval, is_pv_node,
                                      Bound.BOUND_LOWER, best_move_at_node, depth, ss.ply)
                        if best_move_at_node != Pos.NONE: ss.add_killer(best_move_at_node)
                        # TODO: History updates
                        return effective_beta 
        
        if moves_searched_this_node == 0: 
            return mated_in(ss.ply) 

        bound_type_to_store = Bound.BOUND_EXACT if effective_alpha > original_alpha_for_tt_store else Bound.BOUND_UPPER
        self.tt.store(current_zobrist_key, best_value_at_node, ss.static_eval, is_pv_node,
                      bound_type_to_store, best_move_at_node, depth, ss.ply)
                      
        return best_value_at_node
    
    def _quiescence_search(self, board: Board, options: SearchOptions,
                           alpha: int, beta: int, ss: StackEntry) -> int:
        if self.search_data is None: return Value.VALUE_NONE.value

        self.search_data.nodes_this_search += 1

        # Update selective depth if this qsearch node is on a PV path
        # (is_pv_node for qsearch is typically inherited or assumed False unless specific tracking)
        # For simplicity, let's assume qsearch nodes don't extend the main PV's sel_depth count
        # or if they do, the is_pv_node status would need to be passed.
        # Rapfi's vcfsearch updates selDepth. If ss.ttPv is true:
        if ss.tt_pv and self.search_data.sel_depth_max < ss.ply:
             self.search_data.sel_depth_max = ss.ply


        # Check for immediate terminal state (mate) by quickWinCheck
        # Note: ply for mate_in/mated_in is current stack ply
        immediate_mate_score = quick_win_check(board, options.game_rule.rule, ss.ply, beta)
        if immediate_mate_score != Value.VALUE_ZERO.value:
            return immediate_mate_score

        # Stand-pat evaluation (evaluate current position without making further moves)
        # ss.static_eval should have been computed by the caller (_recursive_search before depth <= 0)
        # or we re-evaluate here if it's not considered reliable enough for q-search entry.
        # For q-search, we usually start with the static eval of the current node.
        stand_pat_score = ss.static_eval 
        # If static_eval wasn't set reliably by caller:
        # stand_pat_score = evaluate_board(board, options.game_rule.rule, alpha, beta)


        if stand_pat_score >= beta:
            return beta # Fail-high based on static eval (stand-pat)
        if stand_pat_score > alpha:
            alpha = stand_pat_score

        # Define GenType for quiescence search moves
        # Only interested in tactical/forcing moves: Win, VCF, VCT, Defend critical threats
        qsearch_gen_type = (GenType.WINNING | GenType.VCF | GenType.VCT |
                            GenType.DEFEND_FIVE | GenType.DEFEND_FOUR | GenType.DEFEND_B4F3)
        
        # MovePicker for quiescence moves
        # No TT move passed to qsearch picker usually, no complex history.
        # Stack entries for prev moves might be relevant for countermoves in qsearch if allowed.
        q_move_picker = MovePicker(
            board, options.game_rule.rule, Pos.NONE, # No TT move for qsearch picker itself
            self.search_data.main_history, # Can pass main history
            self.search_data.counter_move_history, # Can pass counter history
            ss, # Current stack entry
            self.search_data.search_stack_manager.get_entry(ss.ply - 1) if ss.ply > 0 else None,
            self.search_data.search_stack_manager.get_entry(ss.ply - 2) if ss.ply > 1 else None
        )
        # Override its generation type for the "quiet moves" stage it will eventually reach
        # This is a bit of a hack. A dedicated QsearchMovePicker would be cleaner.
        # For now, we rely on MovePicker's early stages (WINNING, DEFEND_*) to pick tactical moves
        # and hope that if it reaches GENERATE_REMAINING, that `generate_all_moves` called with
        # `qsearch_gen_type` will only yield tactical ones.
        # A better way: Have MovePicker accept a primary GenType for its tactical stages.
        # Or, have a separate move generator for qsearch.

        # Let's use a direct call to movegen for qsearch for simplicity first.
        # This bypasses the multi-stage MovePicker logic for qsearch.
        q_moves_list: List[ScoredMove] = movegen_module.generate_all_moves(
            board, options.game_rule.rule, qsearch_gen_type
        )
        # TODO: Score these q_moves for better ordering (e.g., self A5 > self B4 > defend oppo A5)
        # For now, iterate in generated order.

        best_value_in_qsearch = stand_pat_score # Start with stand-pat
        # No best_move_at_node needed for qsearch TT store, as qsearch usually doesn't store full PVs

        for scored_move in q_moves_list:
            current_q_move = scored_move.pos
            if current_q_move == Pos.NONE: continue

            # Make move
            board.make_move(options.game_rule.rule, current_q_move, Board.MoveType.NO_EVAL) # Use NO_EVAL for qsearch
            
            child_ss_q = self.search_data.search_stack_manager.get_entry(ss.ply + 1)
            child_ss_q.reset()
            child_ss_q.ply = board.ply()
            child_ss_q.static_eval = evaluate_board(board, options.game_rule.rule) # Eval for child state

            # Recursive call to quiescence search. Depth is not decremented in q-search.
            # Alpha/beta are negated.
            score = -self._quiescence_search(board, options, -beta, -alpha, child_ss_q)
            
            board.undo_move(options.game_rule.rule, Board.MoveType.NO_EVAL)

            if score > best_value_in_qsearch:
                best_value_in_qsearch = score
                if score > alpha:
                    alpha = score
                    if alpha >= beta: # Beta cutoff
                        # Q-search can also store to TT, but often with depth 0 or a special q-depth
                        # self.tt.store(..., bound=Bound.BOUND_LOWER, depth=0, ...)
                        return beta # Fail high

        # No TT store for UPPER bound from qsearch in this simplified version
        return best_value_in_qsearch # Or alpha, if alpha was improved


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