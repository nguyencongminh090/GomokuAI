"""
Handles printing search-related information and statistics.
Based on Rapfi's searchoutput.h and searchoutput.cpp.
"""
from __future__ import annotations # For type hinting
import time 
from typing import List, Optional, TYPE_CHECKING
# Ensure Any is imported if used for placeholders like SearchContext
from typing import Any # Added Any

from .types import Value, ActionType
from .pos import MAX_BOARD_SIZE, Pos 
from .utils import time_text, nodes_text, speed_text, now # <--- IMPORT 'now' HERE
from . import config as engine_config 

if TYPE_CHECKING:
    from .search_datastructures import RootMove, SearchOptions # SearchOptions is needed by _show_realtime
    from .time_control import TimeControl
    SearchContext = Any 
    BestSearchContext = Any 

# Need to define SearchOptions.InfoMode and SearchOptions.BalanceMode if used directly
# Or ensure they are accessible via search_context.options().info_mode
# For standalone test, it's better to define them or import them fully
from .search_datastructures import SearchOptions # For direct use of its enums in _show_realtime etc.

class SearchPrinter:
    """
    Controls message outputs during searching.
    """

    # Constants from Rapfi for controlling realtime output
    REALTIME_MIN_DEPTH: int = 8
    REALTIME_MIN_ELAPSED: int = 200 # milliseconds

    def __init__(self):
        pass # No specific state needed for this basic version yet

    def _output_coord(self, pos: Pos, board_size: int) -> str:
        """Formats a Pos object into a string like 'h8'."""
        if pos == Pos.NONE: return "none"
        if pos == Pos.PASS: return "pass"
        # Assuming 0-indexed x, y and standard chess-like coordinates (a1 bottom-left)
        # Piskvork protocol usually is x from 0-N, y from 0-N.
        # Rapfi's outputCoordXConvert/YConvert depend on IOCoordMode.
        # For simplicity, use 0-indexed x,y directly.
        # Or, (char('a'+x), y+1) for typical gomoku notation.
        # Let's go with (x,y) for now for simplicity.
        return f"{pos.x},{pos.y}"

    def _moves_to_text(self, moves: List[Pos], board_size: int) -> str:
        """Converts a list of Pos (PV) to a string."""
        if not moves or moves[0] == Pos.NONE:
            return "(no pv)"
        return " ".join(self._output_coord(m, board_size) for m in moves if m != Pos.NONE)

    # --- Internal Helper Methods (from Rapfi) ---
    def _show_realtime(self, search_context: SearchContext, time_control: TimeControl, root_depth: int) -> bool:
        # In Rapfi, SearchContext is MainSearchThread (th)
        # SearchOptions would be th.options()
        # For now, assume search_context has .options attribute
        if not hasattr(search_context, 'options'): return False
        opts = search_context.options()
        return bool(opts.info_mode & SearchOptions.InfoMode.INFO_REALTIME) and \
               root_depth >= self.REALTIME_MIN_DEPTH

    def _show_realtime_in_loop(self, search_context: SearchContext, time_control: TimeControl, root_depth: int) -> bool:
        if not hasattr(search_context, 'options'): return False
        opts = search_context.options()
        return self._show_realtime(search_context, time_control, root_depth) and \
               time_control.elapsed() >= self.REALTIME_MIN_ELAPSED and \
               opts.balance_mode != SearchOptions.BalanceMode.BALANCE_TWO # Rapfi condition

    def _show_info(self, search_context: SearchContext) -> bool:
        if not hasattr(search_context, 'options'): return False
        opts = search_context.options()
        return bool(opts.info_mode & SearchOptions.InfoMode.INFO_DETAIL)

    # --- Public Printing Methods ---

    def print_search_starts(self, search_context: SearchContext, time_control: TimeControl):
        if engine_config.MESSAGE_MODE == engine_config.MsgMode.NORMAL:
            # Assume search_context has options and in_ponder attributes
            if hasattr(search_context, 'options') and hasattr(search_context, 'in_ponder'):
                if search_context.options().time_limit_is_active and not search_context.in_ponder:
                    print(f"INFO: OptiTime {time_text(time_control.optimal_time)} | "
                          f"MaxTime {time_text(time_control.maximum_time)}")

    def print_entering_move(self, search_context: SearchContext, time_control: TimeControl,
                            pv_idx: int, root_depth: int, move: Pos):
        # Simplified, Rapfi has REALTIME output here
        if self._show_realtime_in_loop(search_context, time_control, root_depth):
            if hasattr(search_context, 'board_size_for_output') and \
               not getattr(search_context, 'in_ponder', False) and \
               not engine_config.ASPIRATION_WINDOW and pv_idx == 0: # Rapfi conditions
                # print(f"REALTIME POS: {self._output_coord(move, search_context.board_size_for_output)}")
                pass # Keep it quiet for now unless full REALTIME support is added

    def print_leaving_move(self, search_context: SearchContext, time_control: TimeControl,
                           pv_idx: int, root_depth: int, move: Pos):
        # Simplified
        if self._show_realtime_in_loop(search_context, time_control, root_depth):
            if hasattr(search_context, 'board_size_for_output') and \
               not getattr(search_context, 'in_ponder', False) and \
               not engine_config.ASPIRATION_WINDOW and pv_idx == 0:
                # print(f"REALTIME DONE: {self._output_coord(move, search_context.board_size_for_output)}")
                pass

    def print_move_result(self, search_context: SearchContext, time_control: TimeControl,
                          pv_idx: int, num_pv: int, root_depth: int,
                          move: Pos, move_value: int, is_new_best: bool):
        # Simplified
        if self._show_realtime_in_loop(search_context, time_control, root_depth) and pv_idx == 0:
            if not getattr(search_context, 'in_ponder', False):
                if move_value <= Value.VALUE_MATED_IN_MAX_PLY.value:
                    # print(f"REALTIME LOST: {self._output_coord(move, getattr(search_context,'board_size_for_output',15))}")
                    pass
                elif is_new_best:
                    # print(f"REALTIME BEST: {self._output_coord(move, getattr(search_context,'board_size_for_output',15))}")
                    pass

    def print_pv_completes(self, search_context: SearchContext, time_control: TimeControl,
                           root_depth: int, pv_idx: int, num_pv: int,
                           current_root_move: RootMove, total_nodes: int):
        # search_context is typically MainSearchThread in Rapfi
        # current_root_move is th.rootMoves[pvIdx]
        # total_nodes is th.threads.nodesSearched()
        
        if getattr(search_context, 'in_ponder', False):
            return

        elapsed_ms = time_control.elapsed()
        speed = (total_nodes * 1000) // max(elapsed_ms, 1)
        
        # Assuming search_context provides options and board_size for output formatting
        board_size = getattr(search_context, 'board_size_for_output', 15)

        if self._show_info(search_context):
            # This is for INFO protocol, typically for GUIs like Piskvork
            print(f"INFO PV {pv_idx}")
            print(f"INFO NUMPV {num_pv}")
            print(f"INFO DEPTH {root_depth}")
            print(f"INFO SELDEPTH {current_root_move.sel_depth}")
            print(f"INFO NODES {current_root_move.nodes_searched}") # Nodes for this PV
            print(f"INFO TOTALNODES {total_nodes}")
            print(f"INFO TOTALTIME {elapsed_ms}")
            print(f"INFO SPEED {speed}") # Nodes per second
            print(f"INFO EVAL {current_root_move.value}")
            # print(f"INFO WINRATE {engine_config.value_to_win_rate(current_root_move.value)}") # If available
            pv_text = self._moves_to_text(current_root_move.pv, board_size)
            print(f"INFO BESTLINE {pv_text}")
            print(f"INFO PV DONE")
        
        if num_pv > 1 and engine_config.MESSAGE_MODE == engine_config.MsgMode.NORMAL:
            pv_text = self._moves_to_text(current_root_move.pv, board_size)
            print(f"({pv_idx + 1}) {current_root_move.value} | {root_depth}-{current_root_move.sel_depth} | {pv_text}")
        elif engine_config.MESSAGE_MODE == engine_config.MsgMode.UCILIKE:
            pv_text = self._moves_to_text(current_root_move.pv, board_size)
            prefix = f"depth {root_depth}-{current_root_move.sel_depth}"
            if num_pv > 1:
                prefix += f" multipv {pv_idx + 1}"
            print(f"{prefix} score cp {current_root_move.value} nodes {total_nodes} nps {speed} time {elapsed_ms} pv {pv_text}")


    def print_depth_completes(self, search_context: SearchContext, time_control: TimeControl, 
                              root_depth: int, best_root_move: RootMove):
        # best_root_move is th.rootMoves[0]
        if engine_config.MESSAGE_MODE == engine_config.MsgMode.NORMAL:
            is_pondering = getattr(search_context, 'in_ponder', False)
            board_size = getattr(search_context, 'board_size_for_output', 15)
            pv_text = self._moves_to_text(best_root_move.pv, board_size)
            
            print(f"{'[Pondering] ' if is_pondering else ''}"
                  f"Depth {root_depth}-{best_root_move.sel_depth} | "
                  f"Eval {best_root_move.value} | Time {time_text(time_control.elapsed())} | "
                  f"{pv_text}")

    def print_search_ends(self, search_context: SearchContext, time_control: TimeControl,
                        final_depth: int, best_thread_context: BestSearchContext, total_nodes: int):
    
        unique_call_id = time.time() # Get a timestamp for uniqueness
        print(f"DEBUG: print_search_ends called at {unique_call_id}") # Unique marker
        # final_depth is completed depth from best thread/overall
        # best_thread_context.rootMoves[0] gives the best PV and score
        # total_nodes is aggregated from all threads (or single thread total)
        if engine_config.MESSAGE_MODE == engine_config.MsgMode.NORMAL or \
            engine_config.MESSAGE_MODE == engine_config.MsgMode.BRIEF:
            
            elapsed_ms = time_control.elapsed()
            speed = (total_nodes * 1000) // max(elapsed_ms, 1)
            is_pondering = getattr(search_context, 'in_ponder', False)
            
            best_rm = getattr(best_thread_context, 'best_root_move_obj', None) # Assume best_thread_context has this
            if not best_rm and hasattr(best_thread_context, 'root_moves') and best_thread_context.root_moves:
                best_rm = best_thread_context.root_moves[0]
                

            if best_rm:
                board_size = getattr(search_context, 'board_size_for_output', 15)
                pv_text = self._moves_to_text(best_rm.pv, board_size)
                
                print(f"[{unique_call_id}] " # Add marker
                        f"{'[Pondering] ' if is_pondering else ''}"
                        f"Speed {speed_text(speed)} | Depth {final_depth}-{best_rm.sel_depth} | "
                        f"Eval {best_rm.value} | Node {nodes_text(total_nodes)} | "
                        f"Time {time_text(elapsed_ms)}")
                
                if engine_config.MESSAGE_MODE == engine_config.MsgMode.BRIEF or \
                   best_thread_context != search_context: # If best result came from another "thread"
                    # Output PV if brief mode or if best thread is different from main context
                    # (for multi-PV, this helps show the top PV line)
                    # Rapfi logic for choosing previousPv is if current PV is too short
                    chosen_pv = best_rm.pv
                    if hasattr(best_rm, 'previous_pv') and len(best_rm.pv) <=2 and len(best_rm.previous_pv) > 2:
                        chosen_pv = best_rm.previous_pv
                    pv_text_final = self._moves_to_text(chosen_pv, board_size)
                    print(f"[{unique_call_id}] Bestline {pv_text_final}")
                    # print(f"Bestline {pv_text_final}")
            else:
                print(f"[{unique_call_id}] " # Add marker
                  f"{'[Pondering] ' if is_pondering else ''}Search ended. No best move found? Time: {time_text(elapsed_ms)}")
                # print(f"{'[Pondering] ' if is_pondering else ''}Search ended. No best move found? Time: {time_text(elapsed_ms)}")


    def print_bestmove_uci_style(self, best_move: Pos, ponder_move: Optional[Pos] = None):
        """Prints the bestmove in a UCI-like format."""
        # This is the most critical output for GUIs
        # Piskvork uses simple coordinate, e.g., "7,7"
        # UCI for chess uses "e2e4"
        # For Gomoku, usually just the single move.
        if best_move == Pos.PASS:
            print("MESSAGE PASS") # Example for Piskvork if it supports pass
            print("pass") # UCI-like pass
        elif best_move != Pos.NONE:
            # Assume board_size might be needed for formatting if not simple x,y
            # For Piskvork, it expects "X,Y" without spaces.
            coord_str = f"{best_move.x},{best_move.y}"
            # print(f"MESSAGE बेस्ट चाल: {coord_str}") # Example for Piskvork debug
            print(f"{coord_str}") # Simple output for Piskvork protocol
            
            # For UCI style with ponder:
            # uci_output = f"bestmove {self._output_coord(best_move, MAX_BOARD_SIZE)}" # Needs board_size
            # if ponder_move and ponder_move != Pos.NONE:
            #     uci_output += f" ponder {self._output_coord(ponder_move, MAX_BOARD_SIZE)}"
            # print(uci_output)
        else:
            print("MESSAGE No valid best move found.")
            # print("nobestmove") # Or some error indication


if __name__ == '__main__':
    from .search_datastructures import RootMove, SearchOptions # Already here
    from .time_control import TimeControl
    # 'now' is imported at the module level now.

    print("--- SearchOutput Tests ---")
    printer = SearchPrinter()

    # Mock SearchContext (like MainSearchThread)
    class MockSearchContext:
        def __init__(self):
            self.options_obj = SearchOptions() # Use the imported SearchOptions
            # Access inner enums via the class:
            self.options_obj.info_mode = SearchOptions.InfoMode.INFO_REALTIME_AND_DETAIL
            self.in_ponder = False
            self.board_size_for_output = 15 
            self.root_moves: List[RootMove] = [] 
            self.best_root_move_obj: Optional[RootMove] = None 

        # Add the options() method that printer helpers expect
        def options(self) -> SearchOptions:
            return self.options_obj


    ctx = MockSearchContext()
    tc = TimeControl()
    tc.optimal_time = 2000
    tc.maximum_time = 5000
    tc.start_time = now() - 1000 # 'now' is now defined

    # ... (rest of the test cases) ...
    print("\nTest: print_search_starts")
    ctx.options_obj.time_limit_is_active = True
    printer.print_search_starts(ctx, tc) 

    print("\nTest: print_pv_completes (UCI-like for Piskvork INFO)")
    rm = RootMove(Pos(7,7))
    rm.value = 150
    rm.sel_depth = 10
    rm.nodes_searched = 10000
    rm.pv = [Pos(7,7), Pos(8,8), Pos(7,8), Pos.NONE] # Ensure pv is terminated if _moves_to_text expects it
    # Clean PV for text representation
    rm.pv = [p for p in rm.pv if p != Pos.NONE] 

    ctx.root_moves = [rm] 
    
    original_msg_mode = engine_config.MESSAGE_MODE
    
    engine_config.MESSAGE_MODE = engine_config.MsgMode.UCILIKE
    printer.print_pv_completes(search_context=ctx, time_control=tc, 
                               root_depth=8, pv_idx=0, num_pv=1,
                               current_root_move=rm, total_nodes=20000)

    print("\nTest: print_depth_completes (Normal mode)")
    engine_config.MESSAGE_MODE = engine_config.MsgMode.NORMAL
    printer.print_depth_completes(search_context=ctx, time_control=tc,
                                  root_depth=8, best_root_move=rm)

    print("\nTest: print_search_ends (Brief mode)")
    engine_config.MESSAGE_MODE = engine_config.MsgMode.BRIEF
    ctx.best_root_move_obj = rm 
    printer.print_search_ends(search_context=ctx, time_control=tc,
                              final_depth=8, best_thread_context=ctx, total_nodes=50000)
    
    engine_config.MESSAGE_MODE = original_msg_mode 

    print("\nTest: print_bestmove_uci_style (for Piskvork)")
    printer.print_bestmove_uci_style(Pos(3,4)) 
    printer.print_bestmove_uci_style(Pos.PASS)  

    print("SearchOutput tests completed.")