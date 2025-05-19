import time
import logging
from typing import Tuple, List, Optional
from .types import Color, Value, Rule, Bound, DEPTH_LOWER_BOUND, DEPTH_UPPER_BOUND, VALUE_INFINITE
from .board import Board
from .movegen import MovePicker
from .eval import Evaluator

logger = logging.getLogger(__name__)

class SearchThread:
    """
    Base class to hold search-related data.
    """
    def __init__(self, board: Board, evaluator: Evaluator, rule: Rule = Rule.FREESTYLE):
        self.board = board
        self.evaluator = evaluator
        self.rule = rule
        self.nodes_visited = 0
        self.root_moves = []  # List of (move, value, pv) tuples
        self.transposition_table = {}
        self.sel_depth = 0

class MainSearchThread(SearchThread):
    """
    Subclass for the main search thread, extending SearchThread with additional attributes.
    """
    def __init__(self, board: Board, evaluator: Evaluator, rule: Rule = Rule.FREESTYLE):
        super().__init__(board, evaluator, rule)
        self.search_options = SearchOptions(rule)
        self.best_move = None
        self.pv = []

class SearchOptions:
    """
    Configuration options for the search.
    """
    def __init__(self, rule: Rule):
        self.rule = rule
        self.max_depth = 16
        self.time_limit = True
        self.max_nodes = 0
        self.info_mode = 0
        self.balance_mode = 0

class ABSearcher:
    """
    Implements the alpha-beta pruning search algorithm.
    """
    def __init__(self):
        self.timectl = TimeControl()
        self.nodes_visited = 0

    def _pos_to_notation(self, pos: Tuple[int, int]) -> str:
        """
        Convert board position to standard notation (e.g., 'J10').
        """
        col = chr(ord('A') + pos[1] - 1)
        if col >= 'I':
            col = chr(ord(col) + 1)
        row = pos[0] + 1
        return f"{col}{row}"

    def search(self, thread: MainSearchThread):
        """
        Perform iterative deepening search up to max_depth or time limit.
        """
        depth = 1
        max_depth = thread.search_options.max_depth
        start_time = time.time()
        self.nodes_visited = 0
        thread.sel_depth = 0

        while depth <= max_depth and time.time() - start_time < self.timectl.maximum():
            value, pv = self._alphabeta(thread, depth, Value(-VALUE_INFINITE), Value(VALUE_INFINITE), thread.board.side_to_move(), thread.rule, 0)
            thread.root_moves.append((thread.best_move, value, pv))
            thread.best_move = pv[0] if pv else None
            thread.pv = pv

            elapsed_time = max(1, int(time.time() - start_time))
            nodes_str = f"{self.nodes_visited // 1000}K" if self.nodes_visited >= 1000 else str(self.nodes_visited)
            nps = int(self.nodes_visited / elapsed_time)
            pv_str = " ".join(self._pos_to_notation(move) for move in pv) if pv else ""
            logger.info(f"DEPTH {depth}-{thread.sel_depth} EV {value} N {nodes_str} NPS {nps} TM {elapsed_time} PV {pv_str}")
            depth += 1

    def _alphabeta(self, thread: SearchThread, depth: float, alpha: Value, beta: Value, side: Color, rule: Rule, ply: int) -> Tuple[Value, List[Tuple[int, int]]]:
        """
        Alpha-beta pruning algorithm.
        Returns the evaluation value and the principal variation (PV).
        """
        self.nodes_visited += 1
        thread.sel_depth = max(thread.sel_depth, ply)

        if depth <= 0:
            return self._evaluate_vcf(thread.board, side, rule), []

        best_value = Value(-VALUE_INFINITE) if side == Color.BLACK else Value(VALUE_INFINITE)
        best_move = None
        pv = []

        move_picker = MovePicker(rule, thread.board)
        move_count = 0
        first_move = None
        while (move := move_picker()) is not None:
            if move_count == 0:
                first_move = move
            move_count += 1
            thread.board.move(move)
            value, child_pv = self._alphabeta(thread, depth - 1, -beta, -alpha, Color.WHITE if side == Color.BLACK else Color.BLACK, rule, ply + 1)
            value = -value
            thread.board.undo()

            if side == Color.BLACK:
                if value > best_value:
                    best_value = value
                    best_move = move
                    pv = [move] + child_pv
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break
            else:  # Color.WHITE
                if value < best_value:
                    best_value = value
                    best_move = move
                    pv = [move] + child_pv
                    beta = min(beta, value)
                    if alpha >= beta:
                        break

        if not best_move and first_move:
            best_move = first_move
            pv = [first_move]
            best_value = self._evaluate_vcf(thread.board, side, rule)

        return best_value, pv

    def _evaluate_vcf(self, board: Board, side: Color, rule: Rule) -> Value:
        """
        Evaluate the board position using the Evaluator class.
        """
        evaluator = Evaluator(board.size, board=board, rule=rule)
        return evaluator.evaluate(board, side)

class TimeControl:
    """
    Manage time limits for the search.
    """
    def __init__(self):
        self.start_time = time.time()
        self.max_time = 5.0

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def maximum(self) -> float:
        return self.max_time