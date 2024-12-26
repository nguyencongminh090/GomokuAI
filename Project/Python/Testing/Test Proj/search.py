# search.py

from typing import List, Tuple, Dict, Optional
from board import BitBoard
from pattern import PatternDetector
from evaluator import Evaluator
from candidate import Candidate
from enums import Color, Pattern
from interfaces import BitBoardABC
import time

class TreeNode:
    def __init__(
        self, 
        board_state: BitBoardABC, 
        hash_val: int, 
        parent: Optional['TreeNode'] = None, 
        best_move: Optional[Tuple[int, int]] = None
    ):
        self.board_state = board_state
        self.hash_val = hash_val
        self.parent = parent
        self.best_move = best_move
        self.score: Optional[float] = None
        self.children: List['TreeNode'] = []
        self.priority: float = 0.0

    def add_child(self, child: 'TreeNode'):
        self.children.append(child)

    def __repr__(self):
        return repr(self.priority)

class Search:
    def __init__(self, ai_color: Color = Color.BLACK):
        self.evaluator = Evaluator()
        self.pattern_detector = PatternDetector(rule='STANDARD')
        self.ai_color = ai_color
        self.transposition_table: Dict[int, Tuple[int, float]] = {}

    def get_opponent_side(self, current_side: Color) -> Color:
        return Color.WHITE if current_side == Color.BLACK else Color.BLACK

    def alphabeta(
        self, 
        node: TreeNode, 
        depth: int, 
        alpha: float, 
        beta: float, 
        current_side: Color
    ) -> float:
        if depth == 0 or node.board_state.is_win(self.ai_color.value) or node.board_state.is_win(self.get_opponent_side(self.ai_color).value):
            patterns_ai = self.pattern_detector.evaluate_patterns(node.board_state, self.ai_color)
            patterns_opponent = self.pattern_detector.evaluate_patterns(
                node.board_state, self.get_opponent_side(self.ai_color)
            )
            combined_patterns = [(p, self.ai_color) for p in patterns_ai] + \
                                [(p, self.get_opponent_side(self.ai_color)) for p in patterns_opponent]
            score = self.evaluator.evaluate(combined_patterns, self.ai_color) * (depth + 1)
            node.score = score
            return score

        if node.hash_val in self.transposition_table:
            tt_depth, tt_score = self.transposition_table[node.hash_val]
            if tt_depth >= depth:
                node.score = tt_score
                return tt_score

        possible_moves = node.board_state.get_possible_moves(Candidate(mode=0, size=node.board_state.size))
        if not possible_moves:
            node.score = 0.0
            return 0.0

        is_maximizing = (current_side == self.ai_color)
        opponent_side = self.get_opponent_side(current_side)
        value = -float('inf') if is_maximizing else float('inf')

        prioritized_nodes: List[TreeNode] = self.sort_priority(node, possible_moves, current_side)
        for child_node in prioritized_nodes:
            if child_node.hash_val in self.transposition_table:
                child_score = self.transposition_table[child_node.hash_val][1]
            else:
                child_score = self.alphabeta(child_node, depth - 1, -float('inf'), float('inf'), self.get_opponent_side(current_side))
            
            if is_maximizing:
                value = max(value, child_score)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            else:
                value = min(value, child_score)
                beta = min(beta, value)
                if alpha >= beta:
                    break

        # for move in possible_moves:
        #     child_board = node.board_state.copy()
        #     if not child_board.add_move(move, current_side.value):
        #         continue
        #     child_hash = child_board.hash()

        #     if child_hash in self.transposition_table:
        #         child_score = self.transposition_table[child_hash][1]
        #     else:
        #         child_node = TreeNode(child_board, child_hash, parent=node, best_move=move)
        #         # node.add_child(child_node)
        #         child_score = self.alphabeta(child_node, depth - 1, alpha, beta, opponent_side)

        #     if is_maximizing:
        #         if child_score > value:
        #             value = child_score
        #             node.best_move = move
        #         alpha = max(alpha, value)
        #         if alpha >= beta:
        #             break
        #     else:
        #         if child_score < value:
        #             value = child_score
        #             node.best_move = move
        #         beta = min(beta, value)
        #         if alpha >= beta:
        #             break

        node.score = value
        self.transposition_table[node.hash_val] = (depth, value)
        return value

    def sort_priority(self, node: TreeNode, possible_moves: List[Tuple[int, int]], current_side: Color) -> List[TreeNode]:
        prioritized_nodes = []
        sum_score = 0
        idx = 0
        for move in possible_moves:
            child_board = node.board_state.copy()  # Assuming BitBoard has copy_from method
            if not child_board.add_move(move, current_side.value):
                continue
            patterns = self.pattern_detector.evaluate_move_patterns(child_board, move, current_side)
            heuristic_score = self.evaluator.static_evaluator(patterns, current_side)
            child_hash = child_board.hash()
            child_node = TreeNode(child_board, child_hash, best_move=move)
            child_node.priority = heuristic_score
            sum_score += heuristic_score
            prioritized_nodes.append(child_node)
        
        prioritized_nodes.sort(key=lambda node: node.priority, reverse=True)
        avg_thresh = sum_score / len(prioritized_nodes)
        while True:
            if prioritized_nodes[idx].priority < avg_thresh:
                prioritized_nodes = prioritized_nodes[:idx - 1]
                return prioritized_nodes
            idx += 1
        
    def check_immediate_win(self, node: TreeNode, current_side: Color) -> Optional[Tuple[int, int]]:
        for child_node in node:
            if child_node.board_state.is_win(current_side.value):
                return node.best_move

    def find_best_move(self, node: TreeNode, depth: int, current_side: Color) -> Optional[Tuple[int, int]]:
        possible_moves = node.board_state.get_possible_moves(Candidate(mode=0, size=node.board_state.size))
        if not possible_moves:
            return None

        # win_move = self.check_immediate_win(possible_moves, current_side)
        # if win_move:
        #     node.best_move = win_move
        #     node.score = 100000
        #     return win_move

        prioritized_nodes: List[TreeNode] = self.sort_priority(node, possible_moves, current_side)
        
        check_win = self.check_immediate_win(prioritized_nodes, current_side)
        if check_win:
            return check_win

        best_move = None
        best_score = -float('inf')

        for child_node in prioritized_nodes:
            if child_node.hash_val in self.transposition_table:
                child_score = self.transposition_table[child_node.hash_val][1]
            else:
                child_score = self.alphabeta(child_node, depth - 1, -float('inf'), float('inf'), self.get_opponent_side(current_side))

            # print(child_node.best_move, child_score)

            if child_score > best_score:
                best_score = child_score
                best_move = child_node.best_move
                node.best_move = child_node.best_move

        return best_score, best_move

    def timed_search(self, node: TreeNode, max_time: float) -> Tuple[Optional[Tuple[int, int]], float, int]:
        start_time = time.perf_counter()
        depth = 1
        best_move = None
        best_score = -float('inf')

        while True:
            if time.perf_counter() - start_time >= max_time:
                break
            try:
                current_side = node.board_state.get_current_side()
                move = self.find_best_move(node, depth, current_side)
                if move:
                    child_board = node.board_state.copy()
                    if child_board.add_move(move, current_side.value):
                        child_hash = child_board.hash()
                        child_node = TreeNode(child_board, child_hash, parent=node, best_move=move)
                        score = self.alphabeta(child_node, depth - 1, -float('inf'), float('inf'), self.get_opponent_side(current_side))
                        if score > best_score:
                            best_score = score
                            best_move = move
                depth += 1
            except Exception as e:
                print(f"Error during search at depth {depth}: {e}")
                break
            if time.perf_counter() - start_time >= max_time:
                break

        return best_move, best_score, depth - 1

    def ids_search(self, node: TreeNode, max_depth: int) -> Optional[Tuple[int, int]]:
        best_move = None
        best_score = -float('inf')
        for depth in range(1, max_depth + 1):
            move = self.find_best_move(node, depth, node.board_state.get_current_side())
            if move:
                child_board = node.board_state.copy()
                if child_board.add_move(move, node.board_state.get_current_side().value):
                    child_hash = child_board.hash()
                    child_node = TreeNode(child_board, child_hash, parent=node, best_move=move)
                    score = self.alphabeta(child_node, depth - 1, -float('inf'), float('inf'), self.get_opponent_side(node.board_state.get_current_side()))
                    if score > best_score:
                        best_score = score
                        best_move = move
        return best_move
