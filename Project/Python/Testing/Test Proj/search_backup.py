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
    def __init__(self, root: bool, boardState: BitBoard, depth: int, score: float, hashVal: int, priority: int = 0):
        """
        Initializes a TreeNode representing a state in the game tree.

        Args:
            root (bool): Indicates if this node is the root of the tree.
            boardState (BitBoard): The current state of the board.
            depth (int): The depth of the node in the tree.
            score (float): The evaluation score of the node.
            hashVal (int): The Zobrist hash of the board state.
            priority (int, optional): Priority for move ordering. Defaults to 0.
        """
        self.root = root
        self.boardState = boardState
        self.depth = depth
        self.score = score
        self.hashVal = hashVal
        self.priority = priority
        self.children: List['TreeNode'] = []

    def add_child(self, child: 'TreeNode'):
        """
        Adds a child node to the current node.

        Args:
            child (TreeNode): The child node to be added.
        """
        self.children.append(child)


class Search:
    """
    Implements the Alpha-Beta pruning search algorithm with transposition tables.
    Allows dynamic switching of the player's side.
    """

    transpositionTable: Dict[int, Tuple[int, float]] = {}

    def __init__(self, ai_color: Color = Color.BLACK):
        """
        Initializes the Search class with the AI's color.

        Args:
            ai_color (Color, optional): The AI's color. Defaults to Color.BLACK.
        """
        self.evaluator = Evaluator()
        self.patternDetector = PatternDetector(rule='STANDARD')
        self.ai_color = ai_color

    def get_opponent_side(self, current_side: Color) -> Color:
        """
        Determines the opponent's side based on the current side.

        Args:
            current_side (Color): The current player's color.

        Returns:
            Color: The opponent's color.
        """
        return Color.WHITE if current_side == Color.BLACK else Color.BLACK

    def alphabeta(self, node: TreeNode, depth: int, alpha: float, beta: float, current_side: Color) -> float:
        """
        Performs the Alpha-Beta pruning search algorithm.

        Args:
            node (TreeNode): The current node in the game tree.
            depth (int): The remaining depth to search.
            alpha (float): The alpha value for pruning.
            beta (float): The beta value for pruning.
            current_side (Color): The side to move at the current node.

        Returns:
            float: The evaluation score.
        """
        # Terminal condition: depth == 0 or win/loss
        if depth == 0 or node.boardState.is_win(self.ai_color.value) or node.boardState.is_win(self.get_opponent_side(self.ai_color).value):
            # Detect patterns for both AI and opponent
            patterns_ai = self.patternDetector.evaluate_patterns(board=node.boardState, side=self.ai_color)
            patterns_opponent = self.patternDetector.evaluate_patterns(board=node.boardState, side=self.get_opponent_side(self.ai_color))
            # patterns_opponent = []
            
            # Combine patterns with ownership
            combined_patterns = [(pattern, self.ai_color) for pattern in patterns_ai] + \
                                [(pattern, self.get_opponent_side(self.ai_color)) for pattern in patterns_opponent]
            
            # Evaluate the score
            score = self.evaluator.evaluate(combined_patterns, self.ai_color) * (depth + 1)
            return score

        # Check transposition table
        if node.hashVal in self.transpositionTable:
            tt_depth, tt_score = self.transpositionTable[node.hashVal]
            if tt_depth >= depth:
                return tt_score

        # Generate possible moves
        possible_moves = node.boardState.get_possible_moves(Candidate(mode=0, size=node.boardState.size))
        if not possible_moves:
            return 0.0  # Draw or no moves available

        # Determine if the current player is maximizing or minimizing
        is_maximizing = (current_side == self.ai_color)
        opponent_side = self.get_opponent_side(current_side)

        if is_maximizing:
            value = -float('inf')
            for move in possible_moves:
                # Apply the move to generate a new board state
                child_board = node.boardState.copy()
                move_added = child_board.add_move(move, current_side.value)
                if not move_added:
                    continue  # Skip invalid moves
                # Generate a unique hash for the new board state
                child_hash = child_board.hash()
                # Create a new TreeNode for the child state
                child_node = TreeNode(False, child_board, depth - 1, 0.0, child_hash)
                # Recursively evaluate the move
                score = self.alphabeta(child_node, depth - 1, alpha, beta, opponent_side)
                # Update the value and alpha
                value = max(value, score)
                alpha = max(alpha, value)
                # Beta cutoff
                if alpha >= beta:
                    print('Beta Cutoff', move)
                    break  # Beta Cutoff
        else:
            value = float('inf')
            for move in possible_moves:
                # Apply the move to generate a new board state
                child_board = node.boardState.copy()
                move_added = child_board.add_move(move, current_side.value)
                if not move_added:
                    continue  # Skip invalid moves
                # Generate a unique hash for the new board state
                child_hash = child_board.hash()
                # Create a new TreeNode for the child state
                child_node = TreeNode(False, child_board, depth - 1, 0.0, child_hash)
                # Recursively evaluate the move
                score = self.alphabeta(child_node, depth - 1, alpha, beta, opponent_side)
                # Update the value and beta
                value = min(value, score)
                beta = min(beta, value)
                # Alpha cutoff
                if alpha >= beta:
                    break  # Alpha Cutoff

        # Store only the score in the transposition table
        self.transpositionTable[node.hashVal] = (depth, value)

        return value

    # TODO: add priority
    def find_best_move(self, node: TreeNode, depth: int, current_side: Color) -> Optional[Tuple[int, int]]:
        """
        Determines the best move for the AI using Alpha-Beta pruning, after filtering out useless moves.

        Args:
            node (TreeNode): The current node in the game tree.
            depth (int): The search depth.
            current_side (Color): The AI's side.

        Returns:
            Optional[Tuple[int, int]]: The best move found, or None if no moves are available.
        """
        # Generate all possible moves for the AI
        possible_moves = node.boardState.get_possible_moves(Candidate(mode=0, size=node.boardState.size))
        if not possible_moves:
            return None  # No moves available


        best_move = None
        best_score = -float('inf')  # Initialize to negative infinity for maximization

        stack = []

        # Add Preprocess -> Sort priority
        for move in possible_moves:
            child_board = node.boardState.copy()
            move_added = child_board.add_move(move, current_side.value)
            if child_board.is_win(current_side.value):
                return 100000, move
            pattern = self.patternDetector.evaluate_patterns(board=child_board, side=current_side)
            score = self.evaluator.static_evaluator(pattern, current_side)
            child_hash = child_board.hash()
            child_node = TreeNode(False, child_board, depth - 1, 0.0, child_hash, score)
            stack.append(child_node)
        # Sort stack (Sort TreeNode base on priority [the score])
    

        for move in possible_moves:
            # Apply the move to generate a new board state
            child_board = node.boardState.copy()
            move_added = child_board.add_move(move, current_side.value)
            if not move_added:
                continue  # Skip invalid moves

            if child_board.is_win(current_side.value):
                return 100000, move

            # Generate a unique hash for the new board state
            child_hash = child_board.hash()
            # Create a new TreeNode for the child state
            child_node = TreeNode(False, child_board, depth - 1, 0.0, child_hash)

            # Recursively evaluate the move using alphabeta
            score = self.alphabeta(child_node, depth - 1, -float('inf'), float('inf'), self.get_opponent_side(current_side))

            # Update best_move if a higher score is found
            if score > best_score:
                best_score = score
                best_move = move

        return best_score, best_move

    def timed_search(self, node: TreeNode, max_time: float) -> Tuple[Optional[Tuple[int, int]], float, int]:
        """
        Searches for the best move within the given time constraint using iterative deepening.

        Args:
            node (TreeNode): The root node of the search.
            max_time (float): Maximum time allowed for the search in seconds.

        Returns:
            Tuple[Optional[Tuple[int, int]], float, int]: The best move found, the best score, and the depth reached.
        """
        start_time = time.perf_counter()
        depth = 1
        best_move = None
        best_score = -float('inf')

        while True:
            elapsed_time = time.perf_counter() - start_time
            if elapsed_time >= max_time:
                break

            try:
                current_side = node.boardState.get_current_side()
                move = self.find_best_move(node, depth, current_side)
                if move is not None:
                    # Apply move to get the score
                    child_board = node.boardState.copy()
                    move_added = child_board.add_move(move, current_side.value)
                    if move_added:
                        child_hash = child_board.hash()
                        child_node = TreeNode(False, child_board, depth - 1, 0.0, child_hash)
                        score = self.alphabeta(child_node, depth - 1, -float('inf'), float('inf'), self.get_opponent_side(current_side))
                        if score > best_score:
                            best_score = score
                            best_move = move
                depth += 1
            except Exception as e:
                print(f"Error occurred during search: {e}")
                break

            # Check if there's enough time left for the next depth
            if time.perf_counter() - start_time >= max_time:
                break

        return best_move, best_score, depth - 1

    def ids_search(self, node: TreeNode, max_depth: int) -> Optional[Tuple[int, int]]:
        """
        Performs Iterative Deepening Search up to the specified maximum depth.

        Args:
            node (TreeNode): The root node of the search.
            max_depth (int): The maximum depth to search.

        Returns:
            Optional[Tuple[int, int]]: The best move found, or None if no move is available.
        """
        best_move = None
        best_score = -float('inf')
        for depth in range(1, max_depth + 1):
            current_side = node.boardState.get_current_side()
            move = self.find_best_move(node, depth, current_side)
            if move is not None:
                child_board = node.boardState.copy()
                move_added = child_board.add_move(move, current_side.value)
                if move_added:
                    child_hash = child_board.hash()
                    child_node = TreeNode(False, child_board, depth - 1, 0.0, child_hash)
                    score = self.alphabeta(child_node, depth - 1, -float('inf'), float('inf'), self.get_opponent_side(current_side))
                    if score > best_score:
                        best_score = score
                        best_move = move
        return best_move

    # Placeholder methods for additional search strategies
    def pvs_search(self, node: TreeNode, depth: int, alpha: float, beta: float, first_child: bool) -> float:
        """
        Implements Principal Variation Search (PVS) strategy.

        Args:
            node (TreeNode): The current node.
            depth (int): The remaining depth.
            alpha (float): Alpha value for pruning.
            beta (float): Beta value for pruning.
            first_child (bool): Indicates if the current node is the first child.

        Returns:
            float: The evaluated score.
        """
        return self.alphabeta(node, depth, alpha, beta, node.boardState.get_current_side())

    def vcf_search(self, node: TreeNode, depth: int, alpha: float, beta: float, maximize_player: bool) -> float:
        """
        Implements Variable Cutoff Function (VCF) search strategy.

        Args:
            node (TreeNode): The current node.
            depth (int): The remaining depth.
            alpha (float): Alpha value for pruning.
            beta (float): Beta value for pruning.
            maximize_player (bool): Indicates if the current search is maximizing.

        Returns:
            float: The evaluated score.
        """
        current_side = Color.BLACK if maximize_player else self.get_opponent_side(Color.BLACK)
        return self.alphabeta(node, depth, alpha, beta, current_side)

    def vct_search(self, node: TreeNode, depth: int, alpha: float, beta: float, maximize_player: bool) -> float:
        """
        Implements Variable Cutoff Threshold (VCT) search strategy.

        Args:
            node (TreeNode): The current node.
            depth (int): The remaining depth.
            alpha (float): Alpha value for pruning.
            beta (float): Beta value for pruning.
            maximize_player (bool): Indicates if the current search is maximizing.

        Returns:
            float: The evaluated score.
        """
        current_side = Color.BLACK if maximize_player else self.get_opponent_side(Color.BLACK)
        return self.alphabeta(node, depth, alpha, beta, current_side)

    def attack_search(self, node: TreeNode, depth: int, pattern: Pattern, score: float) -> float:
        """
        Implements an attack-specific search strategy based on detected patterns.

        Args:
            node (TreeNode): The current node.
            depth (int): The remaining depth.
            pattern (Pattern): The pattern to prioritize.
            score (float): The score associated with the pattern.

        Returns:
            float: The evaluated score.
        """
        return self.alphabeta(node, depth, -float('inf'), float('inf'), node.boardState.get_current_side())

    def defend_search(self, node: TreeNode, depth: int, score: float) -> float:
        """
        Implements a defense-specific search strategy.

        Args:
            node (TreeNode): The current node.
            depth (int): The remaining depth.
            score (float): The score associated with the defensive move.

        Returns:
            float: The evaluated score.
        """
        return self.alphabeta(node, depth, -float('inf'), float('inf'), node.boardState.get_current_side())
