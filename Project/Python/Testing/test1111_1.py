import random
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import List, Tuple, Dict
from functools import lru_cache


def checkValid(size: int, move: Tuple[int, int]) -> bool:
    """
    Return True if move is within the board range.
    """
    return 0 <= move[0] < size and 0 <= move[1] < size


class Color(Enum):
    BLACK = 1
    WHITE = 2


class ColorFlag(Enum):
    SELF = auto()
    OPPO = auto()
    EMPT = auto()


class Pattern(Enum):
    DEAD = 0
    OL = 1
    B1 = 2
    F1 = 3
    B2 = 4
    F2 = 5
    F2A = 6
    F2B = 7
    B3 = 8
    F3 = 9
    F3S = 10
    B4 = 11
    F4 = 12
    F5 = 13
    PATTERN_NB = 14


class BitBoardABC(ABC):
    @abstractmethod
    def hash(self) -> int:
        ...

    @abstractmethod
    def addMove(self, move: Tuple[int, int], player: int):
        ...

    @abstractmethod
    def getState(self, move: Tuple[int, int]) -> int:
        ...

    @abstractmethod
    def isWin(self, player: int) -> bool:
        ...


class CandidateABC(ABC):
    @abstractmethod
    def expand(self, board: BitBoardABC) -> List[Tuple[int, int]]:
        ...


class Candidate(CandidateABC):
    def __init__(self, mode=0, size=15):
        """
        Mode 0 = SQUARE_3_LINE_4
        Mode 1 = CIRCLE_SQRT_34
        Mode 2 = FULL_BOARD
        """
        self.__mode = mode
        self.__size = size

    def expand(self, boardState: BitBoardABC) -> List[Tuple[int, int]]:
        candidate = []
        for row in range(self.__size):
            for col in range(self.__size):
                state = boardState.getState((row, col))
                if state in (0b01, 0b10):
                    if self.__mode == 0:
                        self.__squareLine(boardState, row, col, 3, 4)
                    elif self.__mode == 1:
                        self.__circle34(boardState, row, col)
                    elif self.__mode == 2:
                        self.__fullBoard(boardState)
                    else:
                        print('Not supported mode:', self.__mode)

        for row in range(self.__size):
            for col in range(self.__size):
                if boardState.getState((row, col)) == 0b11:
                    candidate.append((row, col))

        return candidate

    def __squareLine(self, boardState: BitBoardABC, x: int, y: int, sq: int, ln: int):
        """
        Mark candidate positions based on square and line patterns.
        """
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]
        for k in range(1, ln + 1):
            for i, j in directions:
                coord = (x + i * k, y + j * k)
                if checkValid(boardState.size, coord) and boardState.getState(coord) == 0:
                    boardState.addMove(coord, 3)

        for i in range(1, sq + 1):
            for j in range(1, sq + 1):
                coords = [
                    (x + i, y + j),
                    (x + i, y - j),
                    (x - i, y + j),
                    (x - i, y - j)
                ]
                for coord in coords:
                    if checkValid(boardState.size, coord) and boardState.getState(coord) == 0:
                        boardState.addMove(coord, 3)

    def __circle34(self, boardState: BitBoardABC, x: int, y: int):
        """
        Mark candidate positions based on circular patterns with sqrt(34) radius.
        """
        cr34 = 34 ** 0.5
        for row in range(-int(cr34), int(cr34) + 1):
            for col in range(-int(cr34), int(cr34) + 1):
                if (row ** 2 + col ** 2) ** 0.5 <= cr34:
                    coord = (x + row, y + col)
                    if checkValid(boardState.size, coord) and boardState.getState(coord) == 0:
                        boardState.addMove(coord, 3)

    def __fullBoard(self, boardState: BitBoardABC):
        """
        Mark all empty positions as candidates.
        """
        for row in range(boardState.size):
            for col in range(boardState.size):
                if boardState.getState((row, col)) == 0:
                    boardState.addMove((row, col), 3)


class BitBoard(BitBoardABC):
    def __init__(self, size=15):
        self.size = size
        self.__zobristTable: Dict[Tuple[int, int, int], int] = self.__generateZobristTable()
        self.board = [[0 for _ in range(size)] for _ in range(size)]  # 0: EMPTY, 1: BLACK, 2: WHITE
        self.last_move = None

    def __generateZobristTable(self) -> Dict[Tuple[int, int, int], int]:
        """
        Generates a Zobrist hashing table.
        """
        table = {}
        for row in range(self.size):
            for col in range(self.size):
                for player in [1, 2]:
                    table[(row, col, player)] = random.getrandbits(64)
        return table

    def getState(self, move: Tuple[int, int]) -> int:
        if not checkValid(self.size, move):
            return -1

        row, col = move
        state = self.board[row][col]
        if state == 1:
            return 1
        elif state == 2:
            return 2
        elif state == 3:
            return 3
        else:
            return 0

    def hash(self) -> int:
        """
        Generates a Zobrist hash for the current board state.
        """
        hashValue = 0
        for row in range(self.size):
            for col in range(self.size):
                state = self.board[row][col]
                if state in [1, 2]:
                    hashValue ^= self.__zobristTable[(row, col, state)]
        return hashValue

    def addMove(self, move: Tuple[int, int], player: int):
        """
        Adds a move to the board.
        """
        if self.getState(move) not in [0, 3]:
            return  # Invalid move, already occupied

        row, col = move
        self.board[row][col] = player
        self.last_move = move

    def view(self) -> str:
        """
        Returns a string representation of the board.
        """
        display = ""
        for row in range(self.size):
            line = ""
            for col in range(self.size):
                state = self.board[row][col]
                if state == 1:
                    line += "X  "
                elif state == 2:
                    line += "O  "
                elif state == 3:
                    line += "*  "  # Marked for candidate moves
                else:
                    line += ".  "
            display += line.strip() + "\n"
        return display.strip()

    def debugDisplayBitBoard(self):
        """
        Prints the binary representation of the board.
        """
        print("BitBoard (Binary Representation):")
        for row in range(self.size):
            row_bits = ""
            for col in range(self.size):
                state_bits = self.board[row][col]
                row_bits += f"{state_bits:02b} "
            print(row_bits)
        print()

    def isWin(self, player: int) -> bool:
        """
        Checks if the specified player has won the game.
        """
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for row in range(self.size):
            for col in range(self.size):
                if self.board[row][col] == player:
                    for dRow, dCol in directions:
                        count = 1
                        r, c = row + dRow, col + dCol
                        while 0 <= r < self.size and 0 <= c < self.size and self.board[r][c] == player:
                            count += 1
                            r += dRow
                            c += dCol
                        if count >= 5:
                            # Check for exact five (not overline)
                            before_r, before_c = row - dRow, col - dCol
                            after_r, after_c = r, c
                            before = self.board[before_r][before_c] if checkValid(self.size, (before_r, before_c)) else -1
                            after = self.board[after_r][after_c] if checkValid(self.size, (after_r, after_c)) else -1
                            if before != player and after != player:
                                return True
        return False

    def getPossibleMoves(self, candidate: CandidateABC) -> List[Tuple[int, int]]:
        """
        Retrieves possible moves based on the candidate generator.
        """
        return candidate.expand(self)


class PatternDetector:
    def __init__(self, rule: str, side: Color):
        """
        Initializes the PatternDetector.

        Args:
            rule (str): The game rule ('FREESTYLE', 'STANDARD', 'RENJU').
            side (Color): The current player's color (Color.BLACK or Color.WHITE).
        """
        self.rule = rule
        self.side = side

    @staticmethod
    def count_line(line: List[ColorFlag]) -> Tuple[int, int, int, int]:
        """
        Counts the number of consecutive SELF stones and the full length of the line.

        Args:
            line (List[ColorFlag]): The line to analyze.

        Returns:
            Tuple[int, int, int, int]: (real_len, full_len, start, end)
        """
        mid = len(line) // 2
        real_len = 1  # Start with the center stone
        full_len = 1
        real_len_inc = 1
        start = mid
        end = mid

        # Left side
        for i in range(mid - 1, -1, -1):
            if line[i] == ColorFlag.SELF:
                real_len += real_len_inc
            elif line[i] == ColorFlag.OPPO:
                break
            else:
                real_len_inc = 0
            full_len += 1
            start = i

        # Right side
        real_len_inc = 1
        for i in range(mid + 1, len(line)):
            if line[i] == ColorFlag.SELF:
                real_len += real_len_inc
            elif line[i] == ColorFlag.OPPO:
                break
            else:
                real_len_inc = 0
            full_len += 1
            end = i

        return real_len, full_len, start, end

    @staticmethod
    @lru_cache(maxsize=None)
    def get_pattern(line_tuple: Tuple[ColorFlag, ...], rule: str, side: Color) -> Pattern:
        """
        Determines the pattern of a given line based on the current rule and side.

        Args:
            line_tuple (Tuple[ColorFlag, ...]): The line represented as a tuple of ColorFlags.
            rule (str): The game rule ('FREESTYLE', 'STANDARD', 'RENJU').
            side (Color): The current player's color.

        Returns:
            Pattern: The detected pattern.
        """
        line = list(line_tuple)
        real_len, full_len, start, end = PatternDetector.count_line(line)
        pattern = Pattern.DEAD

        # Check for Overline (OL)
        if rule in ['STANDARD', 'RENJU'] and real_len >= 6:
            return Pattern.OL
        elif real_len >= 5:
            return Pattern.F5
        elif full_len < 5:
            return Pattern.DEAD
        else:
            # Initialize pattern counts
            pattern_counts = {p: 0 for p in Pattern}
            f5_indices = []

            # Iterate through the line to find empty positions
            for i in range(start, end + 1):
                if line[i] == ColorFlag.EMPT:
                    # Simulate placing a stone at position i
                    new_line = line.copy()
                    new_line[i] = ColorFlag.SELF
                    new_pattern = PatternDetector.get_pattern(tuple(new_line), rule, side)
                    pattern_counts[new_pattern] += 1

                    if new_pattern == Pattern.F5 and len(f5_indices) < 2:
                        f5_indices.append(i)

            # Determine the pattern based on pattern counts
            if pattern_counts[Pattern.F5] >= 2:
                pattern = Pattern.F4
                if rule == 'RENJU' and side == Color.BLACK:
                    # Check if the two F5 patterns are within 5 positions
                    if f5_indices[1] - f5_indices[0] < 5:
                        pattern = Pattern.OL
            elif pattern_counts[Pattern.F5] == 1:
                pattern = Pattern.B4
            elif pattern_counts[Pattern.F4] >= 2:
                pattern = Pattern.F3S
            elif pattern_counts[Pattern.F4] == 1:
                pattern = Pattern.F3
            elif pattern_counts[Pattern.B4] >= 1:
                pattern = Pattern.B3
            elif (pattern_counts[Pattern.F3S] + pattern_counts[Pattern.F3]) >= 4:
                pattern = Pattern.F2B
            elif (pattern_counts[Pattern.F3S] + pattern_counts[Pattern.F3]) >= 3:
                pattern = Pattern.F2A
            elif (pattern_counts[Pattern.F3S] + pattern_counts[Pattern.F3]) >= 1:
                pattern = Pattern.F2
            elif pattern_counts[Pattern.B3] >= 1:
                pattern = Pattern.B2
            elif (pattern_counts[Pattern.F2] + pattern_counts[Pattern.F2A] + pattern_counts[Pattern.F2B]) >= 1:
                pattern = Pattern.F1
            elif pattern_counts[Pattern.B2] >= 1:
                pattern = Pattern.B1
            else:
                pattern = Pattern.DEAD

            return pattern

    def extract_line(self, board: BitBoard, move: Tuple[int, int], dRow: int, dCol: int) -> List[ColorFlag]:
        """
        Extracts a line from the board in a specific direction centered at the move.

        Args:
            board (BitBoard): The current state of the board.
            move (Tuple[int, int]): The center move.
            dRow (int): Row direction.
            dCol (int): Column direction.

        Returns:
            List[ColorFlag]: The extracted line as a list of ColorFlags.
        """
        size = board.size
        x, y = move
        line = []

        for i in range(-4, 5):  # Line length of 9
            r = x + dRow * i
            c = y + dCol * i
            if 0 <= r < size and 0 <= c < size:
                state = board.getState((r, c))
                if state == 1:
                    line.append(ColorFlag.SELF)
                elif state == 2:
                    line.append(ColorFlag.OPPO)
                else:
                    line.append(ColorFlag.EMPT)
            else:
                line.append(ColorFlag.OPPO)  # Treat out-of-bounds as opponent

        return line

    def evaluate_patterns(self, board: BitBoard, move: Tuple[int, int]) -> List[Pattern]:
        """
        Evaluates patterns around the given move in all four directions.

        Args:
            board (BitBoard): The current state of the board.
            move (Tuple[int, int]): The move to evaluate.

        Returns:
            List[Pattern]: A list of detected patterns in each direction.
        """
        patterns = []
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # Horizontal, Vertical, Diagonal /, Diagonal \

        for dRow, dCol in directions:
            line = self.extract_line(board, move, dRow, dCol)
            if line:
                # Convert ColorFlags to tuple for caching
                line_tuple = tuple(line)
                pattern = self.get_pattern(line_tuple, self.rule, self.side)
                patterns.append(pattern)

        return patterns


class Evaluator:
    def __init__(self):
        """
        Initializes the Evaluator.
        """
        pass

    def evaluate(self, patterns: List[Pattern], side: Color) -> int:
        """
        Evaluates the board based on detected patterns.

        Args:
            patterns (List[Pattern]): Detected patterns around the last move.
            side (Color): The current player's color.

        Returns:
            int: The evaluation score.
        """
        score = 0
        for pattern in patterns:
            if side == Color.BLACK:
                if pattern == Pattern.F5:
                    score += 100000
                elif pattern == Pattern.F4:
                    score += 10000
                elif pattern == Pattern.F3S:
                    score += 1000
                elif pattern == Pattern.F3:
                    score += 100
                elif pattern == Pattern.F2B:
                    score += 10
                elif pattern == Pattern.F2A:
                    score += 5
                elif pattern == Pattern.F2:
                    score += 2
                elif pattern == Pattern.B4:
                    score += 5000
                elif pattern == Pattern.B3:
                    score += 500
                elif pattern == Pattern.B2:
                    score += 50
                elif pattern == Pattern.B1:
                    score += 5
            elif side == Color.WHITE:
                # Assign different scores for white if necessary
                if pattern == Pattern.F5:
                    score += 100000
                elif pattern == Pattern.F4:
                    score += 10000
                elif pattern == Pattern.F3S:
                    score += 1000
                elif pattern == Pattern.F3:
                    score += 100
                elif pattern == Pattern.F2B:
                    score += 10
                elif pattern == Pattern.F2A:
                    score += 5
                elif pattern == Pattern.F2:
                    score += 2
                elif pattern == Pattern.B4:
                    score += 5000
                elif pattern == Pattern.B3:
                    score += 500
                elif pattern == Pattern.B2:
                    score += 50
                elif pattern == Pattern.B1:
                    score += 5
        return score


class TreeNode:
    def __init__(self, root: bool, boardState: BitBoard, depth: int, score: int, hash_val: int, priority: int = 0):
        self.root: bool = root
        self.boardState: BitBoard = boardState
        self.depth: int = depth
        self.score: int = score
        self.hash_val: int = hash_val
        self.priority: int = priority
        self.children: List['TreeNode'] = []

    def add_child(self, child: 'TreeNode'):
        self.children.append(child)


class Search:
    TRANSPOSITION_TABLE: Dict[int, Tuple[int, int, int]] = {}  # hash: (depth, score, priority)

    def __init__(self):
        self.__evaluator = Evaluator()
        self.__pattern_detector = PatternDetector(rule='STANDARD', side=Color.BLACK)

    def alphabeta(self, node: TreeNode, depth: int, alpha: int, beta: int, maximizePlayer: bool) -> int:
        """
        Alpha-Beta pruning algorithm.
        """
        if depth == 0 or node.boardState.isWin(Color.BLACK.value) or node.boardState.isWin(Color.WHITE.value):
            return self.__evaluator.evaluate(
                self.__pattern_detector.evaluate_patterns(node.boardState, node.boardState.last_move),
                Color.BLACK if maximizePlayer else Color.WHITE
            )

        if node.hash_val in self.TRANSPOSITION_TABLE:
            tt_depth, tt_score, tt_priority = self.TRANSPOSITION_TABLE[node.hash_val]
            if tt_depth >= depth:
                return tt_score

        possible_moves = node.boardState.getPossibleMoves(Candidate(mode=0, size=node.boardState.size))
        if not possible_moves:
            return 0  # Draw

        if maximizePlayer:
            value = -float('inf')
            for move in possible_moves:
                child_board = BitBoard(node.boardState.size)
                # Copy the board state
                child_board.board = [row.copy() for row in node.boardState.board]
                child_board.last_move = node.boardState.last_move
                child_board.addMove(move, Color.BLACK.value)
                child_hash = child_board.hash()
                child_node = TreeNode(False, child_board, depth - 1, 0, child_hash)
                score = self.alphabeta(child_node, depth - 1, alpha, beta, False)
                value = max(value, score)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            self.TRANSPOSITION_TABLE[node.hash_val] = (depth, value, 0)
            return value
        else:
            value = float('inf')
            for move in possible_moves:
                child_board = BitBoard(node.boardState.size)
                # Copy the board state
                child_board.board = [row.copy() for row in node.boardState.board]
                child_board.last_move = node.boardState.last_move
                child_board.addMove(move, Color.WHITE.value)
                child_hash = child_board.hash()
                child_node = TreeNode(False, child_board, depth - 1, 0, child_hash)
                score = self.alphabeta(child_node, depth - 1, alpha, beta, True)
                value = min(value, score)
                beta = min(beta, value)
                if alpha >= beta:
                    break
            self.TRANSPOSITION_TABLE[node.hash_val] = (depth, value, 0)
            return value

    def IDS_Search(self, node: TreeNode, max_depth: int) -> int:
        """
        Iterative Deepening Search.
        """
        best_score = -float('inf')
        for depth in range(1, max_depth + 1):
            score = self.alphabeta(node, depth, -float('inf'), float('inf'), True)
            best_score = max(best_score, score)
        return best_score

    def PVS_Search(self, node: TreeNode, depth: int, alpha: int, beta: int, first_child: bool) -> int:
        """
        Principal Variation Search.
        """
        if depth == 0 or node.boardState.isWin(Color.BLACK.value) or node.boardState.isWin(Color.WHITE.value):
            return self.__evaluator.evaluate(
                self.__pattern_detector.evaluate_patterns(node.boardState, node.boardState.last_move),
                Color.BLACK if first_child else Color.WHITE
            )

        if node.hash_val in self.TRANSPOSITION_TABLE:
            tt_depth, tt_score, tt_priority = self.TRANSPOSITION_TABLE[node.hash_val]
            if tt_depth >= depth:
                return tt_score

        possible_moves = node.boardState.getPossibleMoves(Candidate(mode=0, size=node.boardState.size))
        if not possible_moves:
            return 0  # Draw

        best_score = -float('inf') if first_child else float('inf')

        for index, move in enumerate(possible_moves):
            child_board = BitBoard(node.boardState.size)
            # Copy the board state
            child_board.board = [row.copy() for row in node.boardState.board]
            child_board.last_move = node.boardState.last_move
            player = Color.BLACK.value if maximizePlayer else Color.WHITE.value
            child_board.addMove(move, player)
            child_hash = child_board.hash()
            child_node = TreeNode(False, child_board, depth - 1, 0, child_hash)
            if index == 0:
                score = self.PVS_Search(child_node, depth - 1, alpha, beta, False)
            else:
                score = self.PVS_Search(child_node, depth - 1, alpha, alpha + 1, False)
                if alpha < score < beta:
                    score = self.PVS_Search(child_node, depth - 1, score, beta, False)
            if first_child:
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                if alpha >= beta:
                    break
            else:
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                if alpha >= beta:
                    break
        if first_child:
            self.TRANSPOSITION_TABLE[node.hash_val] = (depth, best_score, 0)
        else:
            self.TRANSPOSITION_TABLE[node.hash_val] = (depth, best_score, 0)
        return best_score

    def VCF_Search(self, node: TreeNode, depth: int, alpha: int, beta: int, maximizePlayer: bool) -> int:
        """
        Very-Cautious-Full Search (Placeholder).
        """
        # Implement VCF search logic
        return self.alphabeta(node, depth, alpha, beta, maximizePlayer)

    def VCT_Search(self, node: TreeNode, depth: int, alpha: int, beta: int, maximizePlayer: bool) -> int:
        """
        Very-Cautious-Threat Search (Placeholder).
        """
        # Implement VCT search logic
        return self.alphabeta(node, depth, alpha, beta, maximizePlayer)

    def attackSearch(self, node: TreeNode, depth: int, pattern: Pattern, score: int) -> int:
        """
        Attack Search (Placeholder).
        """
        # Implement attack search logic
        return self.alphabeta(node, depth, -float('inf'), float('inf'), True)

    def defendSearch(self, node: TreeNode, depth: int, score: int) -> int:
        """
        Defend Search (Placeholder).
        """
        # Implement defend search logic
        return self.alphabeta(node, depth, -float('inf'), float('inf'), False)


# TESTING THE IMPLEMENTATION
if __name__ == "__main__":
    # Initialize BitBoard and PatternDetector
    bit_board = BitBoard(15)
    pattern_detector = PatternDetector(rule='STANDARD', side=Color.BLACK)

    # Make some moves
    bit_board.addMove((7, 7), 1)  # Black
    bit_board.addMove((7, 6), 2)  # White
    bit_board.addMove((6, 6), 1)  # Black
    bit_board.addMove((5, 5), 2)  # White
    bit_board.addMove((6, 8), 1)  # Black
    bit_board.addMove((6, 5), 2)  # White
    bit_board.addMove((6, 7), 1)  # Black
    bit_board.addMove((5, 7), 2)  # White
    bit_board.addMove((5, 9), 1)  # Black

    print("Current Board:")
    print(bit_board.view())
    print("\nBinary Representation:")
    bit_board.debugDisplayBitBoard()

    # Evaluate patterns after the last move
    patterns = pattern_detector.evaluate_patterns(bit_board, (5, 9))
    print("\nDetected Patterns:", [pattern.name for pattern in patterns])

    # Initialize Search
    root_node = TreeNode(True, bit_board, depth=4, score=0, hash_val=bit_board.hash())
    search_engine = Search()

    # Perform Alpha-Beta Search
    best_score = search_engine.alphabeta(root_node, depth=4, alpha=-float('inf'), beta=float('inf'), maximizePlayer=True)
    print("\nBest Score from Alpha-Beta Search:", best_score)
