# pattern.py

from typing import List, Tuple
from functools import lru_cache
from enums import Color, ColorFlag, Pattern
from board import BitBoard
from candidate import CandidateABC


class PatternDetector:
    def __init__(self, rule: str):
        """
        Initializes the PatternDetector without a fixed side.

        Args:
            rule (str): The rule set to use for pattern detection.
        """
        self.rule = rule
        # Initialize any additional attributes or precompute necessary data here

    @staticmethod
    def count_line(line: List[ColorFlag]) -> Tuple[int, int, int, int]:
        """
        Counts the number of consecutive SELF flags in a line.

        Args:
            line (List[ColorFlag]): The line of ColorFlags.

        Returns:
            Tuple[int, int, int, int]: realLen, fullLen, start index, end index.
        """
        mid = len(line) // 2
        realLen = 1
        fullLen = 1
        realLenInc = 1
        start = mid
        end = mid

        # Left side
        for i in range(mid - 1, -1, -1):
            if line[i] == ColorFlag.SELF:
                realLen += realLenInc
            elif line[i] == ColorFlag.OPPO:
                break
            else:
                realLenInc = 0
            fullLen += 1
            start = i

        # Right side
        realLenInc = 1
        for i in range(mid + 1, len(line)):
            if line[i] == ColorFlag.SELF:
                realLen += realLenInc
            elif line[i] == ColorFlag.OPPO:
                break
            else:
                realLenInc = 0
            fullLen += 1
            end = i

        return realLen, fullLen, start, end

    @staticmethod
    @lru_cache(maxsize=None)
    def get_pattern(line_tuple: Tuple[ColorFlag, ...], rule: str, side: Color) -> Pattern:
        """
        Determines the pattern based on the line of ColorFlags.

        Args:
            line_tuple (Tuple[ColorFlag, ...]): The line as a tuple of ColorFlags.
            rule (str): The rule set being used.
            side (Color): The current player's side.

        Returns:
            Pattern: The detected pattern.
        """
        line = list(line_tuple)
        realLen, fullLen, start, end = PatternDetector.count_line(line)
        pattern = Pattern.DEAD

        if rule in ['STANDARD', 'RENJU'] and realLen >= 6:
            return Pattern.OL
        elif realLen >= 5:
            return Pattern.F5
        elif fullLen < 5:
            return Pattern.DEAD
        else:
            pattern_counts = {p: 0 for p in Pattern}
            f5_indices = []

            for i in range(start, end + 1):
                if line[i] == ColorFlag.EMPT:
                    new_line = line.copy()
                    new_line[i] = ColorFlag.SELF
                    new_pattern = PatternDetector.get_pattern(tuple(new_line), rule, side)
                    pattern_counts[new_pattern] += 1

                    if new_pattern == Pattern.F5 and len(f5_indices) < 2:
                        f5_indices.append(i)

            if pattern_counts[Pattern.F5] >= 2:
                pattern = Pattern.F4
                if rule == 'RENJU' and side == Color.BLACK:
                    if len(f5_indices) >= 2 and (f5_indices[1] - f5_indices[0] < 5):
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

    def extract_line(self, board: BitBoard, move: Tuple[int, int], side: Color, dRow: int, dCol: int) -> List[ColorFlag]:
        """
        Extracts a line of ColorFlags from the board based on direction.

        Args:
            board (BitBoard): The current state of the board.
            move (Tuple[int, int]): The move position (row, column).
            dRow (int): The row direction increment.
            dCol (int): The column direction increment.

        Returns:
            List[ColorFlag]: The extracted line as a list of ColorFlags.
        """
        size = board.size
        x, y = move
        line = []

        for i in range(-4, 5):
            r = x + dRow * i
            c = y + dCol * i
            if 0 <= r < size and 0 <= c < size:
                state = board.get_state((r, c))
                if state == side.value:
                    line.append(ColorFlag.SELF)
                elif state == 0:
                    line.append(ColorFlag.EMPT)
                else:
                    line.append(ColorFlag.OPPO)
            else:
                line.append(ColorFlag.OPPO)  # Treat out-of-bounds as opponent
        return line

    def _evaluate_patterns(self, board: BitBoard, move: Tuple[int, int], side: Color) -> List[Pattern]:
        """
        Evaluates patterns on the board based on the move and side, ensuring no duplicates.

        Args:
            board (BitBoard): The current state of the board.
            move (Tuple[int, int]): The move position (row, column).
            side (Color): The side to evaluate patterns for.

        Returns:
            List[Pattern]: A list of detected patterns.
        """
        patterns = []
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # Vertical, Horizontal, Diagonal /, Diagonal \

        for dRow, dCol in directions:
            # Determine if the current move is the starting point in this direction
            prev_r = move[0] - dRow
            prev_c = move[1] - dCol
            if 0 <= prev_r < board.size and 0 <= prev_c < board.size:
                if board.get_state((prev_r, prev_c)) == side.value:
                    continue  # Not the starting point; pattern already evaluated from the previous stone
            # Extract the line from the starting point
            line = self.extract_line(board, move, side, dRow, dCol)
            if line:
                line_tuple = tuple(line)
                pattern = self.get_pattern(line_tuple, self.rule, side)
                patterns.append(pattern)

        return patterns
    
    def evaluate_patterns(self, board: BitBoard, side: Color) -> List[Pattern]:
        patterns = []
        for y in range(15):
            for x in range(15):
                if board.get_state((y, x)) == side.value:
                    patterns.extend(self._evaluate_patterns(board, (y, x), side))
        return patterns

    def evaluate_move_patterns(self, board: BitBoard, move: Tuple[int, int], side: Color) -> List[Pattern]:
            """
            Evaluates patterns on the board based on a specific move and side.

            Args:
                board (BitBoard): The current state of the board.
                move (Tuple[int, int]): The move position (row, column).
                side (Color): The side to evaluate patterns for.

            Returns:
                List[Pattern]: A list of detected patterns related to the move.
            """
            patterns = []
            directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # Vertical, Horizontal, Diagonal /, Diagonal \

            for dRow, dCol in directions:
                # Extract the line in the current direction
                line = self.extract_line(board, move, side, dRow, dCol)
                if line:
                    line_tuple = tuple(line)
                    pattern = self.get_pattern(line_tuple, self.rule, side)
                    patterns.append(pattern)

            return patterns
    