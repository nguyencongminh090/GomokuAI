from typing import Tuple, List
from functools import lru_cache
from enums import Color, ColorFlag, Pattern
from board import BitBoard

class PatternDetector:
    def __init__(self, rule: str):
        self.rule = rule

    @staticmethod
    def count_line(line: List[ColorFlag]) -> Tuple[int, int, int, int]:
        """
        Counts consecutive SELF stones and the full segment length in a line.
        Returns: (realLen, fullLen, start, end)
        """
        mid = len(line) // 2
        real_len, full_len = 1, 1
        real_len_inc = 1
        start, end = mid, mid

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
        Determines the pattern of a line using dynamic programming with simulation.
        """
        line = list(line_tuple)
        real_len, full_len, start, end = PatternDetector.count_line(line)

        # Base cases for Standard Gomoku
        if rule == "STANDARD":
            if real_len >= 6:
                return Pattern.OL  # Overline, not a win
            elif real_len == 5:
                return Pattern.F5  # Exact five wins
            elif full_len < 5:
                return Pattern.DEAD  # Too short for significant patterns
        # Add Renju rules if needed later

        # Recursive simulation for lower patterns
        return PatternDetector._classify_pattern(line, start, end, rule, side)

    @staticmethod
    def _classify_pattern(line: List[ColorFlag], start: int, end: int, rule: str, side: Color) -> Pattern:
        """
        Classifies the pattern by simulating moves in empty spaces.
        """
        pattern_counts = {p: 0 for p in Pattern}
        f5_indices = []

        # Simulate placing a stone in each empty space
        for i in range(start, end + 1):
            if line[i] == ColorFlag.EMPT:
                new_line = line.copy()
                new_line[i] = ColorFlag.SELF
                new_pattern = PatternDetector.get_pattern(tuple(new_line), rule, side)
                pattern_counts[new_pattern] += 1
                if new_pattern == Pattern.F5 and len(f5_indices) < 2:
                    f5_indices.append(i)

        # Pattern classification based on simulation results
        if pattern_counts[Pattern.F5] >= 2:
            return Pattern.F4  # Open four: two ways to F5
        elif pattern_counts[Pattern.F5] == 1:
            return Pattern.B4  # Blocked four: one way to F5
        elif pattern_counts[Pattern.F4] >= 2:
            return Pattern.F3S  # Special flexible three: two ways to F4
        elif pattern_counts[Pattern.F4] == 1:
            return Pattern.F3   # Flexible three: one way to F4
        elif pattern_counts[Pattern.B4] >= 1:
            return Pattern.B3   # Blocked three
        elif (pattern_counts[Pattern.F3S] + pattern_counts[Pattern.F3]) >= 4:
            return Pattern.F2B  # Strong flexible two
        elif (pattern_counts[Pattern.F3S] + pattern_counts[Pattern.F3]) >= 3:
            return Pattern.F2A  # Medium flexible two
        elif (pattern_counts[Pattern.F3S] + pattern_counts[Pattern.F3]) >= 1:
            return Pattern.F2   # Flexible two
        elif pattern_counts[Pattern.B3] >= 1:
            return Pattern.B2   # Blocked two
        elif (pattern_counts[Pattern.F2] + pattern_counts[Pattern.F2A] + pattern_counts[Pattern.F2B]) >= 1:
            return Pattern.F1   # Flexible one
        elif pattern_counts[Pattern.B2] >= 1:
            return Pattern.B1   # Blocked one
        
        return Pattern.DEAD
    
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
    
    def evaluate_affected_patterns(self, board: BitBoard, move: Tuple[int, int], side: Color, debug: bool = False) -> List[Pattern]:
        """
        Evaluates patterns only for the lines affected by the last move.
        """
        patterns = []
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]  # Vertical, Horizontal, Diagonal /, Diagonal \
        
        for dRow, dCol in directions:
            line = self.extract_line(board, move, side, dRow, dCol)
            if line:
                line_tuple = tuple(line)
                pattern = self.get_pattern(line_tuple, self.rule, side)
                if debug:
                    print(f"Direction ({dRow}, {dCol}): {''.join('S' if c == ColorFlag.SELF else 'O' if c == ColorFlag.OPPO else '.' for c in line)} -> {pattern}")
                patterns.append(pattern)
        
        return patterns
    