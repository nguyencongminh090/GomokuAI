# test_pattern_detection.py

import sys
from enum import Enum, auto
from typing import List, Tuple
from functools import lru_cache

# Define necessary enums
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

# PatternDetector class with debug statements
class PatternDetector:
    def __init__(self, rule: str, side: Color):
        """
        Initializes the PatternDetector.

        Args:
            rule (str): The game rule ('FREESTYLE', 'STANDARD', 'RENJU').
            side (Color): The current player's color.
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

        print(f"\nCounting line: {[line.name for line in line]}")
        print(f"Mid index: {mid}")

        # Left side
        for i in range(mid - 1, -1, -1):
            if line[i] == ColorFlag.SELF:
                real_len += real_len_inc
                print(f"Left side - Index {i}: SELF detected. real_len increased to {real_len}")
            elif line[i] == ColorFlag.OPPO:
                print(f"Left side - Index {i}: OPPO detected. Stopping left count.")
                break
            else:
                real_len_inc = 0
                print(f"Left side - Index {i}: EMPT detected. real_len_inc set to {real_len_inc}")
            full_len += 1
            start = i

        # Right side
        real_len_inc = 1
        for i in range(mid + 1, len(line)):
            if line[i] == ColorFlag.SELF:
                real_len += real_len_inc
                print(f"Right side - Index {i}: SELF detected. real_len increased to {real_len}")
            elif line[i] == ColorFlag.OPPO:
                print(f"Right side - Index {i}: OPPO detected. Stopping right count.")
                break
            else:
                real_len_inc = 0
                print(f"Right side - Index {i}: EMPT detected. real_len_inc set to {real_len_inc}")
            full_len += 1
            end = i

        print(f"Final counts: real_len={real_len}, full_len={full_len}, start={start}, end={end}")
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
        print(f"\nAnalyzing line: {[line_name.name for line_name in line]}")
        real_len, full_len, start, end = PatternDetector.count_line(line)
        pattern = Pattern.DEAD

        # Check for Overline (OL)
        if rule in ['STANDARD', 'RENJU'] and real_len >= 6:
            print("Pattern detected: Overline (OL)")
            return Pattern.OL
        elif real_len >= 5:
            print("Pattern detected: Five in a Row (F5)")
            return Pattern.F5
        elif full_len < 5:
            print("Pattern detected: Dead (DEAD)")
            return Pattern.DEAD
        else:
            # Initialize pattern counts
            pattern_counts = {p: 0 for p in Pattern}
            f5_indices = []

            print("Simulating placing SELF on empty cells to detect patterns:")
            # Iterate through the line to find empty positions
            for i in range(start, end + 1):
                if line[i] == ColorFlag.EMPT:
                    # Simulate placing a stone at position i
                    new_line = line.copy()
                    new_line[i] = ColorFlag.SELF
                    print(f"  Simulating move at index {i}: {[line_name.name for line_name in new_line]}")
                    new_pattern = PatternDetector.get_pattern(tuple(new_line), rule, side)
                    pattern_counts[new_pattern] += 1

                    if new_pattern == Pattern.F5 and len(f5_indices) < 2:
                        f5_indices.append(i)

            print(f"Pattern counts after simulation: {pattern_counts}")

            # Determine the pattern based on pattern counts
            if pattern_counts[Pattern.F5] >= 2:
                pattern = Pattern.F4
                print("Multiple F5 detected. Tentatively Pattern F4")
                if rule == 'RENJU' and side == Color.BLACK:
                    # Check if the two F5 patterns are within 5 positions
                    if len(f5_indices) >= 2 and (f5_indices[1] - f5_indices[0] < 5):
                        pattern = Pattern.OL
                        print("RENJU rule and two F5 patterns are close. Pattern set to Overline (OL)")
            elif pattern_counts[Pattern.F5] == 1:
                pattern = Pattern.B4
                print("Single F5 detected. Pattern set to B4")
            elif pattern_counts[Pattern.F4] >= 2:
                pattern = Pattern.F3S
                print("Multiple F4 detected. Pattern set to F3S")
            elif pattern_counts[Pattern.F4] == 1:
                pattern = Pattern.F3
                print("Single F4 detected. Pattern set to F3")
            elif pattern_counts[Pattern.B4] >= 1:
                pattern = Pattern.B3
                print("At least one B4 detected. Pattern set to B3")
            elif (pattern_counts[Pattern.F3S] + pattern_counts[Pattern.F3]) >= 4:
                pattern = Pattern.F2B
                print("Combined F3S and F3 counts >=4. Pattern set to F2B")
            elif (pattern_counts[Pattern.F3S] + pattern_counts[Pattern.F3]) >= 3:
                pattern = Pattern.F2A
                print("Combined F3S and F3 counts >=3. Pattern set to F2A")
            elif (pattern_counts[Pattern.F3S] + pattern_counts[Pattern.F3]) >= 1:
                pattern = Pattern.F2
                print("At least one F3S or F3 detected. Pattern set to F2")
            elif pattern_counts[Pattern.B3] >= 1:
                pattern = Pattern.B2
                print("At least one B3 detected. Pattern set to B2")
            elif (pattern_counts[Pattern.F2] + pattern_counts[Pattern.F2A] + pattern_counts[Pattern.F2B]) >= 1:
                pattern = Pattern.F1
                print("At least one F2, F2A, or F2B detected. Pattern set to F1")
            elif pattern_counts[Pattern.B2] >= 1:
                pattern = Pattern.B1
                print("At least one B2 detected. Pattern set to B1")
            else:
                pattern = Pattern.DEAD
                print("No significant patterns detected. Pattern remains DEAD")

            print(f"Final detected pattern: {pattern.name}")
            return pattern


    # Test function
def test_get_pattern():
    # Define the specific line: _ o _ x x _ x _ o
    # Mapping symbols to ColorFlag
    symbol_to_colorflag = {
        '_': ColorFlag.EMPT,
        'x': ColorFlag.SELF,
        'o': ColorFlag.OPPO
    }

    input_line_symbols = ['x', '_', 'x', '_', 'x', '_', 'x', '_', '_']
    input_line = tuple(symbol_to_colorflag[symbol] for symbol in input_line_symbols)

    print(f"Input Line Symbols: {' '.join(input_line_symbols)}")
    print(f"Input Line Flags: {input_line}\n")

    # Initialize PatternDetector with 'STANDARD' rule and BLACK side
    detector = PatternDetector(rule='STANDARD', side=Color.BLACK)

    # Invoke get_pattern with debug statements
    detected_pattern = detector.get_pattern(line_tuple=input_line, rule='STANDARD', side=Color.BLACK)

    print(f"\nDetected Pattern: {detected_pattern.name}")

# Execute the test
if __name__ == "__main__":
    test_get_pattern()
