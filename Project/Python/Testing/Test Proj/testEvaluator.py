# test_pattern.py

import unittest
from enums import Color, ColorFlag, Pattern
from board import BitBoard
from pattern import PatternDetector
from typing import List, Tuple

class TestPatternDetector(unittest.TestCase):
    def setUp(self):
        """
        Initialize the board and PatternDetector before each test.
        """
        self.board = BitBoard(size=15)
        self.pattern_detector = PatternDetector(rule='STANDARD')
        self.ai_color = Color.BLACK
        self.opponent_color = Color.WHITE

    def test_pattern_xxx_o___o(self):
        """
        Test the pattern:
        x x x _ o _ _ _ o
        This represents a horizontal line with:
        - AI (Black) having three in a row.
        - Opponent (White) having stones interrupting potential connections.
        """
        # Define the row where the pattern will be placed
        row = 7  # Center row for visualization
        # Define the starting column
        start_col = 4

        # Place AI's stones: x x x
        # self.board.add_move((row, start_col), self.ai_color.value)
        # self.board.add_move((row, start_col + 1), self.ai_color.value)
        # self.board.add_move((row, start_col + 2), self.ai_color.value)
        
        # Place Opponent's stones: o ... o
        self.board.add_move((row, start_col + 4), self.opponent_color.value)
        self.board.add_move((row, start_col + 8), self.opponent_color.value)

        pattern = PatternDetector('STANDARD')._evaluate_patterns(self.board, (row, start_col + 4), self.opponent_color)

        # Display the board
        print("Board Setup:")
        print(self.board.view())

        # The pattern to test is: x x x _ o _ _ _ o
        # Since PatternDetector evaluates lines passing through a specific move,
        # we'll evaluate patterns for each of AI's stones

        

if __name__ == '__main__':
    unittest.main()
