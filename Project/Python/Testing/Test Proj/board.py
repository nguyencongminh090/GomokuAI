# board.py

import random
from typing import List, Tuple, Dict, Optional
from utils import checkValid
from enums import Color, ColorFlag
from interfaces import CandidateABC, BitBoardABC


class BitBoard(BitBoardABC):
    def __init__(self, size: int = 15):
        """
        Initializes the BitBoard with a given size.

        Args:
            size (int): The size of the board (size x size). Defaults to 15.
        """
        self._size = size  # Internal attribute for size
        self.zobrist_table: Dict[Tuple[int, int, int], int] = self._generate_zobrist_table()
        self.bit_board = 0  # Represents the board using bits.
        self.last_move: Optional[Tuple[int, int]] = None  # Last move made
        self.move_count = 0  # Tracks the number of moves made.

    @property
    def size(self) -> int:
        """
        Returns the size of the board.

        Returns:
            int: The size of the board.
        """
        return self._size

    def _generate_zobrist_table(self) -> Dict[Tuple[int, int, int], int]:
        """
        Generates the Zobrist hashing table.

        Returns:
            Dict[Tuple[int, int, int], int]: A dictionary mapping (row, col, player) to random 64-bit integers.
        """
        table = {}
        for row in range(self.size):
            for col in range(self.size):
                for player in [1, 2]:  # 1 for BLACK, 2 for WHITE
                    table[(row, col, player)] = random.getrandbits(64)
        return table

    def get_state(self, move: Tuple[int, int]) -> int:
        """
        Retrieves the state of a specific cell.

        Args:
            move (Tuple[int, int]): The (row, col) position on the board.

        Returns:
            int: 
                0 if empty,
                1 if occupied by BLACK,
                2 if occupied by WHITE,
                3 if marked.
        """
        if not checkValid(self.size, move):
            return -1  # Invalid move
        row, col = move
        pos = row * self.size + col
        mask = 0b11 << (pos * 2)
        state_bits = (self.bit_board & mask) >> (pos * 2)
        if state_bits == 0b01:
            return 1  # BLACK
        elif state_bits == 0b10:
            return 2  # WHITE
        elif state_bits == 0b11:
            return 3  # MARKED
        else:
            return 0  # Empty

    def hash(self) -> int:
        """
        Computes the Zobrist hash of the current board state.

        Returns:
            int: The computed hash value.
        """
        hash_value = 0
        for row in range(self.size):
            for col in range(self.size):
                state = self.get_state((row, col))
                if state in [1, 2]:
                    hash_value ^= self.zobrist_table[(row, col, state)]
        return hash_value

    def add_move(self, move: Tuple[int, int], player: int) -> bool:
        """
        Adds a move to the board for a given player.

        Args:
            move (Tuple[int, int]): The (row, col) position to place the move.
            player (int): The player making the move (1 for BLACK, 2 for WHITE).

        Returns:
            bool: True if the move was successfully added, False otherwise.
        """
        if self.get_state(move) != 0:
            return False  # Cell is not empty
        row, col = move
        pos = row * self.size + col
        self.bit_board |= player << (pos * 2)
        self.last_move = move
        self.move_count += 1
        return True

    def reset_pos(self, move: Tuple[int, int]) -> bool:
        """
        Resets (removes) a move from the board.

        Args:
            move (Tuple[int, int]): The (row, col) position to reset.

        Returns:
            bool: True if the move was successfully reset, False otherwise.
        """
        if not checkValid(self.size, move):
            return False  # Invalid move
        row, col = move
        pos = row * self.size + col
        mask = 0b11 << (pos * 2)
        current_state = (self.bit_board & mask) >> (pos * 2)
        if current_state == 0:
            return False  # Cell is already empty
        self.bit_board &= ~mask
        self.last_move = None
        self.move_count = max(0, self.move_count - 1)
        return True

    def view(self) -> str:
        """
        Returns a string representation of the board.

        Returns:
            str: The board as a string with rows separated by newlines.
        """
        lines = []
        for row in range(self.size):
            current_line = []
            for col in range(self.size):
                state = self.get_state((row, col))
                if state == 1:
                    current_line.append('X')  # BLACK
                elif state == 2:
                    current_line.append('O')  # WHITE
                elif state == 3:
                    current_line.append('*')  # Blocked or special state
                else:
                    current_line.append('.')  # Empty
            lines.append('  '.join(current_line))
        return '\n'.join(lines)

    def is_win(self, player: int) -> bool:
        """
        Checks if the specified player has won the game.

        Args:
            player (int): The player to check (1 for BLACK, 2 for WHITE).

        Returns:
            bool: True if the player has won, False otherwise.
        """
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for row in range(self.size):
            for col in range(self.size):
                if self.get_state((row, col)) != player:
                    continue
                for d_row, d_col in directions:
                    count = 1
                    r, c = row + d_row, col + d_col
                    while 0 <= r < self.size and 0 <= c < self.size and self.get_state((r, c)) == player:
                        count += 1
                        r += d_row
                        c += d_col
                    if count >= 5:
                        # Optionally check for overlines based on game rules
                        before_move = (row - d_row, col - d_col)
                        after_move = (row + d_row * 5, col + d_col * 5)
                        before_state = self.get_state(before_move) if checkValid(self.size, before_move) else -1
                        after_state = self.get_state(after_move) if checkValid(self.size, after_move) else -1
                        if before_state != player and after_state != player:
                            return True
        return False

    def get_possible_moves(self, candidate: CandidateABC) -> List[Tuple[int, int]]:
        """
        Retrieves a list of possible moves based on the candidate strategy.

        Args:
            candidate (CandidateABC): The candidate strategy for generating moves.

        Returns:
            List[Tuple[int, int]]: A list of possible (row, col) moves.
        """
        return candidate.expand(self)

    def debug_display_bitboard(self):
        """
        Prints the binary representation of the bitboard for debugging purposes.
        """
        print("BitBoard (Binary Representation):")
        for row in range(self.size):
            row_bits = ""
            for col in range(self.size):
                pos = row * self.size + col
                mask = 0b11 << (pos * 2)
                state_bits = (self.bit_board & mask) >> (pos * 2)
                row_bits += f"{state_bits:02b} "
            print(row_bits)
        print()

    def copy(self) -> BitBoardABC:
        """
        Creates a deep copy of the current BitBoard.

        Returns:
            BitBoardABC: A new BitBoard instance with the same state.
        """
        new_board = BitBoard(self.size)
        new_board.bit_board = self.bit_board
        new_board.last_move = self.last_move
        new_board.move_count = self.move_count
        return new_board

    def get_current_side(self) -> Color:
        """
        Determines the current side to move based on the move count.

        Returns:
            Color: Color.BLACK if it's BLACK's turn, Color.WHITE otherwise.
        """
        return Color.BLACK if self.move_count % 2 == 0 else Color.WHITE
