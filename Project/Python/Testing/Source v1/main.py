import re
from src.board import Board, Pos, Color
from src.searcher import ABSearcher as Searcher, SearchOptions
from src.types import Rule, Pattern4
from src.movegen import MoveGenerator, GenType

class GomokuGame:
    def __init__(self, board_size=15, rule=Rule.FREESTYLE):
        self.board = Board(board_size=board_size, cand_range=0)
        self.board.new_game(rule)
        self.options = SearchOptions()
        self.options.time_limit = 180000  # 2 seconds per move
        self.size = board_size
        self.rule = rule

    def print_board(self):
        """Visualizes the board with coordinates (e.g., a1 to o15 for 15x15)."""
        print("   " + " ".join(chr(ord('a') + i) for i in range(self.size)))
        for y in range(self.size - 1, -1, -1):
            row = f"{y + 1:2d} "
            for x in range(self.size):
                pos = Pos(x, y)
                piece = self.board.get(pos)
                if piece == Color.BLACK:
                    row += "X "
                elif piece == Color.WHITE:
                    row += "O "
                else:
                    row += ". "
            print(row.rstrip())
        print()

    def parse_move(self, move_str: str) -> Pos:
        """Parses a move string (e.g., 'a1') into a Pos object."""
        move_str = move_str.strip().lower()
        match = re.match(r"([a-o])(\d+)", move_str)
        if not match or len(match.groups()) != 2:
            raise ValueError("Invalid move format. Use 'a1' to 'o15'.")
        col, row = match.groups()
        x = ord(col) - ord('a')
        y = int(row) - 1
        if not (0 <= x < self.size and 0 <= y < self.size):
            raise ValueError(f"Move out of bounds. Use 'a1' to '{chr(ord('a') + self.size - 1)}{self.size}'.")
        return Pos(x, y)

    def computer_vs_human(self):
        """Interactive mode: Human vs Computer."""
        print("Starting Computer vs Human game. You are White (O), Computer is Black (X).")
        print("Enter moves as 'a1' to 'o15'. Type 'quit' to exit.")
        self.print_board()

        center_pos = Pos(self.size // 2, self.size // 2)

        while True:
            if self.board.side_to_move() == Color.BLACK:
                searcher = Searcher()
                if self.board.move_count == 0:
                    move = center_pos
                    print(f"Computer plays at center: {self._pos_to_str(move)}")
                else:
                    move = searcher.search_main(self.board)
                    if not move or move == Pos.NONE:
                        print("Computer has no valid move.")
                        moves = MoveGenerator(self.board.rule).generate(self.board, GenType.ALL)
                        if moves:
                            move = moves[0].pos
                            print(f"Forcing move: {self._pos_to_str(move)}")
                        else:
                            print("No moves available. Game ends.")
                            break
                    else:
                        print(f"Computer plays: {self._pos_to_str(move)}")
                self.board.make_move(move, self.board.rule)
                self.print_board()

                if self._check_game_end():
                    break

            while True:
                try:
                    move_input = input("Your move (e.g., 'a1'): ").strip().lower()
                    if move_input == 'quit':
                        print("Game ended by user.")
                        return
                    move = self.parse_move(move_input)
                    if self.board.is_empty(move) and self.board.make_move(move, self.board.rule):
                        self.print_board()
                        break
                    else:
                        print("Illegal move. Try again.")
                except ValueError as e:
                    print(f"Error: {e}")

            if self._check_game_end():
                break

    def test_position(self):
        print("Test Mode: Enter moves to set up a position (e.g., 'a1 b2 c3').")
        print("Type 'done' when finished, 'quit' to exit.")
        self.print_board()

        while True:
            try:
                move_input = input("Enter moves or 'done': ").strip().lower()
                if move_input == 'quit':
                    print("Test mode ended by user.")
                    return
                if move_input == 'done':
                    break
                
                moves = move_input.split()
                for move_str in moves:
                    move = self.parse_move(move_str)
                    if self.board.is_empty(move) and self.board.make_move(move, self.board.rule):
                        print(f"Added move: {self._pos_to_str(move)}")
                        self.print_board()
                    else:
                        print(f"Illegal move: {move_str}. Resetting board.")
                        self.board.new_game(self.board.rule)
                        break
            except ValueError as e:
                print(f"Error: {e}")

        print("Running AI analysis...")
        self.options.time_limit = 5000
        searcher = Searcher()
        move = searcher.search_main(self.board)
        if move and move != Pos.NONE:
            print(f"AI suggests: {self._pos_to_str(move)}")
            self.board.make_move(move, self.board.rule)
            print("Board after AI move:")
            self.print_board()
        else:
            print("No move found.")

    def _check_game_end(self):
        """Checks if the game has ended and prints result."""
        # Debug: Print p4_count for analysis
        black_fives = self.board.p4_count(Color.BLACK, Pattern4.A_FIVE)
        white_fives = self.board.p4_count(Color.WHITE, Pattern4.A_FIVE)
        print(f"Debug: Black A_FIVE count = {black_fives}, White A_FIVE count = {white_fives}")

        # Check for five in a row (win condition)
        if black_fives > 0:
            print("Winner: Black")
            return True
        if white_fives > 0:
            print("Winner: White")
            return True
        # Check for draw (board full)
        if self.board.move_count >= self.board.board_cell_count:
            print("Game ended in a draw.")
            return True
        return False

    def _pos_to_str(self, pos: Pos):
        """Converts Pos to string (e.g., Pos(0,0) -> 'a1')."""
        return f"{chr(ord('a') + pos.x)}{pos.y + 1}"

def main():
    """Main entry point for the Gomoku engine."""
    print("Gomoku/Renju Engine")
    while True:
        choice = input("Choose mode: (1) Computer vs Human, (2) Test Position, (q) Quit: ").strip().lower()
        game = GomokuGame(board_size=15, rule=Rule.FREESTYLE)
        if choice == '1':
            game.computer_vs_human()
        elif choice == '2':
            game.test_position()
        elif choice == 'q':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()