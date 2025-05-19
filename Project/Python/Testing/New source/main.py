import logging
from typing import List, Tuple
from src.board import Board
from src.eval import Evaluator
from src.search import MainSearchThread, ABSearcher
from src.types import Rule, Color

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def interactive():
    board = Board()
    board.new_game(Rule.STANDARD)
    evaluator = Evaluator(board.size)
    searcher = ABSearcher()

    print("Welcome to Gomoku! You are Black (X), AI is White (O).")
    print("Enter moves as 'row,col' (e.g., '7,7'). Type 'quit' to exit.")

    while True:
        for row in range(board.size):
            line = []
            for col in range(board.size):
                if board.cells[row][col] == Color.BLACK:
                    line.append('X')
                elif board.cells[row][col] == Color.WHITE:
                    line.append('O')
                else:
                    line.append('.')
            print(' '.join(line))
        print()

        if board.side_to_move() == Color.BLACK:
            move_input = input("Your move (row,col): ").strip()
            if move_input.lower() == 'quit':
                break
            try:
                row, col = map(int, move_input.split(','))
                if board.is_empty((row, col)):
                    board.move((row, col))
                else:
                    print("Invalid move: position occupied.")
                    continue
            except ValueError:
                print("Invalid input. Use 'row,col' format.")
                continue
        else:
            thread = MainSearchThread(board, evaluator, Rule.STANDARD)
            searcher.search(thread)
            if thread.best_move:
                move_str = searcher._pos_to_notation(thread.best_move)
                print(f"AI moves to {move_str} ({thread.best_move})")
                board.move(thread.best_move)
            else:
                logger.error("AI failed to find a move, should not happen")

def test_position(moves: List[Tuple[int, int]]):
    board = Board()
    board.new_game(Rule.STANDARD)
    evaluator = Evaluator(board.size)
    searcher = ABSearcher()

    for i, pos in enumerate(moves):
        board.move(pos)
        board.current_side = Color.BLACK if (i + 1) % 2 == 0 else Color.WHITE

    for row in range(board.size):
        line = []
        for col in range(board.size):
            if board.cells[row][col] == Color.BLACK:
                line.append('X')
            elif board.cells[row][col] == Color.WHITE:
                line.append('O')
            else:
                line.append('.')
        print(' '.join(line))
    print(f"Side to move: {'Black' if board.side_to_move() == Color.BLACK else 'White'}")

    thread = MainSearchThread(board, evaluator, Rule.STANDARD)
    searcher.search(thread)
    logger.info(f"Best Move: {thread.best_move}, PV: {thread.pv}")
    print(f"AI suggests: {searcher._pos_to_notation(thread.best_move) if thread.best_move else 'None'} with PV: {thread.pv}")

if __name__ == "__main__":
    print("Choose mode: 1) Interactive, 2) Test Position")
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == '1':
        interactive()
    elif choice == '2':
        test_moves = [(7, 7), (7, 6), (6, 6), (5, 5), (6, 8), (6, 5), (6, 7), (5, 7), (14, 14)]
        test_position(test_moves)
    else:
        print("Invalid choice. Exiting.")