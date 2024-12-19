# general_test.py

from board import BitBoard
from pattern import PatternDetector
from enums import Color, Pattern
from search import TreeNode, Search
from candidate import Candidate

from evaluator import Evaluator

def main():
    # Initialize BitBoard and PatternDetector without a fixed side
    board_size = 15
    bitBoard = BitBoard(board_size)
    patternDetector = PatternDetector(rule='STANDARD')  # Removed 'side' parameter

    # Define the sequence of moves (position, color)
    moves = [
        ((7, 7), Color.BLACK.value),   # Black
        ((7, 6), Color.WHITE.value),   # White
        ((6, 6), Color.BLACK.value),   # Black
        ((5, 5), Color.WHITE.value),   # White
        ((6, 8), Color.BLACK.value),   # Black
        ((6, 5), Color.WHITE.value),   # White
        ((6, 7), Color.BLACK.value),   # Black
        ((5, 7), Color.WHITE.value),   # White
        ((6, 9), Color.BLACK.value),
        # Uncomment the next line to add another move
        # ((5, 9), Color.BLACK.value),  # Black
        # ((8, 8), Color.BLACK.value),
        # ((5, 6), Color.WHITE.value),
    ]

    # Apply the moves to the board
    for move, color in moves:
        success = bitBoard.add_move(move, color)
        if not success:
            print(f"Failed to add move at {move} for {Color(color).name}.")

    # Display the current state of the board
    print("Current Board:")
    print(bitBoard.view())
    print("\nBinary Representation:")
    # bitBoard.debug_display_bitboard()

    # Determine the last move and its side
    if moves:
        last_move, last_color = moves[-1]
        last_side = Color(last_color)
        print(f"\nEvaluating patterns for the last move: {last_move} by {last_side.name}")
    else:
        print("\nNo moves made on the board.")
        return

    # Evaluate patterns based on the last move and its side
    patterns = patternDetector.evaluate_patterns(board=bitBoard, side=last_side)
    for p in patterns:
        print(f"- {p}")
    # print('Score:', score)

    # Initialize Search without a fixed player side
    searchEngine = Search(bitBoard.get_current_side())

    # Create the root node for the search tree
    root_hash = bitBoard.hash()
    rootNode = TreeNode(root=True, boardState=bitBoard, depth=1, score=0, hashVal=root_hash)

    # Determine the current side to move
    current_side = bitBoard.get_current_side()
    print(f"\nCurrent side to move: {current_side.name}")

    # Perform Alpha-Beta Search
    # bestScore = searchEngine.alphabeta(
    #     node=rootNode,
    #     depth=3,
    #     alpha=-float('inf'),
    #     beta=float('inf'),
    #     current_side=current_side
    # )
    bestScore = searchEngine.find_best_move(
        node=rootNode,
        depth=3,
        current_side=current_side
    )
    print("\nBest Score from Alpha-Beta Search:", bestScore)

    bitBoard.add_move(bestScore[1], current_side.value)
    print(bitBoard.view())

    # Check for a win condition
    if bitBoard.is_win(Color.BLACK.value):
        print("\nBlack wins!")
    elif bitBoard.is_win(Color.WHITE.value):
        print("\nWhite wins!")
    else:
        print("\nNo win detected.")

if __name__ == "__main__":
    main()
