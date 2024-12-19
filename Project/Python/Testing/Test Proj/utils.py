# utils.py

from typing import Tuple

def checkValid(size: int, move: Tuple[int, int]) -> bool:
    """Check if the move is within the board boundaries."""
    return 0 <= move[0] < size and 0 <= move[1] < size
