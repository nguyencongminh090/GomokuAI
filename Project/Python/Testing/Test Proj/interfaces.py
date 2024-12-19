# interfaces.py

from abc import ABC, abstractmethod
from typing import List, Tuple

class BitBoardABC(ABC):
    @abstractmethod
    def hash(self) -> int:
        pass

    @abstractmethod
    def add_move(self, move: Tuple[int, int], player: int):
        pass

    @abstractmethod
    def get_state(self, move: Tuple[int, int]) -> int:
        pass

    @abstractmethod
    def is_win(self, player: int) -> bool:
        pass

    def reset_pos(self, move: Tuple[int, int]) -> bool:
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        pass

    @abstractmethod
    def get_possible_moves(self, candidate: 'CandidateABC') -> List[Tuple[int, int]]:
        pass

class CandidateABC(ABC):
    @abstractmethod
    def expand(self, board: BitBoardABC) -> List[Tuple[int, int]]:
        pass
