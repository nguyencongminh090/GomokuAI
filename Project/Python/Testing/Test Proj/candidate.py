# candidate.py

from typing import List, Tuple
from utils import checkValid
from interfaces import BitBoardABC, CandidateABC 

class Candidate(CandidateABC):
    def __init__(self, mode=0, size=15):
        self.mode = mode
        self.size = size

    def expand(self, boardState: BitBoardABC) -> List[Tuple[int, int]]:
        candidate = []
        marked_positions = [] 

        for row in range(self.size):
            for col in range(self.size):
                state = boardState.get_state((row, col))
                if state in (0b01, 0b10): 
                    if self.mode == 0:
                        self.__squareLine(boardState, row, col, 3, 4, marked_positions)
                    elif self.mode == 1:
                        self.__circle34(boardState, row, col, marked_positions)
                    elif self.mode == 2:
                        self.__fullBoard(boardState, marked_positions)
                    else:
                        print('Not supported mode:', self.mode)

        for row in range(self.size):
            for col in range(self.size):
                if boardState.get_state((row, col)) == 0b11:
                    candidate.append((row, col))
                    
        for pos in marked_positions:
            boardState.reset_pos(pos)

        return candidate

    def __squareLine(self, boardState: BitBoardABC, x: int, y: int, sq: int, ln: int, marked_positions: List[Tuple[int, int]]):
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]        
        for k in range(1, ln + 1):
            for i, j in directions:
                coord = (x + i * k, y + j * k)
                if checkValid(boardState.size, coord) and boardState.get_state(coord) == 0:
                    boardState.add_move(coord, 3)
                    marked_positions.append(coord)
                    
        for i in range(1, sq + 1):
            for j in range(1, sq + 1):
                coords = [
                    (x + i, y + j),
                    (x + i, y - j),
                    (x - i, y + j),
                    (x - i, y - j)
                ]
                for coord in coords:
                    if checkValid(boardState.size, coord) and boardState.get_state(coord) == 0:
                        boardState.add_move(coord, 3)
                        marked_positions.append(coord)

    def __circle34(self, boardState: BitBoardABC, x: int, y: int, marked_positions: List[Tuple[int, int]]):
        cr34 = 34 ** 0.5
        for row in range(-int(cr34), int(cr34) + 1):
            for col in range(-int(cr34), int(cr34) + 1):
                if (row ** 2 + col ** 2) ** 0.5 <= cr34:
                    coord = (x + row, y + col)
                    if checkValid(boardState.size, coord) and boardState.get_state(coord) == 0:
                        boardState.add_move(coord, 3)
                        marked_positions.append(coord)

    def __fullBoard(self, boardState: BitBoardABC, marked_positions: List[Tuple[int, int]]):
        for row in range(boardState.size):
            for col in range(boardState.size):
                if boardState.get_state((row, col)) == 0:
                    boardState.add_move((row, col), 3)
                    marked_positions.append((row, col))
