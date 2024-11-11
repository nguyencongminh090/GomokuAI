import random
from abc import ABC, abstractmethod


class BitBoardABC(ABC):
    @abstractmethod  
    def hash(self):
        ...

    @abstractmethod  
    def addMove(self, move: tuple[int, int], player: int):
        ...

    @abstractmethod  
    def getState(self, move: tuple[int, int]) -> int:
        ...

    @abstractmethod  
    def isWin(self, move: tuple[int, int]) -> bool:
        ...


class CandidateABC(ABC):
    @abstractmethod
    def expand(self, board: BitBoardABC):
        ...



class Candidate:
    def __init__(self, mode=0):
        """
        Mode 0 = SQUARE_3_LINE_4
        Mode 1 = CIRCLE_SQRT_34
        Mode 2 = FULL_BOARD
        """
        self.__mode = mode

    def expand(self, boardState: BitBoardABC) -> list[int, int]:
        match self.__mode:
            case 0:
                # Get Available Moves
                # Search for Candidate Range
                # Append to list
                # Remove duplicate or check it during append
                return self.__square3Line4(boardState)
            
    def __square3Line4(self, boardState: BitBoardABC) -> list[int, int]:
        """
        Example Output: [(1,1), (1,2), (3,1), ...]
        NOTE: return Candidate range for 1 move only
        """
        ...

    def __circle34(self, boardState: BitBoardABC) -> list[int, int]:
        ...

    def __fullBoard(self, boardState: BitBoardABC) -> list[int, int]:
        ...


class BitBoard:

    def __init__(self, size=15):
        self.__size         = size        
        self.__zobristTable = self.__generateZobristTable()
        self.bitBoard       = 0

    def __generateZobristTable(self) -> dict:
        """
        1 - Repr for X (Black player)
        2 - Repr for O (White player)
        """
        table = {}
        for row in range(self.__size):
            for col in range(self.__size):
                table[(row, col, 1)] = random.getrandbits(2**6)
                table[(row, col, 2)] = random.getrandbits(2**6)
        return table
    
    def getState(self, move: tuple[int, int]) -> int:
        pos       = move[0] * self.__size + move[1]
        mask      = 0b11 << (pos * 2)
        stateBits = (self.bitBoard & mask) >> (pos * 2)

        if stateBits == 0b01:
            return 1
        elif stateBits == 0b10:
            return 2
        else:
            return 0
    
    def hash(self) -> int:
        """
        Purpose: Generate Hash value for a position.
        """
        hashValue = 0
        for row in range(self.__size):
            for col in range(self.__size):
                state = self.getState((row, col))
                if state:
                    hashValue ^= self.__zobristTable[(row, col, state)]
        return hex(hashValue)[2:].upper()

    def addMove(self, move: tuple[int, int], player):
        """
        1 - Repr for X (Black player)
        2 - Repr for O (White player)
        """
        if self.getState(move):
            return 
        pos = move[0] * self.__size + move[1]
        self.bitBoard |= player << (pos * 2)

    def view(self):
        line = []
        for x in range(self.__size):
            curLine = []
            for y in range(self.__size):
                state = self.getState((x, y))
                if state:
                    curLine.append('X' if state == 1 else "O")
                else:
                    curLine.append('.')
            line.append('  '.join(curLine))
        return '\n'.join(line)
    
    def isWin(self, player: int) -> bool:
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for row in range(self.__size):
            for col in range(self.__size):
                if self.getState((row, col)) == player:
                    for dRow, dCol in directions:
                        count = 1
                        r, c = row + dRow, col + dCol
                        while 0 <= r < self.__size and 0 <= c < self.__size and self.getState((r, c)) == player:
                            count += 1
                            r += dRow
                            c += dCol
                        if count >= 5 and (self.getState((row - dRow, col - dCol)) + 
                                           self.getState((row + 5 * dRow, col + dCol * 5)) == 0):
                            return True
        return False

    def getPossibleMoves(self, candidate: CandidateABC):
        ...

    def __expand(self, candidate: CandidateABC):
        ...


bitBoard = BitBoard(15)
bitBoard.addMove((1, 1), 1)
print(bitBoard.hash())
bitBoard.addMove((1, 2), 1)
print(bitBoard.hash())
bitBoard.addMove((1, 3), 1)
print(bitBoard.hash())
bitBoard.addMove((1, 4), 1)
print(bitBoard.hash())
bitBoard.addMove((1, 5), 1)
bitBoard.addMove((2, 6), 1)
bitBoard.addMove((3, 6), 1)
bitBoard.addMove((5, 6), 1)
bitBoard.addMove((4, 6), 1)
bitBoard.addMove((1, 6), 1)
print(bitBoard.isWin(1))
print(bitBoard.view())