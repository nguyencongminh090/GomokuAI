import random
from abc import ABC, abstractmethod


def checkValid(size: int, move: list[int, int]) -> bool:
        """
        Return True if move in range
        """
        return ((size - 1) >= move[0] >= 0) and ((size - 1) >= move[1] >= 0)


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
    def expand(self, board: BitBoardABC) -> list[int, int]:
        ...



class Candidate:  
    def __init__(self, mode=0, size=15):
        """
        Mode 0 = SQUARE_3_LINE_4
        Mode 1 = CIRCLE_SQRT_34
        Mode 2 = FULL_BOARD
        """
        self.__mode = mode
        self.__size = size

    def expand(self, boardState: BitBoardABC) -> list[int, int]:        
        candidate = []
        for row in range(self.__size):
            for col in range(self.__size):
                state = boardState.getState((row, col))
                if state in (0b01, 0b10):
                    match self.__mode:
                        case 0:
                            self.__squareLine(boardState, row, col, 3, 4)
                        case 1:
                            self.__circle34(boardState, row, col)
                        case 2:
                            self.__fullBoard(boardState)
                        case _:
                            print('Not supported')

        for row in range(self.__size):
            for col in range(self.__size):
                if boardState.getState((row, col)) == 0b11:
                    candidate.append((row, col))

        return candidate
            
    def __squareLine(self, boardState: BitBoardABC, x: int, y: int, sq: int, ln: int):
        """
        Example Output: [(1,1), (1,2), (3,1), ...]
        NOTE: return Candidate range for 1 move only
        """
        direction = ((1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1))
        for k in range(1, ln + 1):
            for i, j in direction:
                coord = (x + i * k, y + j * k)
                if boardState.getState(coord) == 0:
                    boardState.addMove(coord, 3)

        for i in range(1, sq + 1):
            for j in range(1, sq + 1):
                coords = [(x + i, y + j), (x + i, y - j), (x - i, y + j), (x - i, y - j)]
                for coord in coords:                    
                    if boardState.getState(coord) == 0:
                        boardState.addMove(coord, 3)        

    def __circle34(self, boardState: BitBoardABC, x, y):
        def distance(pointA: tuple[int, int], pointB: tuple[int, int]) -> int:
            return ((pointB[0] - pointA[0]) ** 2 + (pointB[1] - pointA[1]) ** 2) ** 0.5
        cr34 = 34 ** 0.5
        for row in range(0, 6):
            for col in range(0, 6):
                coords = [(x + row, y + col), (x + row, y - col), (x - row, y + col), (x - row, y - col)]
                for coord in coords:
                    if boardState.getState(coord) == 0 and distance(coord, (x, y)) <= cr34:
                        boardState.addMove(coord, 3)                 

    def __fullBoard(self, boardState: BitBoardABC):
        for row in range(self.__size):
            for col in range(self.__size):
                if not boardState.getState((row, col)):
                    boardState.addMove((row, col), 3)
                

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
        if not checkValid(self.__size, move):
            return -1

        pos       = move[0] * self.__size + move[1]
        mask      = 0b11 << (pos * 2)
        stateBits = (self.bitBoard & mask) >> (pos * 2)

        if stateBits == 0b01:
            return 1
        elif stateBits == 0b10:
            return 2
        elif stateBits == 0b11:
            return 3
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
        3 - Repr for M (Marked)
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
                    match state:
                        case 1:
                            curLine.append('X')
                        case 2:
                            curLine.append('O')
                        case 3: 
                            # Debug Candidate
                            curLine.append('*')
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
        return candidate.expand(self)

    def debugDisplayBitBoard(self):
        print("BitBoard (Binary Representation):")
        for i in range(self.__size):
            row_bits = ""
            for j in range(self.__size):
                pos = i * self.__size + j
                mask = 0b11 << (pos * 2)
                state_bits = (self.bitBoard & mask) >> (pos * 2)
                row_bits += f"{state_bits:02b} "
            print(row_bits)
        print()


class Evaluator:
    # Read document
    ...

class TreeNode:
    def __init__(self, root: bool, boardState: BitBoardABC, depth: int, score: int, hash: int, priority: int=0):
        self.root:       bool           = root
        self.boardState: BitBoardABC    = boardState
        self.depth:      int            = depth
        self.score:      int            = score
        self.hash :      int            = hash
        self.priority:   int            = priority
        self.child:      list[TreeNode] = []
  
class Search:
    # Transposition Table: Depth, Score, Priority
    # TODO: 
    # + Quiescence Search: VCF, VCT {Threat Search}
    # + Update Tree
    # [+] Use Tree to compare variants (children)
    # + TTEntry, Transposition Table
    TRANSPOSITION_TABLE = {}
    def __init__(self):
        self.__evaluator = Evaluator()

    def alphabeta(self, depth, alpha, beta, maximizePlayer):
        ...

    def PVS_Search(self, depth, score):
        ...

    def VCF_Search(self, depth, score):
        ...

    def VCT_Search(self, depth, score):
        ...

    def attackSearch(self, depth, pattern, score):
        ...

    def defendSearch(self, depth, score):
        ...


# TEST
bitBoard = BitBoard(15)
candidate = Candidate(1, 15)
bitBoard.addMove((7, 2), 1)
bitBoard.addMove((12,4), 1)
bitBoard.addMove((4, 7), 2)
print(bitBoard.getPossibleMoves(candidate))
print(bitBoard.isWin(1))
print(bitBoard.view()) 