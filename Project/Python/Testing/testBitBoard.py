import random

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
            return None
    
    def hash(self) -> int:
        """
        Purpose: Generate Hash value for a position.
        """
        hashValue = 0
        for row in range(self.size):
            for col in range(self.size):
                state = self.getState((row, col))
                if state:
                    hashValue ^= self.__zobristTable[(row, col, state)]
        return hashValue

    def addMove(self, move: tuple[int, int], player):
        """
        1 - Repr for X (Black player)
        2 - Repr for O (White player)
        """
        if self.getState(move) is not None:
            return 
        if player == 1:
            pos = move[0] * self.__size + move[1]
            self.bitBoard = 0b01 << (pos * 2)

    

bitBoard = BitBoard(15)