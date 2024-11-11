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
        if self.getState(move) is not None:
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
    
    def is_win(self, player: int) -> bool:
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for row in range(self.size):
            for col in range(self.size):
                if self.getState((row, col)) == player:
                    for dRow, dCol in directions:
                        count = 1
                        r, c = row + dRow, col + dCol
                        while 0 <= r < self.size and 0 <= c < self.size and self.get_state((r, c)) == player:
                            count += 1
                            r += dRow
                            c += dCol
                        if count >= 5:
                            print(f"Player '{player}' has won!")
                            return True
        print(f"Player '{player}' has not won.")
        return False

    

bitBoard = BitBoard(15)
bitBoard.addMove((7, 12), 1)
print(bitBoard.hash())
bitBoard.addMove((8, 12), 1)
print(bitBoard.hash())
print(bitBoard.view())