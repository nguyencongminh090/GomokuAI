import random

class BitBoard:
    def __init__(self, size):
        self.size = size
        self.bitBoard = 0  # 1-BitBoard dùng để lưu trữ trạng thái của toàn bộ bảng
        self.zobristTable = self.__generateZobristTable()

    def __generateZobristTable(self):
        # Tạo bảng băm Zobrist với các giá trị ngẫu nhiên cho từng vị trí và trạng thái
        table = {}
        for row in range(self.size):
            for col in range(self.size):
                table[(row, col, 'X')] = random.getrandbits(64)
                table[(row, col, 'O')] = random.getrandbits(64)
        return table

    def hash(self) -> int:
        hash_value = 0
        for row in range(self.size):
            for col in range(self.size):
                state = self.get_state((row, col))
                if state == 'X':
                    hash_value ^= self.zobristTable[(row, col, 'X')]
                elif state == 'O':
                    hash_value ^= self.zobristTable[(row, col, 'O')]
        return hash_value

    def add_move(self, move: (int, int), player: str):
        # Kiểm tra người chơi và gán bit tương ứng cho người chơi đó
        if player == 'X':
            player_bit = 0b01
        elif player == 'O':
            player_bit = 0b10
        else:
            raise ValueError("Player phải là 'X' hoặc 'O'")
        
        pos = move[0] * self.size + move[1]
        mask = 0b11 << (pos * 2)  # Mặt nạ bit cho vị trí ô
        # Xóa trạng thái cũ và đặt bit mới cho người chơi
        self.bitBoard = (self.bitBoard & ~mask) | (player_bit << (pos * 2))

    def get_state(self, move: (int, int)) -> str:
        # Trả về trạng thái tại ô (X, O, hoặc trống)
        pos = move[0] * self.size + move[1]
        mask = 0b11 << (pos * 2)
        state_bits = (self.bitBoard & mask) >> (pos * 2)
        
        if state_bits == 0b01:
            return 'X'
        elif state_bits == 0b10:
            return 'O'
        return None  # Trả về None nếu ô trống

    def clear(self):
        self.bitBoard = 0

    def view(self):
        line = []
        for x in range(self.size):
            curLine = []
            for y in range(self.size):
                state = self.get_state((x, y))
                if state:
                    curLine.append(state)
                else:
                    curLine.append('.')
            line.append('  '.join(curLine))
        return '\n'.join(line)

    def is_win(self, player: str) -> bool:
        directions = [
            (1, 0),  # Dọc
            (0, 1),  # Ngang
            (1, 1),  # Chéo chính
            (1, -1)  # Chéo phụ
        ]
        for row in range(self.size):
            for col in range(self.size):
                if self.get_state((row, col)) == player:
                    for dRow, dCol in directions:
                        count = 1
                        r, c = row + dRow, col + dCol
                        while 0 <= r < self.size and 0 <= c < self.size and self.get_state((r, c)) == player:
                            count += 1
                            r += dRow
                            c += dCol
                        if count >= 5:
                            return True
        return False

class Board:
    def __init__(self, size=15):
        self.size = size
        self.bitboard = BitBoard(self.size)

    def add_move(self, move: (int, int), player: str):
        if not self.available(move):
            raise ValueError("Ô đã được đánh dấu!")
        self.bitboard.add_move(move, player)

    def available(self, move):
        return self.bitboard.get_state(move) is None

    def clear(self):
        self.bitboard.clear()

    def view(self):
        print("Current Board:")
        print(self.bitboard.view())

    def is_win(self, player: str):
        return self.bitboard.is_win(player)

    def hash(self):
        return hex(self.bitboard.hash()).upper()[2:]

def play_game():
    size = 15
    board = Board(size)
    players = ['X', 'O']
    turn = 0

    while True:
        board.view()
        player = players[turn % 2]
        print(f"Player {player}'s turn.")
        
        try:
            move = input("Enter your move as 'row col': ")
            row, col = map(int, move.split())
            if row < 0 or row >= size or col < 0 or col >= size:
                print("Invalid move. Try again.")
                continue
            
            if not board.available((row, col)):
                print("Cell already occupied. Try again.")
                continue

            board.add_move((row, col), player)

            if board.is_win(player):
                board.view()
                print(f"Player {player} wins!")
                break

            turn += 1
        except ValueError:
            print("Invalid input format. Please enter row and column as numbers.")

play_game()
