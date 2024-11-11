class BitBoard:
    def __init__(self, board=0):
        self.board = board  # Sử dụng một số nguyên để biểu diễn bàn cờ

    def set_bit(self, pos, player):
        # Đặt bit cho vị trí 'pos' với giá trị của người chơi ('1' cho X, '2' cho O)
        self.board &= ~(3 << (2 * pos))  # Xóa bit hiện tại ở vị trí đó
        if player == 1:  # X
            self.board |= (1 << (2 * pos))  # Đặt bit lẻ
        elif player == 2:  # O
            self.board |= (2 << (2 * pos))  # Đặt bit chẵn

    def get_bit(self, pos):
        # Lấy bit tại vị trí 'pos'
        return (self.board >> (2 * pos)) & 3  # 2 bits per position

    def display(self):
        # Hiển thị bảng cờ dưới dạng 3x3
        symbols = {0: '.', 1: 'X', 2: 'O'}
        print(bin(self.board))
        for i in range(9):
            print(symbols[self.get_bit(i)], end=' ')
            if (i + 1) % 3 == 0:
                print()

    def check_winner(self):
        # Kiểm tra người chiến thắng
        winning_positions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Hàng ngang
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Hàng dọc
            [0, 4, 8], [2, 4, 6]              # Hàng chéo
        ]
        for positions in winning_positions:
            bits = [self.get_bit(pos) for pos in positions]
            if bits[0] != 0 and bits.count(bits[0]) == 3:
                return bits[0]  # Trả về người thắng (1 cho X, 2 cho O)
        return 0  # 0 nghĩa là chưa có ai thắng

    def is_full(self):
        # Kiểm tra nếu tất cả các ô đều đã được điền
        return all(self.get_bit(pos) != 0 for pos in range(9))

# Minimax với Alpha-Beta pruning
def minimax(board, depth, alpha, beta, is_maximizing):
    winner = board.check_winner()
    if winner == 1:  # X thắng
        return 1
    elif winner == 2:  # O thắng
        return -1
    elif board.is_full():  # Hòa
        return 0

    if is_maximizing:
        max_eval = -float('inf')
        for i in range(9):
            if board.get_bit(i) == 0:  # Vị trí trống
                board.set_bit(i, 1)  # Đặt X
                eval = minimax(board, depth + 1, alpha, beta, False)
                board.set_bit(i, 0)  # Hoàn tác bước đi
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
        return max_eval
    else:
        min_eval = float('inf')
        for i in range(9):
            if board.get_bit(i) == 0:  # Vị trí trống
                board.set_bit(i, 2)  # Đặt O
                eval = minimax(board, depth + 1, alpha, beta, True)
                board.set_bit(i, 0)  # Hoàn tác bước đi
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
        return min_eval

# Hàm tìm nước đi tốt nhất cho máy (X)
def find_best_move(board):
    best_move = -1
    best_value = -float('inf')
    for i in range(9):
        if board.get_bit(i) == 0:  # Vị trí trống
            board.set_bit(i, 1)  # Đặt X
            move_value = minimax(board, 0, -float('inf'), float('inf'), False)
            board.set_bit(i, 0)  # Hoàn tác bước đi
            if move_value > best_value:
                best_value = move_value
                best_move = i
    return best_move

# Chơi trò chơi
def play_game():
    board = BitBoard()
    print("Bắt đầu trò chơi: Bạn là O và máy là X.")
    board.display()

    while True:
        # Người chơi (O) đi
        player_move = int(input("Nhập vị trí của bạn (0-8): "))
        if board.get_bit(player_move) != 0:
            print("Vị trí đã được đi, vui lòng chọn vị trí khác.")
            continue
        board.set_bit(player_move, 2)  # Đặt O

        board.display()
        if board.check_winner() == 2:
            print("Bạn thắng!")
            break
        elif board.is_full():
            print("Hòa!")
            break

        # Máy (X) đi
        print("Máy đang suy nghĩ...")
        computer_move = find_best_move(board)
        board.set_bit(computer_move, 1)  # Đặt X
        print(f"Máy đi vào vị trí {computer_move}")
        board.display()

        if board.check_winner() == 1:
            print("Máy thắng!")
            break
        elif board.is_full():
            print("Hòa!")
            break

# Bắt đầu trò chơi
play_game()
