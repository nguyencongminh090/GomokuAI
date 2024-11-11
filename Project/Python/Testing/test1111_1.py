import math
import time

class Game:
    def __init__(self, size=4):
        self.size = size
        self.board = [0] * (size * size)  # 0 = empty, 1 = player X, -1 = player O

    def is_terminal(self):
        # Kiểm tra trạng thái thắng/thua hoặc không còn nước đi
        for row in range(self.size):
            if abs(sum(self.board[row * self.size:(row + 1) * self.size])) == self.size:
                return True  # Có một hàng chiến thắng
        for col in range(self.size):
            if abs(sum(self.board[col::self.size])) == self.size:
                return True  # Có một cột chiến thắng
        if abs(sum(self.board[::self.size + 1])) == self.size or abs(sum(self.board[self.size - 1:self.size * self.size - 1:self.size - 1])) == self.size:
            return True  # Có một đường chéo chiến thắng
        return all(x != 0 for x in self.board)  # Không còn nước đi

    def evaluate(self):
        # Hàm đánh giá trạng thái của bàn cờ
        for row in range(self.size):
            if sum(self.board[row * self.size:(row + 1) * self.size]) == self.size:
                return 10  # X thắng
            elif sum(self.board[row * self.size:(row + 1) * self.size]) == -self.size:
                return -10  # O thắng
        for col in range(self.size):
            if sum(self.board[col::self.size]) == self.size:
                return 10
            elif sum(self.board[col::self.size]) == -self.size:
                return -10
        if sum(self.board[::self.size + 1]) == self.size or sum(self.board[self.size - 1:self.size * self.size - 1:self.size - 1]) == self.size:
            return 10
        elif sum(self.board[::self.size + 1]) == -self.size or sum(self.board[self.size - 1:self.size * self.size - 1:self.size - 1]) == -self.size:
            return -10
        return 0

    def generate_moves(self):
        # Tạo danh sách các vị trí trống trên bảng
        return [i for i, x in enumerate(self.board) if x == 0]

    def make_move(self, move, player):
        self.board[move] = player
        # print(f"Making move at position {move} by {'X' if player == 1 else 'O'}")

    def undo_move(self, move):
        self.board[move] = 0
        # print(f"Undoing move at position {move}")

    def print_board(self):
        symbols = {1: 'X', -1: 'O', 0: '.'}
        for row in range(self.size):
            print(' '.join(symbols[self.board[row * self.size + col]] for col in range(self.size)))
        print()

def alpha_beta(game, depth, alpha, beta, maximizing_player):
    if depth == 0 or game.is_terminal():
        score = game.evaluate()
        print(f"Evaluating score at depth {depth}: {score}")
        return score

    if maximizing_player:
        max_eval = -math.inf
        for move in game.generate_moves():
            game.make_move(move, 1)
            eval = alpha_beta(game, depth - 1, alpha, beta, False)
            game.undo_move(move)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                # print("Pruning branches in maximizing player")
                break
        return max_eval
    else:
        min_eval = math.inf
        for move in game.generate_moves():
            game.make_move(move, -1)
            eval = alpha_beta(game, depth - 1, alpha, beta, True)
            game.undo_move(move)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                # print("Pruning branches in minimizing player")
                break
        return min_eval

def iterative_deepening(game, max_depth, player):
    best_move = None

    for depth in range(1, max_depth + 1):
        print(f"Starting Iterative Deepening at depth {depth}")
        best_score = -math.inf if player == 1 else math.inf

        for move in game.generate_moves():
            game.make_move(move, player)
            score = alpha_beta(game, depth, -math.inf, math.inf, player == -1)
            game.undo_move(move)

            if player == 1 and score > best_score:
                best_score = score
                best_move = move
            elif player == -1 and score < best_score:
                best_score = score
                best_move = move
            print(f"Move {move} at depth {depth} has score {score}")

    return best_move

# Main function for PvP game
def play_game():
    game = Game()
    current_player = 1  # X starts

    while not game.is_terminal():
        game.print_board()
        if current_player == 1:
            move = iterative_deepening(game, max_depth=3, player=current_player)
        else:
            available_moves = game.generate_moves()
            move = int(input(f"Player {'O' if current_player == -1 else 'X'}'s turn. Choose move {available_moves}: "))

        if move is not None and move in game.generate_moves():
            game.make_move(move, current_player)
            if game.is_terminal():
                print("Game Over!")
                break
            current_player = -current_player
        else:
            print("Invalid move! Try again.")
    
    game.print_board()
    print("Final board state:")
    result = game.evaluate()
    if result > 0:
        print("Player X wins!")
    elif result < 0:
        print("Player O wins!")
    else:
        print("It's a draw!")

play_game()
