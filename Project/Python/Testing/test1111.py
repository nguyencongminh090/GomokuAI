import math
import random

class Game:
    def __init__(self):
        self.board = [' ' for _ in range(9)]  # 3x3 board
        self.current_player = 'X'  # X starts

    def is_winner(self, player):
        win_positions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]  # Diagonals
        ]
        for positions in win_positions:
            if all(self.board[pos] == player for pos in positions):
                return True
        return False

    def is_draw(self):
        return ' ' not in self.board

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def make_move(self, position, player):
        self.board[position] = player

    def undo_move(self, position):
        self.board[position] = ' '

    def evaluate(self):
        # Kiểm tra xem có ai thắng không
        if self.is_winner('X'):
            return 100  # X thắng
        elif self.is_winner('O'):
            return -100  # O thắng

        # Nếu không thắng, ta tính điểm dựa trên số lượng các quân cờ liên tiếp
        x_score = 0
        o_score = 0

        # Các hướng kiểm tra: ngang, dọc và chéo
        win_positions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]  # Diagonals
        ]

        for positions in win_positions:
            x_count = 0
            o_count = 0
            for pos in positions:
                if self.board[pos] == 'X':
                    x_count += 1
                elif self.board[pos] == 'O':
                    o_count += 1

            # Đánh giá điểm cho từng dòng
            if x_count == 2 and o_count == 0:
                x_score += 10  # 2 X liên tiếp
            elif o_count == 2 and x_count == 0:
                o_score += 10  # 2 O liên tiếp

            if x_count == 3:
                x_score += 200  # Dòng hoàn chỉnh của X
            elif o_count == 3:
                o_score += 200  # Dòng hoàn chỉnh của O

        return x_score - o_score  # Trả về điểm chênh lệch giữa X và O

    def display(self):
        for i in range(3):
            print('|'.join(self.board[i*3:(i+1)*3]))
            if i < 2:
                print('-' * 5)

class IterativeDeepeningAlphaBeta:
    def __init__(self):
        self.game = Game()
        self.max_depth = 9  # Max depth for search

    def alpha_beta(self, game, depth, alpha, beta, maximizing_player):
        if depth == 0 or game.is_draw() or game.is_winner('X') or game.is_winner('O'):
            return game.evaluate()

        if maximizing_player: # Max
            max_eval = -math.inf
            for move in game.available_moves(): # Candidate
                game.make_move(move, 'X')
                eval = self.alpha_beta(game, depth-1, alpha, beta, False)
                game.undo_move(move)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # PVS: prune the branch
            return max_eval
        else: # Min
            min_eval = math.inf
            for move in game.available_moves():
                game.make_move(move, 'O')
                eval = self.alpha_beta(game, depth-1, alpha, beta, True)
                game.undo_move(move)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # PVS: prune the branch
            return min_eval

    def iterative_deepening(self):
        best_move = None
        for depth in range(1, self.max_depth + 1):
            best_value = -math.inf
            for move in self.game.available_moves():
                self.game.make_move(move, 'X')
                move_value = self.alpha_beta(self.game, depth-1, -math.inf, math.inf, False)
                self.game.undo_move(move)

                if move_value > best_value:
                    best_value = move_value
                    best_move = move

            print(f"Depth {depth}: Best move {best_move} with score {best_value}")
        return best_move

    def play(self):
        while not self.game.is_draw():
            self.game.display()

            if self.game.current_player == 'X':
                print("Player X's move:")
                move = self.iterative_deepening()
                self.game.make_move(move, 'X')
                self.game.current_player = 'O'
            else:
                print("Player O's move:")
                move = self.iterative_deepening()
                self.game.make_move(move, 'O')
                self.game.current_player = 'X'

            if self.game.is_winner('X'):
                self.game.display()
                print("Player X wins!")
                break
            elif self.game.is_winner('O'):
                self.game.display()
                print("Player O wins!")
                break

        if self.game.is_draw():
            self.game.display()
            print("It's a draw!")

# Run the game
game = IterativeDeepeningAlphaBeta()
game.play()
