#include <bits/stdc++.h>

const int BOARD_SIZE = 15;
std::unordered_map<std::string, int> transpositionTable;
std::mutex tableMutex;

std::string generateBoardKey(int board[BOARD_SIZE][BOARD_SIZE]) {
    std::string key;
    for (int i = 0; i < BOARD_SIZE; ++i) {
        for (int j = 0; j < BOARD_SIZE; ++j) {
            key += std::to_string(board[i][j]);
        }
    }
    return key;
}

bool isWinningMove(int board[BOARD_SIZE][BOARD_SIZE], int player) {
    // Kiểm tra hàng ngang
    for (int row = 0; row < BOARD_SIZE; ++row) {
        for (int col = 0; col <= BOARD_SIZE - 5; ++col) {
            bool win = true;
            for (int k = 0; k < 5; ++k) {
                if (board[row][col + k] != player) {
                    win = false;
                    break;
                }
            }
            if (win) return true;
        }
    }

    // Kiểm tra hàng dọc
    for (int col = 0; col < BOARD_SIZE; ++col) {
        for (int row = 0; row <= BOARD_SIZE - 5; ++row) {
            bool win = true;
            for (int k = 0; k < 5; ++k) {
                if (board[row + k][col] != player) {
                    win = false;
                    break;
                }
            }
            if (win) return true;
        }
    }

    // Kiểm tra đường chéo chính
    for (int row = 0; row <= BOARD_SIZE - 5; ++row) {
        for (int col = 0; col <= BOARD_SIZE - 5; ++col) {
            bool win = true;
            for (int k = 0; k < 5; ++k) {
                if (board[row + k][col + k] != player) {
                    win = false;
                    break;
                }
            }
            if (win) return true;
        }
    }

    // Kiểm tra đường chéo phụ
    for (int row = 4; row < BOARD_SIZE; ++row) {
        for (int col = 0; col <= BOARD_SIZE - 5; ++col) {
            bool win = true;
            for (int k = 0; k < 5; ++k) {
                if (board[row - k][col + k] != player) {
                    win = false;
                    break;
                }
            }
            if (win) return true;
        }
    }

    return false;
}

bool isDraw(int board[BOARD_SIZE][BOARD_SIZE]) {
    for (int row = 0; row < BOARD_SIZE; ++row) {
        for (int col = 0; col < BOARD_SIZE; ++col) {
            if (board[row][col] == 0) return false;
        }
    }
    return true;
}


void printBoard(int board[BOARD_SIZE][BOARD_SIZE]) {
    for (int row = 0; row < BOARD_SIZE; ++row) {
        for (int col = 0; col < BOARD_SIZE; ++col) {
            if (board[row][col] == 1) std::cout << "X ";
            else if (board[row][col] == -1) std::cout << "O ";
            else std::cout << ". ";
        }
        std::cout << std::endl;
    }
}

void getPlayerMove(int board[BOARD_SIZE][BOARD_SIZE], bool isX) {
    int row, col;
    std::cout << "Enter your move (row and column): ";
    std::cin >> row >> col;
    while (row < 0 || row >= BOARD_SIZE || col < 0 || col >= BOARD_SIZE || board[row][col] != 0) {
        std::cout << "Invalid move. Enter your move (row and column): ";
        std::cin >> row >> col;
    }
    board[row][col] = isX ? 1 : -1;
}


int evaluateBoard(int board[BOARD_SIZE][BOARD_SIZE]) {
    int score = 0;

    // Đánh giá các hàng
    for (int row = 0; row < BOARD_SIZE; ++row) {
        for (int col = 0; col <= BOARD_SIZE - 5; ++col) {
            int countX = 0, countO = 0;
            for (int k = 0; k < 5; ++k) {
                if (board[row][col + k] == 1) countX++;
                else if (board[row][col + k] == -1) countO++;
            }
            if (countX > 0 && countO == 0) score += (countX * countX);
            else if (countO > 0 && countX == 0) score -= (countO * countO);
        }
    }

    // Đánh giá các cột
    for (int col = 0; col < BOARD_SIZE; ++col) {
        for (int row = 0; row <= BOARD_SIZE - 5; ++row) {
            int countX = 0, countO = 0;
            for (int k = 0; k < 5; ++k) {
                if (board[row + k][col] == 1) countX++;
                else if (board[row + k][col] == -1) countO++;
            }
            if (countX > 0 && countO == 0) score += (countX * countX);
            else if (countO > 0 && countX == 0) score -= (countO * countO);
        }
    }

    // Đánh giá các đường chéo chính
    for (int row = 0; row <= BOARD_SIZE - 5; ++row) {
        for (int col = 0; col <= BOARD_SIZE - 5; ++col) {
            int countX = 0, countO = 0;
            for (int k = 0; k < 5; ++k) {
                if (board[row + k][col + k] == 1) countX++;
                else if (board[row + k][col + k] == -1) countO++;
            }
            if (countX > 0 && countO == 0) score += (countX * countX);
            else if (countO > 0 && countX == 0) score -= (countO * countO);
        }
    }

    // Đánh giá các đường chéo phụ
    for (int row = 4; row < BOARD_SIZE; ++row) {
        for (int col = 0; col <= BOARD_SIZE - 5; ++col) {
            int countX = 0, countO = 0;
            for (int k = 0; k < 5; ++k) {
                if (board[row - k][col + k] == 1) countX++;
                else if (board[row - k][col + k] == -1) countO++;
            }
            if (countX > 0 && countO == 0) score += (countX * countX);
            else if (countO > 0 && countX == 0) score -= (countO * countO);
        }
    }

    return score;
}

std::vector<int> generateMoves(int board[BOARD_SIZE][BOARD_SIZE], bool maximizingPlayer) {
    std::vector<int> moves;

    for (int row = 0; row < BOARD_SIZE; ++row) {
        for (int col = 0; col < BOARD_SIZE; ++col) {
            if (board[row][col] == 0) {
                moves.push_back(row * BOARD_SIZE + col);
            }
        }
    }

    return moves;
}

void orderMoves(std::vector<int>& moves, int board[BOARD_SIZE][BOARD_SIZE], bool maximizingPlayer) {
    std::vector<std::pair<int, int>> scoredMoves;

    for (int move : moves) {
        int row = move / BOARD_SIZE;
        int col = move % BOARD_SIZE;
        board[row][col] = maximizingPlayer ? 1 : -1;
        int score = evaluateBoard(board);
        scoredMoves.push_back({score, move});
        board[row][col] = 0;
    }

    std::sort(scoredMoves.begin(), scoredMoves.end(), [&](std::pair<int, int> a, std::pair<int, int> b) {
        return maximizingPlayer ? (a.first > b.first) : (a.first < b.first);
    });

    moves.clear();
    for (auto& scoredMove : scoredMoves) {
        moves.push_back(scoredMove.second);
    }
}


int quiescenceSearch(int board[BOARD_SIZE][BOARD_SIZE], int alpha, int beta, bool maximizingPlayer) {
    int stand_pat = evaluateBoard(board);
    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;

    for (auto& move : generateMoves(board, maximizingPlayer)) {
        int score = -quiescenceSearch(board, -beta, -alpha, !maximizingPlayer);
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
    }
    return alpha;
}

int alphaBeta(int board[BOARD_SIZE][BOARD_SIZE], int depth, int alpha, int beta, bool maximizingPlayer) {
    std::string boardKey = generateBoardKey(board);
    
    {
        std::lock_guard<std::mutex> lock(tableMutex);
        if (transpositionTable.find(boardKey) != transpositionTable.end()) {
            return transpositionTable[boardKey];
        }
    }

    int eval;
    if (depth == 0) {
        eval = quiescenceSearch(board, alpha, beta, maximizingPlayer);
    } else {
        if (maximizingPlayer) {
            eval = std::numeric_limits<int>::min();
            for (auto& move : generateMoves(board, true)) {
                eval = std::max(eval, alphaBeta(board, depth - 1, alpha, beta, false));
                alpha = std::max(alpha, eval);
                if (beta <= alpha) break;
            }
        } else {
            eval = std::numeric_limits<int>::max();
            for (auto& move : generateMoves(board, false)) {
                eval = std::min(eval, alphaBeta(board, depth - 1, alpha, beta, true));
                beta = std::min(beta, eval);
                if (beta <= alpha) break;
            }
        }
    }

    {
        std::lock_guard<std::mutex> lock(tableMutex);
        transpositionTable[boardKey] = eval;
    }
    return eval;
}

void parallelAlphaBeta(int board[BOARD_SIZE][BOARD_SIZE], int depth, int& result, bool maximizingPlayer) {
    std::vector<std::thread> threads;
    std::vector<int> moves = generateMoves(board, maximizingPlayer);
    orderMoves(moves, board, maximizingPlayer);  // Cập nhật đối số

    for (auto& move : moves) {
        threads.emplace_back([&](int move) {
            int localResult = alphaBeta(board, depth - 1, std::numeric_limits<int>::min(), std::numeric_limits<int>::max(), !maximizingPlayer);
            {
                std::lock_guard<std::mutex> lock(tableMutex);
                result = std::max(result, localResult);
            }
        }, move);
    }

    for (auto& th : threads) {
        if (th.joinable()) {
            th.join();
        }
    }
}


int iterativeDeepening(int board[BOARD_SIZE][BOARD_SIZE], int maxDepth) {
    int bestMove;
    for (int depth = 1; depth <= maxDepth; ++depth) {
        parallelAlphaBeta(board, depth, bestMove, true);
    }
    return bestMove;
}

int main() {
    int board[BOARD_SIZE][BOARD_SIZE] = {0}; // Khởi tạo bảng cờ caro
    int maxDepth = 6; // Độ sâu tối đa cho thuật toán
    bool isX = true; // X bắt đầu trước

    while (true) {
        printBoard(board);
        if (isX) {
            std::cout << "Player X's turn" << std::endl;
            getPlayerMove(board, isX);
        } else {
            std::cout << "Bot O's turn" << std::endl;
            int bestMove = iterativeDeepening(board, maxDepth);
            int row = bestMove / BOARD_SIZE;
            int col = bestMove % BOARD_SIZE;
            board[row][col] = -1; // Bot đánh dấu O
        }
        
        if (isWinningMove(board, isX ? 1 : -1)) {
            printBoard(board);
            std::cout << (isX ? "Player X" : "Bot O") << " wins!" << std::endl;
            break;
        }

        if (isDraw(board)) {
            printBoard(board);
            std::cout << "It's a draw!" << std::endl;
            break;
        }

        isX = !isX; // Đổi lượt người chơi
    }

    return 0;
}


