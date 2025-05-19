#include <iostream>
#include <vector>
#include <unordered_map>
#include <thread>
#include <future>
#include <mutex>
#include <random>
#include <chrono>
#include <cmath>
#include <limits>

using namespace std;

// Enums
enum class Color {
    BLACK = 1,
    WHITE = 2
};

enum class ColorFlag {
    SELF = 1,
    OPPO = 2,
    EMPT = 3
};

// Utility functions
bool check_valid(int size, pair<int, int> move) {
    return move.first >= 0 && move.first < size && move.second >= 0 && move.second < size;
}

string pos_to_notation(pair<int, int> pos) {
    int row = pos.first;
    int col = pos.second;
    return string(1, 'A' + col) + to_string(15 - row);  // For 15x15 board
}

// Board Representation
class BitBoard {
public:
    int size;
    unsigned long long bit_board;
    pair<int, int> last_move;
    int move_count;

    BitBoard(int size = 15) : size(size), bit_board(0), move_count(0) {}

    bool add_move(pair<int, int> move, int player) {
        if (get_state(move) != 0) {
            return false;  // If the position is already filled, return false
        }
        int row = move.first;
        int col = move.second;
        int pos = row * size + col;
        bit_board |= (player << (pos * 2));  // Set the player's move on the board
        last_move = move;
        move_count++;
        return true;
    }

    int get_state(pair<int, int> move) {
        if (!check_valid(size, move)) return -1;  // If the position is invalid, return -1
        int row = move.first;
        int col = move.second;
        int pos = row * size + col;
        unsigned long long mask = 0b11 << (pos * 2);
        int state_bits = (bit_board & mask) >> (pos * 2);
        return state_bits & 0b11;  // Return the state of the position
    }

    bool is_win(int player) {
        static const vector<pair<int, int>> directions = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
        for (int row = 0; row < size; ++row) {
            for (int col = 0; col < size; ++col) {
                if (get_state({row, col}) != player) continue;
                for (const auto& dir : directions) {
                    int count = 1;
                    int r = row + dir.first, c = col + dir.second;
                    while (check_valid(size, {r, c}) && get_state({r, c}) == player) {
                        count++;
                        r += dir.first;
                        c += dir.second;
                    }
                    if (count == 5) {
                        return true;  // If 5 in a row, player wins
                    }
                }
            }
        }
        return false;
    }

    Color get_current_side() {
        return move_count % 2 == 0 ? Color::BLACK : Color::WHITE;  // Black starts first
    }

    void view() {
        for (int row = 0; row < size; ++row) {
            for (int col = 0; col < size; ++col) {
                int state = get_state({row, col});
                if (state == 1) cout << "X ";  // BLACK
                else if (state == 2) cout << "O ";  // WHITE
                else cout << ". ";  // EMPTY
            }
            cout << endl;
        }
    }
};

// Candidate Move Generator
class Candidate {
public:
    int size;
    Candidate(int size = 15) : size(size) {}

    vector<pair<int, int>> expand(BitBoard& board) {
        vector<pair<int, int>> candidate;
        for (int row = 0; row < size; ++row) {
            for (int col = 0; col < size; ++col) {
                if (board.get_state({row, col}) == 0) {
                    candidate.push_back({row, col});  // Add empty cells to candidate moves
                }
            }
        }
        return candidate;
    }
};

// Search with parallelization (multi-threading)
class Search {
public:
    Color ai_color;

    Search(Color ai_color = Color::BLACK) : ai_color(ai_color) {}

    Color get_opponent_side(Color current_side) {
        return current_side == Color::BLACK ? Color::WHITE : Color::BLACK;
    }

    vector<pair<int, int>> get_possible_moves(BitBoard& board) {
        Candidate candidate(board.size);
        return candidate.expand(board);  // Generate possible moves
    }

    float alphabeta(BitBoard& board, int depth, float alpha, float beta, Color current_side, int max_depth) {
        if (depth == 0 || board.is_win((int)ai_color)) {
            return 1.0f;  // Simple evaluation for this example
        }

        float value = (current_side == ai_color) ? -numeric_limits<float>::infinity() : numeric_limits<float>::infinity();
        vector<pair<int, int>> possible_moves = get_possible_moves(board);

        for (const auto& move : possible_moves) {
            BitBoard child_board = board;
            if (child_board.add_move(move, (int)current_side)) {
                float score = -alphabeta(child_board, depth - 1, -beta, -alpha, get_opponent_side(current_side), max_depth);
                if (current_side == ai_color) {
                    value = max(value, score);
                    alpha = max(alpha, value);
                } else {
                    value = min(value, score);
                    beta = min(beta, value);
                }

                if (alpha >= beta) break;  // Beta cutoff
            }
        }

        return value;
    }

    // Modify the return type to return a pair: move and score
    pair<pair<int, int>, float> parallel_alphabeta(BitBoard& board, pair<int, int> move, int depth, float alpha, float beta, Color current_side, int max_depth) {
        BitBoard child_board = board;
        if (!child_board.add_move(move, (int)current_side)) return {{-1, -1}, -numeric_limits<float>::infinity()}; // Invalid move

        float score = alphabeta(child_board, depth - 1, -beta, -alpha, get_opponent_side(current_side), max_depth);
        return {move, score};  // Return the move and its score as a pair
    }

    pair<pair<int, int>, float> ids_search(BitBoard& board, int max_depth, float max_time) {
        auto start_time = chrono::high_resolution_clock::now();
        pair<int, int> best_move;
        float best_score = -numeric_limits<float>::infinity();
        Color current_side = board.get_current_side();

        for (int depth = 1; depth <= max_depth; ++depth) {
            vector<pair<pair<int, int>, float>> results;
            vector<future<pair<pair<int, int>, float>>> futures;

            vector<pair<int, int>> possible_moves = get_possible_moves(board);

            for (const auto& move : possible_moves) {
                // Use async to run parallel moves
                futures.push_back(async(launch::async, &Search::parallel_alphabeta, this, ref(board), move, depth, -numeric_limits<float>::infinity(), numeric_limits<float>::infinity(), current_side, max_depth));
            }

            // Collect results from all futures
            for (auto& fut : futures) {
                auto result = fut.get();
                if (result.second > best_score) {
                    best_score = result.second;
                    best_move = result.first;
                }
            }

            auto elapsed = chrono::high_resolution_clock::now() - start_time;
            auto duration = chrono::duration_cast<chrono::seconds>(elapsed).count();
            if (duration >= max_time) break;
        }

        return {best_move, best_score};
    }
};

// Main game loop
void main_game_loop() {
    BitBoard board(15);
    board.add_move({7, 7}, 1);  // First move by AI
    Search ai(Color::BLACK);
    Color player_color = Color::WHITE;
    cout << "Gomoku Game - You are 'O' (White), AI is 'X' (Black)" << endl;

    while (true) {
        cout << "\nCurrent Board:" << endl;
        board.view();

        // AI Move
        if (board.get_current_side() == Color::BLACK) {
            cout << "AI thinking..." << endl;
            auto [move, score] = ai.ids_search(board, 4, 30.0f);
            board.add_move(move, (int)Color::BLACK);
            cout << "AI moved to " << pos_to_notation(move) << endl;

            if (board.is_win((int)Color::BLACK)) {
                cout << "\nFinal Board:" << endl;
                board.view();
                cout << "AI (Black) wins!" << endl;
                break;
            }
        }

        // Player Move
        else {
            string move_str;
            cout << "Enter your move (row, col) e.g., '7,7': ";
            getline(cin, move_str);
            int row, col;
            sscanf(move_str.c_str(), "%d,%d", &row, &col);
            if (board.add_move({row, col}, (int)Color::WHITE)) {
                if (board.is_win((int)Color::WHITE)) {
                    cout << "\nFinal Board:" << endl;
                    board.view();
                    cout << "You (White) win!" << endl;
                    break;
                }
            } else {
                cout << "Invalid move, try again." << endl;
            }
        }
    }
}

int main() {
    main_game_loop();
    return 0;
}
