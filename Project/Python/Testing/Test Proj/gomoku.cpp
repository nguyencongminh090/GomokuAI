#include <iostream>
#include <vector>
#include <unordered_map>
#include <utility>
#include <string>
#include <cmath>
#include <random>
#include <chrono>
#include <future>
#include <sstream>
#include <tuple>

// Enums
enum class Color { BLACK = 1, WHITE = 2 };

enum class ColorFlag { SELF, OPPO, EMPT };

enum class Pattern {
    DEAD = 0, OL, B1, F1, B2, F2, F2A, F2B, B3, F3, F3S, B4, F4, F5, PATTERN_NB
};

enum class Pattern4 {
    NONE = 0, L_FLEX2, K_BLOCK3, J_FLEX2_2X, I_BLOCK3_PLUS, H_FLEX3,
    G_FLEX3_PLUS, F_FLEX3_2X, E_BLOCK4, D_BLOCK4_PLUS, C_BLOCK4_FLEX3,
    B_FLEX4, A_FIVE, PATTERN4_NB
};

// Utility Function
std::string posToNotation(std::pair<int, int> pos) {
    char col = 'A' + pos.second;
    int row = 15 - pos.first;
    return std::string(1, col) + std::to_string(row);
}

// BitBoard Class
class BitBoard {
public:
    static const int SIZE = 15;
    static const int BITS_PER_POS = 2;
    static const int TOTAL_BITS = SIZE * SIZE * BITS_PER_POS;
    static const int UINT64_BITS = 64;
    static const int ARRAY_SIZE = (TOTAL_BITS + UINT64_BITS - 1) / UINT64_BITS;

    BitBoard() : bitBoard(ARRAY_SIZE, 0), lastMove(-1, -1), moveCount(0) {
        generateZobristTable();
    }

    int getState(std::pair<int, int> move) const {
        if (!checkValid(move)) return -1;
        int pos = move.first * SIZE + move.second;
        int arrayIdx = pos * BITS_PER_POS / UINT64_BITS;
        int bitOffset = pos * BITS_PER_POS % UINT64_BITS;
        uint64_t mask = 0b11ULL << bitOffset;
        return (bitBoard[arrayIdx] & mask) >> bitOffset;
    }

    bool addMove(std::pair<int, int> move, int player) {
        if (getState(move) != 0) return false;
        int pos = move.first * SIZE + move.second;
        int arrayIdx = pos * BITS_PER_POS / UINT64_BITS;
        int bitOffset = pos * BITS_PER_POS % UINT64_BITS;
        bitBoard[arrayIdx] |= (uint64_t)player << bitOffset;
        lastMove = move;
        moveCount++;
        return true;
    }

    bool resetPos(std::pair<int, int> move) {
        if (!checkValid(move)) return false;
        int pos = move.first * SIZE + move.second;
        int arrayIdx = pos * BITS_PER_POS / UINT64_BITS;
        int bitOffset = pos * BITS_PER_POS % UINT64_BITS;
        uint64_t mask = 0b11ULL << bitOffset;
        if ((bitBoard[arrayIdx] & mask) == 0) return false;
        bitBoard[arrayIdx] &= ~mask;
        lastMove = {-1, -1};
        moveCount = std::max(0, moveCount - 1);
        return true;
    }

    bool checkValid(std::pair<int, int> move) const {
        return move.first >= 0 && move.first < SIZE && move.second >= 0 && move.second < SIZE;
    }

    bool isWin(int player) const {
        const std::vector<std::pair<int, int>> directions = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
        for (int row = 0; row < SIZE; ++row) {
            for (int col = 0; col < SIZE; ++col) {
                if (getState({row, col}) != player) continue;
                for (const auto& [dRow, dCol] : directions) {
                    int count = 1;
                    int r = row + dRow, c = col + dCol;
                    while (r >= 0 && r < SIZE && c >= 0 && c < SIZE && getState({r, c}) == player) {
                        count++;
                        r += dRow;
                        c += dCol;
                    }
                    if (count == 5) {
                        auto before = std::pair<int, int>{row - dRow, col - dCol};
                        auto after = std::pair<int, int>{row + dRow * 5, col + dCol * 5};
                        int bState = checkValid(before) ? getState(before) : -1;
                        int aState = checkValid(after) ? getState(after) : -1;
                        if (bState != player && aState != player) return true;
                    }
                }
            }
        }
        return false;
    }

    Color getCurrentSide() const {
        return (moveCount % 2 == 0) ? Color::BLACK : Color::WHITE;
    }

    std::string view() const {
        std::ostringstream oss;
        for (int row = 0; row < SIZE; ++row) {
            for (int col = 0; col < SIZE; ++col) {
                int state = getState({row, col});
                char c = (state == 1) ? 'X' : (state == 2) ? 'O' : (state == 3) ? '*' : '.';
                oss << c << "  ";
            }
            oss << "\n";
        }
        return oss.str();
    }

    BitBoard copy() const {
        BitBoard newBoard;
        newBoard.bitBoard = bitBoard;
        newBoard.lastMove = lastMove;
        newBoard.moveCount = moveCount;
        newBoard.zobristTable = zobristTable;
        return newBoard;
    }

    uint64_t hash() const {
        uint64_t hashValue = 0;
        for (int row = 0; row < SIZE; ++row) {
            for (int col = 0; col < SIZE; ++col) {
                int state = getState({row, col});
                if (state == 1 || state == 2) {
                    hashValue ^= zobristTable.at({row, col, state});
                }
            }
        }
        return hashValue;
    }

private:
    std::vector<uint64_t> bitBoard;
    std::pair<int, int> lastMove;
    int moveCount;
    struct TupleHash {
        std::size_t operator()(const std::tuple<int, int, int>& t) const {
            return std::hash<int>{}(std::get<0>(t)) ^
                   std::hash<int>{}(std::get<1>(t)) ^
                   std::hash<int>{}(std::get<2>(t));
        }
    };
    std::unordered_map<std::tuple<int, int, int>, uint64_t, TupleHash> zobristTable;

    void generateZobristTable() {
        std::random_device rd;
        std::mt19937_64 gen(rd());
        std::uniform_int_distribution<uint64_t> dis;
        for (int row = 0; row < SIZE; ++row) {
            for (int col = 0; col < SIZE; ++col) {
                for (int player : {1, 2}) {
                    zobristTable[{row, col, player}] = dis(gen);
                }
            }
        }
    }
};

// Candidate Class
class Candidate {
public:
    Candidate(int mode = 0, int size = 15) : mode(mode), size(size) {}

    std::vector<std::pair<int, int>> expand(BitBoard& board) {
        std::vector<std::pair<int, int>> candidate;
        std::vector<std::pair<int, int>> marked;

        for (int row = 0; row < size; ++row) {
            for (int col = 0; col < size; ++col) {
                int state = board.getState({row, col});
                if (state == 1 || state == 2) {
                    if (mode == 0) squareLine(board, row, col, 3, 4, marked);
                    else if (mode == 1) circle34(board, row, col, marked);
                    else if (mode == 2) fullBoard(board, marked);
                }
            }
        }

        for (int row = 0; row < size; ++row) {
            for (int col = 0; col < size; ++col) {
                if (board.getState({row, col}) == 0b11) {
                    candidate.push_back({row, col});
                }
            }
        }

        for (const auto& pos : marked) {
            board.resetPos(pos);
        }

        return candidate;
    }

private:
    int mode;
    int size;

    void squareLine(BitBoard& board, int x, int y, int sq, int ln, std::vector<std::pair<int, int>>& marked) {
        const std::vector<std::pair<int, int>> directions = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}, {1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int k = 1; k <= ln; ++k) {
            for (const auto& [i, j] : directions) {
                std::pair<int, int> coord = {x + i * k, y + j * k};
                if (board.checkValid(coord) && board.getState(coord) == 0) {
                    board.addMove(coord, 3);
                    marked.push_back(coord);
                }
            }
        }
        for (int i = 1; i <= sq; ++i) {
            for (int j = 1; j <= sq; ++j) {
                std::vector<std::pair<int, int>> coords = {{x + i, y + j}, {x + i, y - j}, {x - i, y + j}, {x - i, y - j}};
                for (const auto& coord : coords) {
                    if (board.checkValid(coord) && board.getState(coord) == 0) {
                        board.addMove(coord, 3);
                        marked.push_back(coord);
                    }
                }
            }
        }
    }

    void circle34(BitBoard& board, int x, int y, std::vector<std::pair<int, int>>& marked) {
        double cr34 = std::sqrt(34);
        for (int row = -static_cast<int>(cr34); row <= static_cast<int>(cr34); ++row) {
            for (int col = -static_cast<int>(cr34); col <= static_cast<int>(cr34); ++col) {
                if (std::sqrt(row * row + col * col) <= cr34) {
                    std::pair<int, int> coord = {x + row, y + col};
                    if (board.checkValid(coord) && board.getState(coord) == 0) {
                        board.addMove(coord, 3);
                        marked.push_back(coord);
                    }
                }
            }
        }
    }

    void fullBoard(BitBoard& board, std::vector<std::pair<int, int>>& marked) {
        for (int row = 0; row < size; ++row) {
            for (int col = 0; col < size; ++col) {
                std::pair<int, int> coord = {row, col};
                if (board.getState(coord) == 0) {
                    board.addMove(coord, 3);
                    marked.push_back(coord);
                }
            }
        }
    }
};

// PatternDetector Class
class PatternDetector {
public:
    PatternDetector(const std::string& rule = "STANDARD") : rule(rule) {}

    Pattern getLinePattern(const std::vector<ColorFlag>& line, Color side) const {
        auto [realLen, fullLen, start, end] = countLine(line);

        if (rule == "STANDARD") {
            if (realLen >= 6) return Pattern::OL;
            if (realLen == 5) return Pattern::F5;
            if (fullLen < 5) return Pattern::DEAD;
        }

        std::vector<int> patternCounts(static_cast<int>(Pattern::PATTERN_NB), 0);
        std::vector<int> f5Indices;
        for (int i = start; i <= end; ++i) {
            if (line[i] == ColorFlag::EMPT) {
                auto newLine = line;
                newLine[i] = ColorFlag::SELF;
                Pattern newPattern = getLinePattern(newLine, side);
                patternCounts[static_cast<int>(newPattern)]++;
                if (newPattern == Pattern::F5 && f5Indices.size() < 2) f5Indices.push_back(i);
            }
        }

        if (patternCounts[static_cast<int>(Pattern::F5)] >= 2) return Pattern::F4;
        if (patternCounts[static_cast<int>(Pattern::F5)] == 1) return Pattern::B4;
        if (patternCounts[static_cast<int>(Pattern::F4)] >= 2) return Pattern::F3S;
        if (patternCounts[static_cast<int>(Pattern::F4)] == 1) return Pattern::F3;
        if (patternCounts[static_cast<int>(Pattern::B4)] >= 1) return Pattern::B3;
        int f3Count = patternCounts[static_cast<int>(Pattern::F3S)] + patternCounts[static_cast<int>(Pattern::F3)];
        if (f3Count >= 4) return Pattern::F2B;
        if (f3Count >= 3) return Pattern::F2A;
        if (f3Count >= 1) return Pattern::F2;
        if (patternCounts[static_cast<int>(Pattern::B3)] >= 1) return Pattern::B2;
        int f2Count = patternCounts[static_cast<int>(Pattern::F2)] + patternCounts[static_cast<int>(Pattern::F2A)] + patternCounts[static_cast<int>(Pattern::F2B)];
        if (f2Count >= 1) return Pattern::F1;
        if (patternCounts[static_cast<int>(Pattern::B2)] >= 1) return Pattern::B1;
        return Pattern::DEAD;
    }

    Pattern4 getCombinedPattern(const BitBoard& board, std::pair<int, int> pos, Color side) const {
        const std::vector<std::pair<int, int>> directions = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
        std::vector<Pattern> patterns;
        for (const auto& [dRow, dCol] : directions) {
            auto line = extractLine(board, pos, side, dRow, dCol);
            if (!line.empty()) patterns.push_back(getLinePattern(line, side));
        }
        while (patterns.size() < 4) patterns.push_back(Pattern::DEAD);

        std::vector<int> n(static_cast<int>(Pattern::PATTERN_NB), 0);
        for (const auto& p : patterns) n[static_cast<int>(p)]++;

        if (n[static_cast<int>(Pattern::F5)] >= 1) return Pattern4::A_FIVE;
        if (n[static_cast<int>(Pattern::OL)] >= 1) return Pattern4::NONE;
        if (n[static_cast<int>(Pattern::B4)] >= 2) return Pattern4::B_FLEX4;
        if (n[static_cast<int>(Pattern::F4)] >= 1) return Pattern4::B_FLEX4;
        if (n[static_cast<int>(Pattern::B4)] >= 1) {
            int f3Count = n[static_cast<int>(Pattern::F3)] + n[static_cast<int>(Pattern::F3S)];
            if (f3Count >= 1) return Pattern4::C_BLOCK4_FLEX3;
            int f2Count = n[static_cast<int>(Pattern::F2)] + n[static_cast<int>(Pattern::F2A)] + n[static_cast<int>(Pattern::F2B)];
            if (n[static_cast<int>(Pattern::B3)] >= 1 || f2Count >= 1) return Pattern4::D_BLOCK4_PLUS;
            return Pattern4::E_BLOCK4;
        }
        int f3Count = n[static_cast<int>(Pattern::F3)] + n[static_cast<int>(Pattern::F3S)];
        if (f3Count >= 1) {
            if (f3Count >= 2) return Pattern4::F_FLEX3_2X;
            int f2Count = n[static_cast<int>(Pattern::F2)] + n[static_cast<int>(Pattern::F2A)] + n[static_cast<int>(Pattern::F2B)];
            if (n[static_cast<int>(Pattern::B3)] >= 1 || f2Count >= 1) return Pattern4::G_FLEX3_PLUS;
            return Pattern4::H_FLEX3;
        }
        if (n[static_cast<int>(Pattern::B3)] >= 1) {
            int f2Count = n[static_cast<int>(Pattern::F2)] + n[static_cast<int>(Pattern::F2A)] + n[static_cast<int>(Pattern::F2B)];
            if (n[static_cast<int>(Pattern::B3)] >= 2 || f2Count >= 1) return Pattern4::I_BLOCK3_PLUS;
            return Pattern4::K_BLOCK3;
        }
        int f2Count = n[static_cast<int>(Pattern::F2)] + n[static_cast<int>(Pattern::F2A)] + n[static_cast<int>(Pattern::F2B)];
        if (f2Count >= 2) return Pattern4::J_FLEX2_2X;
        if (f2Count >= 1) return Pattern4::L_FLEX2;
        return Pattern4::NONE;
    }

private:
    std::string rule;

    std::tuple<int, int, int, int> countLine(const std::vector<ColorFlag>& line) const {
        int mid = line.size() / 2;
        int realLen = 1, fullLen = 1, realLenInc = 1;
        int start = mid, end = mid;

        for (int i = mid - 1; i >= 0; --i) {
            if (line[i] == ColorFlag::SELF) realLen += realLenInc;
            else if (line[i] == ColorFlag::OPPO) break;
            else realLenInc = 0;
            fullLen++;
            start = i;
        }
        realLenInc = 1;
        for (int i = mid + 1; i < line.size(); ++i) {
            if (line[i] == ColorFlag::SELF) realLen += realLenInc;
            else if (line[i] == ColorFlag::OPPO) break;
            else realLenInc = 0;
            fullLen++;
            end = i;
        }
        return {realLen, fullLen, start, end};
    }

    std::vector<ColorFlag> extractLine(const BitBoard& board, std::pair<int, int> move, Color side, int dRow, int dCol) const {
        std::vector<ColorFlag> line;
        int x = move.first, y = move.second;
        for (int i = -4; i <= 4; ++i) {
            int r = x + dRow * i, c = y + dCol * i;
            if (board.checkValid({r, c})) {
                int state = board.getState({r, c});
                if (state == static_cast<int>(side)) line.push_back(ColorFlag::SELF);
                else if (state == 0) line.push_back(ColorFlag::EMPT);
                else line.push_back(ColorFlag::OPPO);
            } else {
                line.push_back(ColorFlag::OPPO);
            }
        }
        return line;
    }
};

// PatternTracker Class
class PatternTracker {
public:
    PatternTracker(int boardSize) : boardSize(boardSize), detector("STANDARD") {
        patterns[Color::BLACK] = {};
        patterns[Color::WHITE] = {};
    }

    void update(const BitBoard& board, std::pair<int, int> move) {
        for (auto side : {Color::BLACK, Color::WHITE}) {
            for (int dy = -4; dy <= 4; ++dy) {
                for (int dx = -4; dx <= 4; ++dx) {
                    std::pair<int, int> pos = {move.first + dy, move.second + dx};
                    if (board.checkValid(pos) && board.getState(pos) == static_cast<int>(side)) {
                        patterns[side][pos] = detector.getCombinedPattern(board, pos, side);
                    }
                }
            }
        }
    }

    std::unordered_map<std::pair<int, int>, Pattern4, PairHash> getPatterns(Color side) const {
        return patterns.at(side);
    }

    PatternDetector detector; // Public for Evaluator access

private:
    int boardSize;
    struct PairHash {
        std::size_t operator()(const std::pair<int, int>& p) const {
            return std::hash<int>{}(p.first) ^ std::hash<int>{}(p.second);
        }
    };
    struct EnumHash {
        std::size_t operator()(Color c) const {
            return static_cast<std::size_t>(c);
        }
    };
    std::unordered_map<Color, std::unordered_map<std::pair<int, int>, Pattern4, PairHash>, EnumHash> patterns;
};

// Evaluator Class
class Evaluator {
public:
    Evaluator(int boardSize) : tracker(boardSize) {
        weights[Pattern4::A_FIVE] = 30000;
        weights[Pattern4::B_FLEX4] = 10000;
        weights[Pattern4::C_BLOCK4_FLEX3] = 5000;
        weights[Pattern4::D_BLOCK4_PLUS] = 2000;
        weights[Pattern4::E_BLOCK4] = 1000;
        weights[Pattern4::F_FLEX3_2X] = 500;
        weights[Pattern4::G_FLEX3_PLUS] = 200;
        weights[Pattern4::H_FLEX3] = 100;
        weights[Pattern4::I_BLOCK3_PLUS] = 50;
        weights[Pattern4::J_FLEX2_2X] = 20;
        weights[Pattern4::K_BLOCK3] = 10;
        weights[Pattern4::L_FLEX2] = 5;
        weights[Pattern4::NONE] = 0;
    }

    float evaluate(const BitBoard& board, std::pair<int, int> move, Color aiColor) {
        tracker.update(board, move);
        auto aiPatterns = tracker.getPatterns(aiColor);
        auto oppColor = (aiColor == Color::BLACK) ? Color::WHITE : Color::BLACK;
        auto oppPatterns = tracker.getPatterns(oppColor);

        float aiScore = 0, oppScore = 0;
        for (const auto& [_, p] : aiPatterns) aiScore += weights.at(p);
        for (const auto& [_, p] : oppPatterns) oppScore += weights.at(p);
        return aiScore - oppScore;
    }

    PatternTracker tracker; // Public for Search access

private:
    struct EnumHash {
        std::size_t operator()(Pattern4 p) const {
            return static_cast<std::size_t>(p);
        }
    };
    std::unordered_map<Pattern4, float, EnumHash> weights;
};

// Search Class
struct TreeNode {
    BitBoard boardState;
    uint64_t hashVal;
    std::pair<int, int> bestMove;
    float score;
    std::vector<std::pair<int, int>> pv;

    TreeNode(const BitBoard& board, uint64_t hash, std::pair<int, int> move = {-1, -1})
        : boardState(board), hashVal(hash), bestMove(move), score(0) {}
};

class Search {
public:
    Search(Color aiColor = Color::BLACK) : evaluator(15), aiColor(aiColor), nodesVisited(0) {}

    std::pair<std::pair<int, int>, std::vector<std::pair<int, int>>> idsSearch(const BitBoard& board, int maxDepth, float maxTime) {
        auto startTime = std::chrono::high_resolution_clock::now();
        std::pair<int, int> bestMove = {-1, -1};
        float bestScore = -std::numeric_limits<float>::infinity();
        std::vector<std::pair<int, int>> bestPv;
        Color currentSide = board.getCurrentSide();

        for (int depth = 1; depth <= maxDepth; ++depth) {
            nodesVisited = 0;
            transpositionTable.clear();
            auto iterationStart = std::chrono::high_resolution_clock::now();
            TreeNode root(board, board.hash());
            auto possibleMoves = getPossibleMoves(board);
            if (possibleMoves.empty()) break;

            std::vector<std::future<std::pair<float, std::vector<std::pair<int, int>>>>> futures;
            for (const auto& move : possibleMoves) {
                futures.push_back(std::async(std::launch::async, &Search::parallelAlphabeta, this, board, move, depth + 1, -std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), currentSide, maxDepth));
            }

            float iterationScore = -std::numeric_limits<float>::infinity();
            std::pair<int, int> iterationMove = {-1, -1};
            std::vector<std::pair<int, int>> iterationPv;
            int maxSelfDepth = depth;

            for (size_t i = 0; i < futures.size(); ++i) {
                try {
                    auto [score, pv] = futures[i].get();
                    int selfDepth = depth + pv.size();
                    maxSelfDepth = std::max(maxSelfDepth, selfDepth);
                    if (score > iterationScore) {
                        iterationScore = score;
                        iterationMove = possibleMoves[i];
                        iterationPv = pv;
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Error evaluating move: " << e.what() << std::endl;
                }
            }

            if (iterationMove.first != -1 && iterationScore > bestScore) {
                bestScore = iterationScore;
                bestMove = iterationMove;
                bestPv = iterationPv;
            }

            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - startTime).count();
            int nps = nodesVisited / std::max(1, static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - iterationStart).count()));
            float ev = (currentSide == aiColor) ? bestScore : -bestScore;
            std::ostringstream pvStr;
            for (const auto& m : bestPv) pvStr << posToNotation(m) << " ";
            std::cout << "DEPTH " << depth << "-" << maxSelfDepth << " EV " << static_cast<int>(ev) << " N " << nodesVisited << " NPS " << nps << " TM " << elapsed << " PV " << pvStr.str() << std::endl;

            if (elapsed >= maxTime) {
                std::cout << "Time limit reached at depth " << depth << std::endl;
                break;
            }
        }

        return {bestMove, bestPv};
    }

private:
    Evaluator evaluator;
    Color aiColor;
    std::unordered_map<uint64_t, std::pair<int, float>> transpositionTable;
    int nodesVisited;

    Color getOpponentSide(Color currentSide) const {
        return (currentSide == Color::BLACK) ? Color::WHITE : Color::BLACK;
    }

    std::vector<std::pair<int, int>> getPossibleMoves(const BitBoard& board) {
        Candidate candidate(0, BitBoard::SIZE);
        return candidate.expand(const_cast<BitBoard&>(board));
    }

    float alphabeta(TreeNode& node, int depth, float alpha, float beta, Color currentSide, int maxDepth) {
        nodesVisited++;
        if (depth == 0 || node.boardState.isWin(static_cast<int>(aiColor)) || node.boardState.isWin(static_cast<int>(getOpponentSide(aiColor)))) {
            return evaluator.evaluate(node.boardState, node.bestMove.first != -1 ? node.bestMove : std::pair<int, int>{7, 7}, currentSide);
        }

        bool extend = false;
        if (node.bestMove.first != -1) {
            auto pattern = evaluator.tracker.detector.getCombinedPattern(node.boardState, node.bestMove, currentSide);
            if (pattern == Pattern4::B_FLEX4 || pattern == Pattern4::C_BLOCK4_FLEX3 || pattern == Pattern4::E_BLOCK4) extend = true;
        }

        if (transpositionTable.count(node.hashVal) && transpositionTable[node.hashVal].first >= depth) {
            return transpositionTable[node.hashVal].second;
        }

        auto possibleMoves = getPossibleMoves(node.boardState);
        if (possibleMoves.empty()) return 0.0f;

        bool isMaximizing = (currentSide == aiColor);
        float value = isMaximizing ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
        std::pair<int, int> bestMoveHere = {-1, -1};

        for (const auto& move : possibleMoves) {
            auto childBoard = node.boardState.copy();
            if (childBoard.addMove(move, static_cast<int>(currentSide))) {
                TreeNode childNode(childBoard, childBoard.hash(), move);
                int searchDepth = (extend && isMaximizing && depth < maxDepth) ? depth : depth - 1;
                float score = -alphabeta(childNode, searchDepth, -beta, -alpha, getOpponentSide(currentSide), maxDepth);

                if (isMaximizing) {
                    if (score > value) {
                        value = score;
                        bestMoveHere = move;
                        node.pv = {move};
                        node.pv.insert(node.pv.end(), childNode.pv.begin(), childNode.pv.end());
                    }
                    alpha = std::max(alpha, value);
                    if (alpha >= beta) break;
                } else {
                    if (score < value) {
                        value = score;
                        bestMoveHere = move;
                        node.pv = {move};
                        node.pv.insert(node.pv.end(), childNode.pv.begin(), childNode.pv.end());
                    }
                    beta = std::min(beta, value);
                    if (alpha >= beta) break;
                }
            }
        }

        node.bestMove = bestMoveHere;
        node.score = value;
        transpositionTable[node.hashVal] = {depth, value};
        return value;
    }

    std::pair<float, std::vector<std::pair<int, int>>> parallelAlphabeta(const BitBoard& board, std::pair<int, int> move, int depth, float alpha, float beta, Color currentSide, int maxDepth) {
        auto childBoard = board.copy();
        if (!childBoard.addMove(move, static_cast<int>(currentSide))) {
            return {-std::numeric_limits<float>::infinity(), {}};
        }
        TreeNode childNode(childBoard, childBoard.hash(), move);
        float score = -alphabeta(childNode, depth - 1, -beta, -alpha, getOpponentSide(currentSide), maxDepth);
        return {score, childNode.pv};
    }
};

// Main Function
int main() {
    BitBoard board;
    board.addMove({7, 7}, static_cast<int>(Color::BLACK));
    Search ai(Color::BLACK);
    Color playerColor = Color::WHITE;
    std::cout << "Gomoku Game - You are 'O' (White), AI is 'X' (Black)\n";

    std::vector<std::pair<std::pair<int, int>, int>> moves = {
        {{7, 7}, static_cast<int>(Color::BLACK)}, {{7, 6}, static_cast<int>(Color::WHITE)},
        {{6, 6}, static_cast<int>(Color::BLACK)}, {{5, 5}, static_cast<int>(Color::WHITE)},
        {{6, 8}, static_cast<int>(Color::BLACK)}, {{6, 5}, static_cast<int>(Color::WHITE)},
        {{6, 7}, static_cast<int>(Color::BLACK)}, {{5, 7}, static_cast<int>(Color::WHITE)}
    };

    for (const auto& [move, color] : moves) {
        if (!board.addMove(move, color)) {
            std::cout << "Failed to add move at (" << move.first << "," << move.second << ") for " << (color == 1 ? "BLACK" : "WHITE") << ".\n";
        }
    }

    while (true) {
        std::cout << "\nCurrent Board:\n" << board.view() << std::endl;

        if (board.getCurrentSide() == Color::BLACK) {
            std::cout << "AI thinking...\n";
            auto start = std::chrono::high_resolution_clock::now();
            auto [move, pv] = ai.idsSearch(board, 4, std::numeric_limits<float>::infinity());
            if (move.first != -1) {
                board.addMove(move, static_cast<int>(Color::BLACK));
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1000.0;
                std::ostringstream pvStr;
                for (const auto& m : pv) pvStr << posToNotation(m) << " ";
                std::cout << "AI moved to (" << move.first << "," << move.second << ") in " << duration << "s, PV: " << pvStr.str() << std::endl;
            } else {
                std::cout << "AI has no valid move!\n";
                break;
            }
            if (board.isWin(static_cast<int>(Color::BLACK))) {
                std::cout << "\nFinal Board:\n" << board.view() << "\nAI (Black) wins!\n";
                break;
            }
        } else {
            std::cout << "Enter your move (row,col) e.g., '7,7': ";
            std::string input;
            std::getline(std::cin, input);
            std::istringstream iss(input);
            int row, col;
            char comma;
            iss >> row >> comma >> col;
            std::pair<int, int> move = {row, col};
            if (board.addMove(move, static_cast<int>(Color::WHITE))) {
                if (board.isWin(static_cast<int>(Color::WHITE))) {
                    std::cout << "\nFinal Board:\n" << board.view() << "\nYou (White) win!\n";
                    break;
                }
            } else {
                std::cout << "Invalid move, try again.\n";
            }
        }
    }
    return 0;
}