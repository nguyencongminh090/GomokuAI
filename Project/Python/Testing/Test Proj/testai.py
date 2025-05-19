import logging
from typing import Tuple, List, Dict, Optional
from enum import Enum, auto
from functools import lru_cache
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Enums
class Color(Enum):
    BLACK = 1
    WHITE = 2

class ColorFlag(Enum):
    SELF = auto()
    OPPO = auto()
    EMPT = auto()

class Pattern(Enum):
    DEAD = 0
    OL = 1
    B1 = 2
    F1 = 3
    B2 = 4
    F2 = 5
    F2A = 6
    F2B = 7
    B3 = 8
    F3 = 9
    F3S = 10
    B4 = 11
    F4 = 12
    F5 = 13
    PATTERN_NB = 14

class Pattern4(Enum):
    NONE = 0
    L_FLEX2 = 1
    K_BLOCK3 = 2
    J_FLEX2_2X = 3
    I_BLOCK3_PLUS = 4
    H_FLEX3 = 5
    G_FLEX3_PLUS = 6
    F_FLEX3_2X = 7
    E_BLOCK4 = 8
    D_BLOCK4_PLUS = 9
    C_BLOCK4_FLEX3 = 10
    B_FLEX4 = 11
    A_FIVE = 12
    PATTERN4_NB = 13

# Utility Functions
def check_valid(size: int, move: Tuple[int, int]) -> bool:
    return 0 <= move[0] < size and 0 <= move[1] < size

def pos_to_notation(pos: Tuple[int, int]) -> str:
    """Convert (row, col) to Gomoku notation (e.g., (7, 7) -> H8)."""
    row, col = pos
    return f"{chr(ord('A') + col)}{15 - row}"

# Board Representation
class BitBoard:
    def __init__(self, size: int = 15):
        self.size = size
        self.bit_board = 0
        self.last_move = None
        self.move_count = 0
        self.zobrist_table = self._generate_zobrist_table()

    def _generate_zobrist_table(self) -> Dict[Tuple[int, int, int], int]:
        table = {}
        for row in range(self.size):
            for col in range(self.size):
                for player in [1, 2]:
                    table[(row, col, player)] = random.getrandbits(64)
        return table

    def get_state(self, move: Tuple[int, int]) -> int:
        if not check_valid(self.size, move):
            return -1
        row, col = move
        pos = row * self.size + col
        mask = 0b11 << (pos * 2)
        state_bits = (self.bit_board & mask) >> (pos * 2)
        return state_bits & 0b11

    def hash(self) -> int:
        hash_value = 0
        for row in range(self.size):
            for col in range(self.size):
                state = self.get_state((row, col))
                if state in [1, 2]:
                    hash_value ^= self.zobrist_table[(row, col, state)]
        return hash_value

    def add_move(self, move: Tuple[int, int], player: int) -> bool:
        if self.get_state(move) != 0:
            return False
        row, col = move
        pos = row * self.size + col
        self.bit_board |= player << (pos * 2)
        self.last_move = move
        self.move_count += 1
        return True

    def reset_pos(self, move: Tuple[int, int]) -> bool:
        if not check_valid(self.size, move):
            return False
        row, col = move
        pos = row * self.size + col
        mask = 0b11 << (pos * 2)
        current_state = (self.bit_board & mask) >> (pos * 2)
        if current_state == 0:
            return False
        self.bit_board &= ~mask
        self.last_move = None
        self.move_count = max(0, self.move_count - 1)
        return True

    def is_win(self, player: int) -> bool:
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for row in range(self.size):
            for col in range(self.size):
                if self.get_state((row, col)) != player:
                    continue
                for d_row, d_col in directions:
                    count = 1
                    r, c = row + d_row, col + d_col
                    while 0 <= r < self.size and 0 <= c < self.size and self.get_state((r, c)) == player:
                        count += 1
                        r += d_row
                        c += d_col
                    if count == 5:
                        before = (row - d_row, col - d_col)
                        after = (row + d_row * 5, col + d_col * 5)
                        b_state = self.get_state(before) if check_valid(self.size, before) else -1
                        a_state = self.get_state(after) if check_valid(self.size, after) else -1
                        if b_state != player and a_state != player:
                            return True
        return False

    def get_current_side(self) -> Color:
        return Color.BLACK if self.move_count % 2 == 0 else Color.WHITE
    
    def view(self) -> str:
        """
        Returns a string representation of the board.

        Returns:
            str: The board as a string with rows separated by newlines.
        """
        lines = []
        for row in range(self.size):
            current_line = []
            for col in range(self.size):
                state = self.get_state((row, col))
                if state == 1:
                    current_line.append('X')  # BLACK
                elif state == 2:
                    current_line.append('O')  # WHITE
                elif state == 3:
                    current_line.append('*')  # Blocked or special state
                else:
                    current_line.append('.')  # Empty
            lines.append('  '.join(current_line))
        return '\n'.join(lines)

    def copy(self) -> 'BitBoard':
        new_board = BitBoard(self.size)
        new_board.bit_board = self.bit_board
        new_board.last_move = self.last_move
        new_board.move_count = self.move_count
        return new_board

# Candidate Move Generator with Priority
class Candidate:
    def __init__(self, mode=0, size=15, evaluator: Optional['Evaluator'] = None):
        self.mode = mode
        self.size = size
        self.evaluator = evaluator  # Inject Evaluator for scoring moves

    def _make_threat_mask(self, board: BitBoard, side: Color) -> int:
        """
        Create a threat mask based on pattern4 counts, similar to Rapfi’s makeThreatMask.
        Returns a binary mask indicating critical threats (e.g., opponent five, self flex four).
        """
        opposite = Color.WHITE if side == Color.BLACK else Color.BLACK
        patterns = self.evaluator.tracker.get_patterns(side)
        opp_patterns = self.evaluator.tracker.get_patterns(opposite)

        # Count patterns for self and opponent
        self_counts = {p: 0 for p in Pattern4}
        opp_counts = {p: 0 for p in Pattern4}
        for p in patterns.values():
            self_counts[p] += 1
        for p in opp_patterns.values():
            opp_counts[p] += 1

        mask = 0
        # Check for threats (bit positions match Rapfi’s logic, 0-based)
        mask |= 0b1 & -int(opp_counts[Pattern4.A_FIVE] > 0)  # Opponent five
        mask |= 0b10 & -int(self_counts[Pattern4.B_FLEX4] > 0)  # Self flex four
        mask |= 0b100 & -int(opp_counts[Pattern4.B_FLEX4] > 0)  # Opponent flex four
        mask |= 0b1000 & -int(self_counts[Pattern4.D_BLOCK4_PLUS] + self_counts[Pattern4.C_BLOCK4_FLEX3] > 0)  # Self four-plus
        mask |= 0b10000 & -int(self_counts[Pattern4.E_BLOCK4] > 0)  # Self four
        mask |= 0b100000 & -int(self_counts[Pattern4.G_FLEX3_PLUS] + self_counts[Pattern4.F_FLEX3_2X] > 0)  # Self three-plus
        mask |= 0b1000000 & -int(self_counts[Pattern4.H_FLEX3] > 0)  # Self three
        mask |= 0b10000000 & -int(opp_counts[Pattern4.D_BLOCK4_PLUS] + opp_counts[Pattern4.C_BLOCK4_FLEX3] > 0)  # Opponent four-plus
        mask |= 0b100000000 & -int(opp_counts[Pattern4.E_BLOCK4] > 0)  # Opponent four
        mask |= 0b1000000000 & -int(opp_counts[Pattern4.G_FLEX3_PLUS] + opp_counts[Pattern4.F_FLEX3_2X] > 0)  # Opponent three-plus
        mask |= 0b10000000000 & -int(opp_counts[Pattern4.H_FLEX3] > 0)  # Opponent three

        return mask

    def _evaluate_move_priority(self, board: BitBoard, move: Tuple[int, int], current_side: Color) -> float:
        """
        Evaluate move priority based on threats, patterns, and proximity, without arbitrary bonuses.
        """
        row, col = move
        priority = 0.0

        # Check adjacent positions for patterns and threats
        for dr, dc in [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, 1), (-1, -1)]:
            nr, nc = row + dr, col + dc
            if check_valid(board.size, (nr, nc)):
                state = board.get_state((nr, nc))
                if state in [1, 2]:  # Adjacent to Black or White stone
                    priority += 20.0
                    # Check pattern at adjacent position
                    pattern = self.evaluator.tracker.detector.get_combined_pattern(board, (nr, nc), current_side)
                    if pattern in [Pattern4.B_FLEX4, Pattern4.C_BLOCK4_FLEX3, Pattern4.E_BLOCK4, Pattern4.A_FIVE]:
                        priority += 100.0  # High priority for critical patterns
                    elif pattern in [Pattern4.D_BLOCK4_PLUS, Pattern4.G_FLEX3_PLUS, Pattern4.H_FLEX3]:
                        priority += 50.0  # Moderate priority for other threats
                elif state == 0:  # Adjacent to empty
                    priority += 2.0

        # Bonus for moves near the last move or center (natural positioning, not arbitrary)
        if board.last_move:
            last_row, last_col = board.last_move
            dist = abs(row - last_row) + abs(col - last_col)
            if dist <= 2:
                priority += 15.0
        center_row, center_col = self.size // 2, self.size // 2
        if abs(row - center_row) <= 2 and abs(col - center_col) <= 2:
            priority += 10.0

        # Use threat mask to boost priority for moves addressing critical threats
        threat_mask = self._make_threat_mask(board, current_side)
        if threat_mask & 0b1:  # Opponent five (urgent defense needed)
            if self.evaluator.tracker.get_patterns(Color.WHITE if current_side == Color.BLACK else Color.BLACK).get(move, Pattern4.NONE) in [Pattern4.E_BLOCK4, Pattern4.D_BLOCK4_PLUS]:
                priority += 200.0  # High priority to block opponent five
        if threat_mask & 0b10:  # Self flex four (extend to win)
            if self.evaluator.tracker.get_patterns(current_side).get(move, Pattern4.NONE) in [Pattern4.B_FLEX4, Pattern4.A_FIVE]:
                priority += 150.0  # High priority to extend to five

        return priority

    def expand(self, board: BitBoard) -> List[Tuple[int, int]]:
        """
        Return moves sorted by their threat-based priority, limited to top 8.
        """
        candidate = []
        marked_positions = []

        # Mark potential moves around existing stones (similar to Rapfi’s move generation)
        for row in range(self.size):
            for col in range(self.size):
                state = board.get_state((row, col))
                if state in (1, 2):  # Only for Black or White stones
                    self._square_line(board, row, col, 3, 4, marked_positions)

        # Use marked positions as candidate moves (strategic positions near existing stones)
        current_side = board.get_current_side()
        for pos in marked_positions:
            if board.get_state(pos) == 0:  # Ensure the position is still empty
                candidate.append(pos)

        # If no candidates or no evaluator, return default
        if not candidate or not self.evaluator:
            return candidate[:15] if candidate else [(self.size // 2, self.size // 2)]

        # Sort moves by priority (using threat-based evaluation)
        moves_with_priorities = []
        for move in candidate:
            priority = self._evaluate_move_priority(board, move, current_side)
            moves_with_priorities.append((move, priority))

        # Sort by priority in descending order (highest priority first)
        moves_with_priorities.sort(key=lambda x: x[1], reverse=True)
        sorted_moves = [move for move, _ in moves_with_priorities]

        # Reset marked positions to avoid corrupting the board state
        for pos in marked_positions:
            board.reset_pos(pos)

        # Limit to top 8 moves for efficiency (increased to cover more strategic options)
        return sorted_moves[:15] if sorted_moves else [(self.size // 2, self.size // 2)]

    def _square_line(self, board: BitBoard, x: int, y: int, sq: int, ln: int, marked_positions: List[Tuple[int, int]]):
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)]
        for k in range(1, ln + 1):
            for i, j in directions:
                coord = (x + i * k, y + j * k)
                if check_valid(board.size, coord) and board.get_state(coord) == 0:
                    marked_positions.append(coord)  # Mark without modifying bit_board
        for i in range(1, sq + 1):
            for j in range(1, sq + 1):
                coords = [(x + i, y + j), (x + i, y - j), (x - i, y + j), (x - i, y - j)]
                for coord in coords:
                    if check_valid(board.size, coord) and board.get_state(coord) == 0:
                        marked_positions.append(coord)

    def _circle_34(self, board: BitBoard, x: int, y: int, marked_positions: List[Tuple[int, int]]):
        cr34 = 34 ** 0.5
        for row in range(-int(cr34), int(cr34) + 1):
            for col in range(-int(cr34), int(cr34) + 1):
                if (row ** 2 + col ** 2) ** 0.5 <= cr34:
                    coord = (x + row, y + col)
                    if check_valid(board.size, coord) and board.get_state(coord) == 0:
                        marked_positions.append(coord)  # Mark without modifying bit_board

    def _full_board(self, board: BitBoard, marked_positions: List[Tuple[int, int]]):
        for row in range(board.size):
            for col in range(board.size):
                if board.get_state((row, col)) == 0:
                    marked_positions.append((row, col))  # Mark without modifying bit_board

# Pattern Detection
class PatternDetector:
    def __init__(self, rule: str = "STANDARD"):
        self.rule = rule

    @staticmethod
    def count_line(line: List[ColorFlag]) -> Tuple[int, int, int, int]:
        mid = len(line) // 2
        real_len, full_len = 1, 1
        real_len_inc = 1
        start, end = mid, mid
        for i in range(mid - 1, -1, -1):
            if line[i] == ColorFlag.SELF:
                real_len += real_len_inc
            elif line[i] == ColorFlag.OPPO:
                break
            else:
                real_len_inc = 0
            full_len += 1
            start = i
        real_len_inc = 1
        for i in range(mid + 1, len(line)):
            if line[i] == ColorFlag.SELF:
                real_len += real_len_inc
            elif line[i] == ColorFlag.OPPO:
                break
            else:
                real_len_inc = 0
            full_len += 1
            end = i
        return real_len, full_len, start, end

    @staticmethod
    @lru_cache(maxsize=None)
    def get_line_pattern(line_tuple: Tuple[ColorFlag, ...], rule: str, side: Color) -> Pattern:
        line = list(line_tuple)
        real_len, full_len, start, end = PatternDetector.count_line(line)

        if rule == "STANDARD":
            if real_len >= 6:
                return Pattern.OL
            elif real_len == 5:
                return Pattern.F5
            elif full_len < 5:
                return Pattern.DEAD

        pattern_counts = {p: 0 for p in Pattern}
        f5_indices = []
        for i in range(start, end + 1):
            if line[i] == ColorFlag.EMPT:
                new_line = line.copy()
                new_line[i] = ColorFlag.SELF
                new_pattern = PatternDetector.get_line_pattern(tuple(new_line), rule, side)
                pattern_counts[new_pattern] += 1
                if new_pattern == Pattern.F5 and len(f5_indices) < 2:
                    f5_indices.append(i)

        if pattern_counts[Pattern.F5] >= 2:
            return Pattern.F4
        elif pattern_counts[Pattern.F5] == 1:
            return Pattern.B4
        elif pattern_counts[Pattern.F4] >= 2:
            return Pattern.F3S
        elif pattern_counts[Pattern.F4] == 1:
            return Pattern.F3
        elif pattern_counts[Pattern.B4] >= 1:
            return Pattern.B3
        elif (pattern_counts[Pattern.F3S] + pattern_counts[Pattern.F3]) >= 4:
            return Pattern.F2B
        elif (pattern_counts[Pattern.F3S] + pattern_counts[Pattern.F3]) >= 3:
            return Pattern.F2A
        elif (pattern_counts[Pattern.F3S] + pattern_counts[Pattern.F3]) >= 1:
            return Pattern.F2
        elif pattern_counts[Pattern.B3] >= 1:
            return Pattern.B2
        elif (pattern_counts[Pattern.F2] + pattern_counts[Pattern.F2A] + pattern_counts[Pattern.F2B]) >= 1:
            return Pattern.F1
        elif pattern_counts[Pattern.B2] >= 1:
            return Pattern.B1
        return Pattern.DEAD

    def extract_line(self, board: BitBoard, move: Tuple[int, int], side: Color, dRow: int, dCol: int) -> List[ColorFlag]:
        size = board.size
        x, y = move
        line = []
        for i in range(-4, 5):
            r = x + dRow * i
            c = y + dCol * i
            if 0 <= r < size and 0 <= c < size:
                state = board.get_state((r, c))
                if state == side.value:
                    line.append(ColorFlag.SELF)
                elif state == 0:
                    line.append(ColorFlag.EMPT)
                else:
                    line.append(ColorFlag.OPPO)
            else:
                line.append(ColorFlag.OPPO)
        return line

    def get_combined_pattern(self, board: BitBoard, pos: Tuple[int, int], side: Color) -> Pattern4:
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        patterns = []
        for dRow, dCol in directions:
            line = self.extract_line(board, pos, side, dRow, dCol)
            if line:
                pattern = self.get_line_pattern(tuple(line), self.rule, side)
                patterns.append(pattern)
        while len(patterns) < 4:
            patterns.append(Pattern.DEAD)

        n = {p: patterns.count(p) for p in set(patterns)}
        if n.get(Pattern.F5, 0) >= 1:
            return Pattern4.A_FIVE
        if n.get(Pattern.OL, 0) >= 1:
            return Pattern4.NONE
        if n.get(Pattern.B4, 0) >= 2 or n.get(Pattern.F4, 0) >= 1:
            return Pattern4.B_FLEX4
        if n.get(Pattern.B4, 0) >= 1:
            if (n.get(Pattern.F3, 0) + n.get(Pattern.F3S, 0)) >= 1:
                return Pattern4.C_BLOCK4_FLEX3
            if n.get(Pattern.B3, 0) >= 1 or (n.get(Pattern.F2, 0) + n.get(Pattern.F2A, 0) + n.get(Pattern.F2B, 0)) >= 1:
                return Pattern4.D_BLOCK4_PLUS
            return Pattern4.E_BLOCK4
        if (n.get(Pattern.F3, 0) + n.get(Pattern.F3S, 0)) >= 1:
            if (n.get(Pattern.F3, 0) + n.get(Pattern.F3S, 0)) >= 2:
                return Pattern4.F_FLEX3_2X
            if n.get(Pattern.B3, 0) >= 1 or (n.get(Pattern.F2, 0) + n.get(Pattern.F2A, 0) + n.get(Pattern.F2B, 0)) >= 1:
                return Pattern4.G_FLEX3_PLUS
            return Pattern4.H_FLEX3
        if n.get(Pattern.B3, 0) >= 1:
            if n.get(Pattern.B3, 0) >= 2 or (n.get(Pattern.F2, 0) + n.get(Pattern.F2A, 0) + n.get(Pattern.F2B, 0)) >= 1:
                return Pattern4.I_BLOCK3_PLUS
            return Pattern4.K_BLOCK3
        if (n.get(Pattern.F2, 0) + n.get(Pattern.F2A, 0) + n.get(Pattern.F2B, 0)) >= 2:
            return Pattern4.J_FLEX2_2X
        if (n.get(Pattern.F2, 0) + n.get(Pattern.F2A, 0) + n.get(Pattern.F2B, 0)) >= 1:
            return Pattern4.L_FLEX2
        return Pattern4.NONE

# Pattern Tracker
class PatternTracker:
    def __init__(self, board_size: int):
        self.board_size = board_size
        self.patterns = {Color.BLACK: {}, Color.WHITE: {}}
        self.detector = PatternDetector("STANDARD")

    def update(self, board: BitBoard, move: Tuple[int, int]):
        for side in [Color.BLACK, Color.WHITE]:
            for dy in range(-4, 5):
                for dx in range(-4, 5):
                    pos = (move[0] + dy, move[1] + dx)
                    if check_valid(self.board_size, pos):
                        if board.get_state(pos) == side.value:
                            self.patterns[side][pos] = self.detector.get_combined_pattern(board, pos, side)

    def get_patterns(self, side: Color) -> Dict[Tuple[int, int], Pattern4]:
        return self.patterns[side]
    

class TreeNode:
    def __init__(self, board_state: BitBoard, hash_val: int, best_move: Optional[Tuple[int, int]] = None):
        self.board_state = board_state
        self.hash_val = hash_val
        self.best_move = best_move
        self.score = None
        self.pv = []

class Evaluator:
    def __init__(self, board_size: int):
        self.tracker = PatternTracker(board_size)
        self.weights = {
            Pattern4.A_FIVE: 50000,  # Highest priority for winning
            Pattern4.B_FLEX4: 20000,  # High priority for near-winning moves
            Pattern4.C_BLOCK4_FLEX3: 10000,
            Pattern4.D_BLOCK4_PLUS: 5000,
            Pattern4.E_BLOCK4: 3000,  # Prioritize blocking opponent threats
            Pattern4.F_FLEX3_2X: 1000,
            Pattern4.G_FLEX3_PLUS: 500,
            Pattern4.H_FLEX3: 200,
            Pattern4.I_BLOCK3_PLUS: 100,
            Pattern4.J_FLEX2_2X: 50,
            Pattern4.K_BLOCK3: 30,
            Pattern4.L_FLEX2: 10,
            Pattern4.NONE: 0
        }

    def _make_threat_mask(self, board: BitBoard, side: Color) -> int:
        """
        Create a threat mask similar to Rapfi’s makeThreatMask, using pattern counts.
        """
        opposite = Color.WHITE if side == Color.BLACK else Color.BLACK
        patterns = self.tracker.get_patterns(side)
        opp_patterns = self.tracker.get_patterns(opposite)

        self_counts = {p: 0 for p in Pattern4}
        opp_counts = {p: 0 for p in Pattern4}
        for p in patterns.values():
            self_counts[p] += 1
        for p in opp_patterns.values():
            opp_counts[p] += 1

        mask = 0
        mask |= 0b1 & -int(opp_counts[Pattern4.A_FIVE] > 0)  # Opponent five
        mask |= 0b10 & -int(self_counts[Pattern4.B_FLEX4] > 0)  # Self flex four
        mask |= 0b100 & -int(opp_counts[Pattern4.B_FLEX4] > 0)  # Opponent flex four
        mask |= 0b1000 & -int(self_counts[Pattern4.D_BLOCK4_PLUS] + self_counts[Pattern4.C_BLOCK4_FLEX3] > 0)  # Self four-plus
        mask |= 0b10000 & -int(self_counts[Pattern4.E_BLOCK4] > 0)  # Self four
        mask |= 0b100000 & -int(self_counts[Pattern4.G_FLEX3_PLUS] + self_counts[Pattern4.F_FLEX3_2X] > 0)  # Self three-plus
        mask |= 0b1000000 & -int(self_counts[Pattern4.H_FLEX3] > 0)  # Self three
        mask |= 0b10000000 & -int(opp_counts[Pattern4.D_BLOCK4_PLUS] + opp_counts[Pattern4.C_BLOCK4_FLEX3] > 0)  # Opponent four-plus
        mask |= 0b100000000 & -int(opp_counts[Pattern4.E_BLOCK4] > 0)  # Opponent four
        mask |= 0b1000000000 & -int(opp_counts[Pattern4.G_FLEX3_PLUS] + opp_counts[Pattern4.F_FLEX3_2X] > 0)  # Opponent three-plus
        mask |= 0b10000000000 & -int(opp_counts[Pattern4.H_FLEX3] > 0)  # Opponent three

        return mask

    def evaluate(self, board: BitBoard, move: Tuple[int, int], ai_color: Color) -> float:
        """
        Evaluate the board state using basic patterns and threat adjustments, inspired by Rapfi.
        """
        self.tracker.update(board, move)
        ai_patterns = self.tracker.get_patterns(ai_color)
        opp_patterns = self.tracker.get_patterns(Color.WHITE if ai_color == Color.BLACK else Color.BLACK)

        # Basic pattern evaluation (sum of pattern weights)
        ai_score = sum(self.weights.get(p, 0) for p in ai_patterns.values())
        opp_score = sum(self.weights.get(p, 0) for p in opp_patterns.values())

        # Threat-based evaluation (Rapfi-style)
        threat_mask = self._make_threat_mask(board, ai_color)
        threat_eval = 0.0
        if threat_mask & 0b1:  # Opponent five (urgent defense needed)
            # Prioritize blocking opponent’s five-in-a-row
            if ai_patterns.get(move, Pattern4.NONE) in [Pattern4.E_BLOCK4, Pattern4.D_BLOCK4_PLUS]:
                threat_eval += 10000.0  # High value for blocking opponent five
        if threat_mask & 0b10:  # Self flex four (extend to win)
            # Prioritize extending to five-in-a-row
            if ai_patterns.get(move, Pattern4.NONE) in [Pattern4.B_FLEX4, Pattern4.A_FIVE]:
                threat_eval += 15000.0  # High value for extending to win

        # Combine basic and threat evaluations
        total_score = ai_score - opp_score + threat_eval
        return total_score

# Optimized Search with Score-Based Move Ordering
class Search:
    def __init__(self, ai_color: Color = Color.BLACK):
        self.evaluator = Evaluator(15)
        self.ai_color = ai_color
        self.transposition_table = {}
        self.nodes_visited = 0

    def get_opponent_side(self, current_side: Color) -> Color:
        return Color.WHITE if current_side == Color.BLACK else Color.BLACK

    def get_possible_moves(self, board: BitBoard) -> List[Tuple[int, int]]:
        """
        Get moves sorted by threat-based priority.
        """
        candidate = Candidate(mode=0, size=board.size, evaluator=self.evaluator)
        return candidate.expand(board)

    def _should_extend(self, board: BitBoard, move: Tuple[int, int], current_side: Color, depth: int) -> bool:
        """
        Determine if a move should be extended based on patterns and threats, inspired by Rapfi’s extensions.
        """
        if depth <= 0:
            return False
        pattern = self.evaluator.tracker.detector.get_combined_pattern(board, move, current_side)
        threat_mask = self.evaluator._make_threat_mask(board, current_side)
        if threat_mask & 0b1:  # Opponent five (extend to defend)
            if pattern in [Pattern4.E_BLOCK4, Pattern4.D_BLOCK4_PLUS]:
                return True
        if threat_mask & 0b10:  # Self flex four (extend to win)
            if pattern in [Pattern4.B_FLEX4, Pattern4.A_FIVE]:
                return True
        if pattern in [Pattern4.B_FLEX4, Pattern4.C_BLOCK4_FLEX3, Pattern4.E_BLOCK4]:
            return True  # Extend for critical patterns
        return False

    def alphabeta(self, node: TreeNode, depth: int, alpha: float, beta: float, current_side: Color, max_depth: int) -> Tuple[float, int]:
        """
        Optimized alpha-beta pruning with threat-based extensions and pruning, inspired by Rapfi.
        """
        self.nodes_visited += 1
        if depth == 0 or node.board_state.is_win(self.ai_color.value) or node.board_state.is_win(self.get_opponent_side(self.ai_color).value):
            score = self.evaluator.evaluate(node.board_state, node.best_move or (7, 7), current_side)
            return score, 1  # Return score and 1 node for terminal evaluation

        # Dynamic depth reduction for less promising positions (Rapfi-style futility)
        if depth > 2 and not node.best_move and not self._has_strong_threat(node.board_state, current_side):
            depth = max(1, depth - 1)

        extend = False
        if node.best_move and self._should_extend(node.board_state, node.best_move, current_side, depth):
            extend = True

        if node.hash_val in self.transposition_table and self.transposition_table[node.hash_val][0] >= depth:
            return self.transposition_table[node.hash_val][1], 0  # No new nodes visited for TT hit

        possible_moves = self.get_possible_moves(node.board_state)
        if not possible_moves:
            return 0.0, 1  # Return 1 node for this evaluation

        is_maximizing = (current_side == self.ai_color)
        value = float('-inf') if is_maximizing else float('inf')
        best_move_here = None
        nodes = 1  # Count this node

        # Rapfi-style move pruning: skip trivial or low-priority moves
        for move in possible_moves:  # Limit to top 8 moves for efficiency
            child_board = node.board_state.copy()
            if not child_board.add_move(move, current_side.value):
                continue

            # Rapfi-style futility pruning: skip moves unlikely to improve alpha
            static_eval = self.evaluator.evaluate(child_board, move, current_side)
            if not is_maximizing and static_eval < alpha - 200.0 and depth < 3:  # Adjust margin as needed
                continue
            if is_maximizing and static_eval > beta + 200.0 and depth < 3:
                continue

            child_node = TreeNode(child_board, child_board.hash(), best_move=move)
            search_depth = depth - 1 if not extend else min(depth + 1, max_depth)  # Extend depth for threats
            score, child_nodes = self.alphabeta(child_node, search_depth, -beta, -alpha, self.get_opponent_side(current_side), max_depth)
            nodes += child_nodes

            if is_maximizing:
                if score > value:
                    value = score
                    best_move_here = move
                    node.pv = [move] + child_node.pv
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            else:
                if score < value:
                    value = score
                    best_move_here = move
                    node.pv = [move] + child_node.pv
                beta = min(beta, value)
                if alpha >= beta:
                    break

        node.best_move = best_move_here
        node.score = value
        self.transposition_table[node.hash_val] = (depth, value)
        return value, nodes

    def _has_strong_threat(self, board: BitBoard, side: Color) -> bool:
        """
        Check if there’s a strong threat (e.g., self flex four, opponent five) to justify deeper search.
        """
        threat_mask = self.evaluator._make_threat_mask(board, side)
        return bool(threat_mask & (0b1 | 0b10))  # Opponent five or self flex four

    def parallel_alphabeta(self, board: BitBoard, move: Tuple[int, int], depth: int, alpha: float, beta: float, current_side: Color, max_depth: int) -> Tuple[float, List[Tuple[int, int]], int]:
        """
        Optimized parallel evaluation, returning (score, pv, nodes_visited).
        """
        child_board = board.copy()
        if not child_board.add_move(move, current_side.value):
            return float('-inf'), [], 0
        child_node = TreeNode(child_board, child_board.hash(), best_move=move)
        score, nodes = self.alphabeta(child_node, depth - 1, -beta, -alpha, self.get_opponent_side(current_side), max_depth)
        return score, child_node.pv, nodes

    def ids_search(self, board: BitBoard, max_depth: int, max_time: float) -> Tuple[Optional[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Optimized IDS with threat-based extensions and pruning.
        """
        logger.info(f"Starting IDS search, Max Depth: {max_depth}, Max Time: {max_time}s")
        start_time = time.time()
        best_move = None
        best_score = float('-inf')
        best_pv = []
        current_side = board.get_current_side()

        # Dynamic max_depth based on threats and game stage
        effective_max_depth = min(max_depth, 7 + board.move_count // 5)
        if self._has_strong_threat(board, current_side):
            effective_max_depth = min(effective_max_depth + 2, max_depth)  # Extend for threats

        for depth in range(1, effective_max_depth + 1):
            self.nodes_visited = 0
            self.transposition_table.clear()
            iteration_start = time.time()
            root = TreeNode(board, board.hash())
            possible_moves = self.get_possible_moves(board)
            if not possible_moves:
                break

            with ProcessPoolExecutor(max_workers=min(4, len(possible_moves))) as executor:
                futures = {}
                for move in possible_moves[:8]:  # Limit to top 8 moves for efficiency
                    futures[executor.submit(self.parallel_alphabeta, board, move, depth + 1, float('-inf'), float('inf'), current_side, effective_max_depth)] = move

                iteration_score = float('-inf')
                iteration_move = None
                iteration_pv = []
                max_self_depth = depth
                total_nodes = 0

                for future in as_completed(futures):
                    move = futures[future]
                    try:
                        score, pv, nodes = future.result()
                        self.nodes_visited += nodes
                        total_nodes += nodes
                        self_depth = depth + len(pv)
                        max_self_depth = max(max_self_depth, self_depth)
                        if score > iteration_score:
                            iteration_score = score
                            iteration_move = move
                            iteration_pv = [move] + pv
                    except Exception as e:
                        logger.error(f"Error evaluating move {move}: {e}")

                if iteration_move and iteration_score > best_score:
                    best_score = iteration_score
                    best_move = iteration_move
                    best_pv = iteration_pv

            elapsed = time.time() - start_time
            nps = int(total_nodes / (time.time() - iteration_start)) if time.time() > iteration_start else 0
            ev = best_score if current_side == self.ai_color else -best_score
            logger.info(f"DEPTH {depth}-{max_self_depth} EV {int(ev)} N {self.nodes_visited} NPS {nps} TM {int(elapsed)} PV {' '.join(pos_to_notation(m) for m in best_pv)}")

            if elapsed >= max_time:
                logger.info(f"Time limit reached at depth {depth}")
                break

        return best_move, best_pv

# Main Game Loop
def main():
    board = BitBoard(15)
    board.add_move((7, 7), 1)
    ai = Search(ai_color=Color.BLACK)
    player_color = Color.WHITE
    print("Gomoku Game - You are 'O' (White), AI is 'X' (Black)")

    moves = [
        ((7, 7), Color.BLACK.value),   # Black
        ((7, 6), Color.WHITE.value),   # White
        ((6, 6), Color.BLACK.value),   # Black
        ((5, 5), Color.WHITE.value),   # White
        ((6, 8), Color.BLACK.value),   # Black
        ((6, 5), Color.WHITE.value),   # White
        ((6, 7), Color.BLACK.value),   # Black
        ((5, 7), Color.WHITE.value),   # White
    ]

    # Apply the moves to the board
    for move, color in moves:
        success = board.add_move(move, color)
        if not success:
            print(f"Failed to add move at {move} for {Color(color).name}.")

    while True:
        print("\nCurrent Board:")
        print(board.view())

        # AI Move
        if board.get_current_side() == Color.BLACK:
            print("AI thinking...")
            start_time = time.time()
            move, pv = ai.ids_search(board, max_depth=5, max_time=60)
            if move:
                board.add_move(move, Color.BLACK.value)
                print(f"AI moved to {move} in {time.time() - start_time:.2f}s, PV: {' '.join(pos_to_notation(m) for m in pv)}")
            else:
                print("AI has no valid move!")
                break
            if board.is_win(Color.BLACK.value):
                print("\nFinal Board:")
                print(board.view())
                print("AI (Black) wins!")
                break

        # Player Move
        else:
            move = input("Enter your move (row, col) e.g., '7,7': ").strip().split(',')
            move = (int(move[0]), int(move[1]))
            if board.add_move(move, Color.WHITE.value):
                if board.is_win(Color.WHITE.value):
                    print("\nFinal Board:")
                    print(board.view())
                    print("You (White) win!")
                    break
            else:
                print("Invalid move, try again.")

if __name__ == "__main__":
    main()