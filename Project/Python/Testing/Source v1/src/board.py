import numpy as np
from .types import Pos, Color, Rule, Value, Pattern, Pattern4
from .constants import MAX_BOARD_SIZE, BOARD_BOUNDARY, FULL_BOARD_SIZE, FULL_BOARD_CELL_COUNT, FULL_BOARD_START, FULL_BOARD_END, DIRECTIONS, MAX_FIND_DIST
from .config import Hash, Config
from .pattern import PatternConfig

# Set Pos constants
Pos.NONE = Pos(-BOARD_BOUNDARY, -BOARD_BOUNDARY)
Pos.NONE_INDEX = FULL_BOARD_CELL_COUNT
Pos.FULL_BOARD_START = FULL_BOARD_START
Pos.FULL_BOARD_END = FULL_BOARD_END

# Candidate range arrays
RANGE_SQUARE2 = [Pos(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)]  # 25 positions
RANGE_SQUARE2_LINE3 = (
    [Pos(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)] +
    [Pos(-3, 0), Pos(3, 0), Pos(0, -3), Pos(0, 3)] +
    [Pos(-2, -3), Pos(-2, 3), Pos(2, -3), Pos(2, 3), Pos(-3, -2), Pos(-3, 2), Pos(3, -2), Pos(3, 2)]
)  # 37 positions
RANGE_SQUARE3 = [Pos(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4)]  # 49 positions
RANGE_SQUARE3_LINE4 = (
    [Pos(dx, dy) for dx in range(-3, 4) for dy in range(-3, 4)] +
    [Pos(-4, 0), Pos(4, 0), Pos(0, -4), Pos(0, 4)] +
    [Pos(-3, -4), Pos(-3, 4), Pos(3, -4), Pos(3, 4), Pos(-4, -3), Pos(-4, 3), Pos(4, -3), Pos(4, 3)]
)  # 61 positions
RANGE_SQUARE4 = [Pos(dx, dy) for dx in range(-4, 5) for dy in range(-4, 5)]  # 81 positions

class Cell:
    """Matches Cell struct in board.h with added attributes."""
    def __init__(self):
        self.piece = Color.EMPTY
        self.pattern = [Pattern.DEAD] * 4  # 4 directions
        self.pattern2x = [PatternConfig.PATTERN2x[Rule.FREESTYLE][0]] * 4  # Default Pattern2x
        self.pattern4 = [Pattern4.NONE] * Color.COLOR_NB  # BLACK, WHITE
        self.score = [Value(0)] * Color.COLOR_NB  # BLACK, WHITE scores
        self.value_black = Value(0)  # Value from Black's perspective
        self.cand = 0  # Candidate counter

    def pcode(self, side):
        """Matches pcode<Side>()—uses first direction’s pattern."""
        return self.pattern2x[0].pat(side)

    def pattern(self, side, dir):
        """Matches pattern(Side, dir)."""
        return self.pattern2x[dir].pat(side)

    def pat(self, side):
        """Helper for pattern4 access."""
        return self.pattern4[side] if self.piece == Color.EMPTY else Pattern4.NONE

    def update_pattern4_and_score(self, rule, pcode_black, pcode_white):
        """Matches updatePattern4AndScore<R>."""
        if rule == Rule.RENJU and self.piece == Color.EMPTY and pcode_black >= Pattern.FORBID:
            self.pattern4[Color.BLACK] = Pattern4.FORBID
            self.pattern4[Color.WHITE] = PatternConfig.pattern4_from_pcode(pcode_white)
        else:
            self.pattern4[Color.BLACK] = PatternConfig.pattern4_from_pcode(pcode_black)
            self.pattern4[Color.WHITE] = PatternConfig.pattern4_from_pcode(pcode_white)
        self.score[Color.BLACK] = Config.get_score(rule, Color.BLACK, pcode_black, pcode_white)
        self.score[Color.WHITE] = Config.get_score(rule, Color.WHITE, pcode_black, pcode_white)
        self.value_black = Config.get_value_black(rule, pcode_black, pcode_white)

    def is_candidate(self):
        """Matches isCandidate()."""
        return self.cand > 0

class StateInfo:
    """Matches StateInfo struct in board.h."""
    def __init__(self):
        self.p4_count = [[0] * Pattern4.PATTERN4_NB for _ in range(Color.COLOR_NB)]
        self.last_move = Pos.NONE
        self.value_black = Value(0)
        self.last_pattern4_move = [
            [Pos.NONE] * (Pattern4.PATTERN4_NB - Pattern4.C_BLOCK4_FLEX3) for _ in range(Color.COLOR_NB)
        ]
        self.last_flex4_attack_move = [Pos.NONE, Pos.NONE]  # BLACK, WHITE
        self.cand_area = CandArea()

    def last_pattern4(self, side, p4):
        """Matches lastPattern4."""
        if p4 >= Pattern4.C_BLOCK4_FLEX3:
            return self.last_pattern4_move[side][p4 - Pattern4.C_BLOCK4_FLEX3]
        return Pos.NONE

class CandArea:
    """Matches CandArea struct in board.h."""
    def __init__(self):
        self.min_x = 0
        self.max_x = 0
        self.min_y = 0
        self.max_y = 0

    def expand(self, pos, board_size, dist):
        """Matches expand()."""
        self.min_x = max(0, pos.x - dist)
        self.max_x = min(board_size - 1, pos.x + dist)
        self.min_y = max(0, pos.y - dist)
        self.max_y = min(board_size - 1, pos.y + dist)

class UpdateCache:
    """Matches UpdateCache tuple in board.h."""
    def __init__(self):
        self.pattern4 = [Pattern4.NONE, Pattern4.NONE]
        self.score = [Value(0), Value(0)]
        self.value_black = Value(0)

class Board:
    """Matches Board class in board.h/cpp."""
    def __init__(self, board_size, cand_range=0):
        assert 0 < board_size <= MAX_BOARD_SIZE
        self.board_size = board_size
        self.board_cell_count = board_size * board_size
        self.move_count = 0
        self.pass_count = [0, 0]  # BLACK, WHITE
        self.current_side = Color.BLACK
        self.current_zobrist_key = np.uint64(0)
        self.rule = None  # Game rule
        self.candidate_range = []
        self.candidate_range_size = 0
        self.cand_area_expand_dist = 0
        self.evaluator = None
        self.this_thread = None

        self.cells = np.array([Cell() for _ in range(FULL_BOARD_CELL_COUNT)], dtype=object)
        self.bit_key0 = np.zeros(FULL_BOARD_SIZE, dtype=np.uint64)
        self.bit_key1 = np.zeros(FULL_BOARD_SIZE, dtype=np.uint64)
        self.bit_key2 = np.zeros(2 * FULL_BOARD_SIZE - 1, dtype=np.uint64)
        self.bit_key3 = np.zeros(2 * FULL_BOARD_SIZE - 1, dtype=np.uint64)
        self.state_infos = [StateInfo() for _ in range(1 + self.board_cell_count * 2)]
        self.update_cache = [UpdateCache() for _ in range(1 + self.board_cell_count * 2)]

        # Set candidate range
        if cand_range == 0:
            self.candidate_range = []
            self.candidate_range_size = 0
            self.cand_area_expand_dist = 0
        elif cand_range == 1:
            self.candidate_range = RANGE_SQUARE2
            self.candidate_range_size = len(RANGE_SQUARE2)
            self.cand_area_expand_dist = 2
        elif cand_range == 2:
            self.candidate_range = RANGE_SQUARE2_LINE3
            self.candidate_range_size = len(RANGE_SQUARE2_LINE3)
            self.cand_area_expand_dist = 3
        elif cand_range == 3:
            self.candidate_range = RANGE_SQUARE3
            self.candidate_range_size = len(RANGE_SQUARE3)
            self.cand_area_expand_dist = 3
        elif cand_range == 4:
            self.candidate_range = RANGE_SQUARE3_LINE4
            self.candidate_range_size = len(RANGE_SQUARE3_LINE4)
            self.cand_area_expand_dist = 3
        elif cand_range == 5:
            self.candidate_range = RANGE_SQUARE4
            self.candidate_range_size = len(RANGE_SQUARE4)
            self.cand_area_expand_dist = 4

    def new_game(self, rule):
        """Matches newGame<R> in board.cpp."""
        self.rule = rule
        for i in range(FULL_BOARD_CELL_COUNT):
            self.cells[i] = Cell()
        self.bit_key0.fill(0)
        self.bit_key1.fill(0)
        self.bit_key2.fill(0)
        self.bit_key3.fill(0)

        self.move_count = 0
        self.pass_count = [0, 0]
        self.current_side = Color.BLACK
        self.current_zobrist_key = np.uint64(0)

        for i in range(FULL_BOARD_START, FULL_BOARD_END + 1):
            pos = Pos.from_index(i)
            self.cells[i].piece = Color.EMPTY if pos.is_in_board(self.board_size, self.board_size) else Color.WALL
            if self.cells[i].piece == Color.EMPTY:
                self._set_bit_key(pos, Color.BLACK)
                self._set_bit_key(pos, Color.WHITE)

        st = self.state_infos[0]
        st.p4_count = [[0] * Pattern4.PATTERN4_NB for _ in range(Color.COLOR_NB)]
        st.last_move = Pos.NONE
        st.value_black = Value(0)
        st.last_pattern4_move = [
            [Pos.NONE] * (Pattern4.PATTERN4_NB - Pattern4.C_BLOCK4_FLEX3) for _ in range(Color.COLOR_NB)
        ]
        st.last_flex4_attack_move = [Pos.NONE, Pos.NONE]
        st.cand_area = CandArea()

        value_black = Value(0)
        for x in range(self.board_size):
            for y in range(self.board_size):
                pos = Pos(x, y)
                idx = pos.to_index()
                c = self.cells[idx]
                for dir in range(4):
                    c.pattern2x[dir] = PatternConfig.lookup_pattern(rule, self._get_key_at(pos, dir, rule))
                pcode = [
                    PatternConfig.PCODE[c.pattern2x[0].pat(Color.BLACK), c.pattern2x[1].pat(Color.BLACK),
                                        c.pattern2x[2].pat(Color.BLACK), c.pattern2x[3].pat(Color.BLACK)],
                    PatternConfig.PCODE[c.pattern2x[0].pat(Color.WHITE), c.pattern2x[1].pat(Color.WHITE),
                                        c.pattern2x[2].pat(Color.WHITE), c.pattern2x[3].pat(Color.WHITE)]
                ]
                c.update_pattern4_and_score(rule, pcode[Color.BLACK], pcode[Color.WHITE])
                st.p4_count[Color.BLACK][c.pattern4[Color.BLACK]] += 1
                st.p4_count[Color.WHITE][c.pattern4[Color.WHITE]] += 1
                value_black += c.value_black
        st.value_black = value_black

        if self.candidate_range_size == 0:
            self.expand_cand_area(self.center_pos(), self.board_size // 2, 0)

        if self.evaluator:
            self.evaluator.init_empty_board()

    def make_move(self, pos, rule=None):
        """Matches move<R, MoveType::NORMAL> in board.cpp."""
        rule = self.rule if rule is None else rule
        if pos == Pos.NONE:
            assert self.pass_move_count() < self.board_cell_count
            self.move_count += 1
            st = self.state_infos[self.move_count]
            st.__dict__.update(self.state_infos[self.move_count - 1].__dict__)
            st.last_move = Pos.NONE
            self.pass_count[self.current_side] += 1
            self.current_side = Color.opposite(self.current_side)
            if self.evaluator:
                self.evaluator.after_pass(self)
            return True

        assert pos.is_valid() and self.is_empty(pos)
        if self.evaluator:
            self.evaluator.before_move(self, pos)

        self.move_count += 1
        pc = self.update_cache[self.move_count - 1]
        st = self.state_infos[self.move_count]
        st.__dict__.update(self.state_infos[self.move_count - 1].__dict__)
        st.last_move = pos
        st.cand_area.expand(pos, self.board_size, self.cand_area_expand_dist)

        # Place the stone
        self.cells[pos.to_index()].piece = self.current_side
        self.current_zobrist_key ^= Hash.zobrist[self.current_side][pos.to_index()]
        self._flip_bit_key(pos, self.current_side)

        delta_value_black = Value(0)
        f4_count_before_move = [self.p4_count(Color.BLACK, Pattern4.B_FLEX4),
                                self.p4_count(Color.WHITE, Pattern4.B_FLEX4)]

        half_len = PatternConfig.half_line_len(rule)
        affected_positions = set()
        for i in range(-half_len, half_len + 1):
            if i == 0:  # Skip the move position here
                continue
            for dir in range(4):
                dx, dy = DIRECTIONS[dir]
                posi = Pos(pos.x + dx * i, pos.y + dy * i)
                idx = posi.to_index()
                if not (0 <= idx < FULL_BOARD_CELL_COUNT) or self.cells[idx].piece != Color.EMPTY:
                    continue
                affected_positions.add(idx)

        update_cache_idx = 0
        for idx in affected_positions.union({pos.to_index()}):
            c = self.cells[idx]
            old_black_p4 = c.pattern4[Color.BLACK]
            old_white_p4 = c.pattern4[Color.WHITE]
            delta_value_black -= c.value_black

            for dir in range(4):
                posi = Pos.from_index(idx)
                key = self._get_key_at(posi, dir, rule)
                c.pattern2x[dir] = PatternConfig.lookup_pattern(rule, key)

            pcode = [
                PatternConfig.PCODE[c.pattern2x[0].pat(Color.BLACK), c.pattern2x[1].pat(Color.BLACK),
                                    c.pattern2x[2].pat(Color.BLACK), c.pattern2x[3].pat(Color.BLACK)],
                PatternConfig.PCODE[c.pattern2x[0].pat(Color.WHITE), c.pattern2x[1].pat(Color.WHITE),
                                    c.pattern2x[2].pat(Color.WHITE), c.pattern2x[3].pat(Color.WHITE)]
            ]
            # Debug pcode and pattern4
            if idx == pos.to_index():
                print(f"Debug Move: Pos {posi}, pcode: {pcode}, Black patterns: {[c.pattern2x[d].pat(Color.BLACK) for d in range(4)]}")

            c.update_pattern4_and_score(rule, pcode[Color.BLACK], pcode[Color.WHITE])

            if idx != pos.to_index():
                st.p4_count[Color.BLACK][old_black_p4] -= 1
                st.p4_count[Color.WHITE][old_white_p4] -= 1
            new_black_p4 = c.pattern4[Color.BLACK]
            new_white_p4 = c.pattern4[Color.WHITE]
            st.p4_count[Color.BLACK][new_black_p4] += 1
            st.p4_count[Color.WHITE][new_white_p4] += 1
            delta_value_black += c.value_black

            if idx != pos.to_index():
                pc.pattern4[Color.BLACK] = old_black_p4
                pc.pattern4[Color.WHITE] = old_white_p4
                pc.score[Color.BLACK] = c.score[Color.BLACK]
                pc.score[Color.WHITE] = c.score[Color.WHITE]
                pc.value_black = c.value_black
                self.update_cache[update_cache_idx] = pc
                update_cache_idx += 1

            if new_black_p4 >= Pattern4.C_BLOCK4_FLEX3:
                st.last_pattern4_move[Color.BLACK][new_black_p4 - Pattern4.C_BLOCK4_FLEX3] = posi
            if new_white_p4 >= Pattern4.C_BLOCK4_FLEX3:
                st.last_pattern4_move[Color.WHITE][new_white_p4 - Pattern4.C_BLOCK4_FLEX3] = posi

        st.value_black += delta_value_black

        self.current_side = Color.opposite(self.current_side)

        for offset in self.candidate_range:
            cand_idx = (pos + offset).to_index()
            if 0 <= cand_idx < FULL_BOARD_CELL_COUNT and self.is_empty(pos + offset):
                self.cells[cand_idx].cand += 1

        for side in [Color.BLACK, Color.WHITE]:
            if not f4_count_before_move[side] and self.p4_count(side, Pattern4.B_FLEX4):
                st.last_flex4_attack_move[side] = pos

        if self.evaluator:
            self.evaluator.after_move(self, pos)
        return True

    def undo_move(self, rule=None):
        """Matches undo<R, MoveType::NORMAL> in board.cpp."""
        rule = self.rule if rule is None else rule
        assert self.move_count > 0
        last_pos = self.get_last_move()

        if last_pos == Pos.NONE:  # Pass move
            self.current_side = Color.opposite(self.current_side)
            assert self.pass_count[self.current_side] > 0
            self.pass_count[self.current_side] -= 1
            self.move_count -= 1
            if self.evaluator:
                self.evaluator.after_undo_pass(self)
            return

        assert last_pos.is_valid() and self.get(last_pos) == self.current_side
        if self.evaluator:
            self.evaluator.before_undo(self, last_pos)

        self.current_side = Color.opposite(self.current_side)
        self._flip_bit_key(last_pos, self.current_side)
        self.current_zobrist_key ^= Hash.zobrist[self.current_side][last_pos.to_index()]
        self.cells[last_pos.to_index()].piece = Color.EMPTY

        self.move_count -= 1
        pc = self.update_cache[self.move_count]
        update_cache_idx = 0

        x = last_pos.x + BOARD_BOUNDARY
        y = last_pos.y + BOARD_BOUNDARY
        half_len = PatternConfig.half_line_len(rule)
        bit_key = [
            self.bit_key0[y],
            self.bit_key1[x],
            self.bit_key2[x + y],
            self.bit_key3[FULL_BOARD_SIZE - 1 + x - y]
        ]

        for i in range(-half_len, half_len + 1):
            if i == -1:
                continue
            for dir in range(4):
                dx, dy = DIRECTIONS[dir]
                posi = Pos(last_pos.x + dx * i, last_pos.y + dy * i)
                idx = posi.to_index()
                if not (0 <= idx < FULL_BOARD_CELL_COUNT) or self.cells[idx].piece != Color.EMPTY:
                    continue

                c = self.cells[idx]
                shift = np.uint64(2 * (i + half_len))
                c.pattern2x[dir] = PatternConfig.lookup_pattern(rule, np.right_shift(bit_key[dir], shift))
                c.pattern4[Color.BLACK] = pc.pattern4[Color.BLACK]
                c.pattern4[Color.WHITE] = pc.pattern4[Color.WHITE]
                c.score[Color.BLACK] = pc.score[Color.BLACK]
                c.score[Color.WHITE] = pc.score[Color.WHITE]
                c.value_black = pc.value_black
                update_cache_idx += 1

        for offset in self.candidate_range:
            cand_idx = (last_pos + offset).to_index()
            if 0 <= cand_idx < FULL_BOARD_CELL_COUNT and self.cells[cand_idx].cand > 0:
                self.cells[cand_idx].cand -= 1

        if self.evaluator:
            self.evaluator.after_undo(self, last_pos)

    def check_forbidden_point(self, pos):
        """Matches checkForbiddenPoint in board.cpp."""
        fp_cell = self.cells[pos.to_index()]
        if fp_cell.pattern4[Color.BLACK] != Pattern4.FORBID:
            return False

        win_by_four = 0
        for dir in range(4):
            pat_black = fp_cell.pattern2x[dir].pat(Color.BLACK)
            if pat_black == Pattern.OL:
                return True
            elif pat_black in (Pattern.B4, Pattern.F4):
                win_by_four += 1
                if win_by_four >= 2:
                    return True

        prev_side = self.current_side
        self.current_side = Color.BLACK
        self.make_move(pos, self.rule)

        win_by_three = 0
        for dir in range(4):
            p = fp_cell.pattern2x[dir].pat(Color.BLACK)
            if p not in (Pattern.F3, Pattern.F3S):
                continue

            posi = pos
            for i in range(MAX_FIND_DIST):
                dx, dy = DIRECTIONS[dir]
                posi = Pos(posi.x - dx, posi.y - dy)
                idx = posi.to_index()
                if not (0 <= idx < FULL_BOARD_CELL_COUNT) or self.cells[idx].piece != Color.EMPTY:
                    break
                c = self.cells[idx]
                if c.pattern4[Color.BLACK] == Pattern4.B_FLEX4 or c.pattern(Color.BLACK, dir) == Pattern.F5 or \
                   (c.pattern4[Color.BLACK] == Pattern4.FORBID and c.pattern(Color.BLACK, dir) == Pattern.F4 and not self.check_forbidden_point(posi)):
                    win_by_three += 1
                    break

            posi = pos
            for i in range(MAX_FIND_DIST):
                dx, dy = DIRECTIONS[dir]
                posi = Pos(posi.x + dx, posi.y + dy)
                idx = posi.to_index()
                if not (0 <= idx < FULL_BOARD_CELL_COUNT) or self.cells[idx].piece != Color.EMPTY:
                    break
                c = self.cells[idx]
                if c.pattern4[Color.BLACK] == Pattern4.B_FLEX4 or c.pattern(Color.BLACK, dir) == Pattern.F5 or \
                   (c.pattern4[Color.BLACK] == Pattern4.FORBID and c.pattern(Color.BLACK, dir) == Pattern.F4 and not self.check_forbidden_point(posi)):
                    win_by_three += 1
                    break

            if win_by_three >= 2:
                break

        self.undo_move(self.rule)
        self.current_side = prev_side
        return win_by_three >= 2

    def get_last_actual_move_of_side(self, side):
        """Matches getLastActualMoveOfSide."""
        assert side in (Color.BLACK, Color.WHITE)
        for i in range(self.move_count - 1, -1, -1):
            move = self.get_recent_move(i)
            if move != Pos.NONE and self.get(move) == side:
                return move
        return Pos.NONE

    def expand_cand_area(self, pos, fill_dist, line_dist):
        """Matches expandCandArea."""
        area = self.state_infos[self.move_count].cand_area
        x, y = pos.x, pos.y

        def cand_condition(p):
            idx = p.to_index()
            return 0 <= idx < FULL_BOARD_CELL_COUNT and self.is_empty(p) and not self.cells[idx].is_candidate()

        area.expand(pos, self.board_size, max(fill_dist, line_dist))
        for i in range(max(3, fill_dist + 1), line_dist + 1):
            for dir in range(4):
                dx, dy = DIRECTIONS[dir]
                posi = Pos(x + dx * i, y + dy * i)
                if cand_condition(posi):
                    self.cells[posi.to_index()].cand += 1
        for xi in range(-fill_dist, fill_dist + 1):
            for yi in range(-fill_dist, fill_dist + 1):
                posi = Pos(x + xi, y + yi)
                if cand_condition(posi):
                    self.cells[posi.to_index()].cand += 1

    # Helper Methods
    def center_pos(self):
        return Pos(self.board_size // 2, self.board_size // 2)

    def is_empty(self, pos):
        return self.cells[pos.to_index()].piece == Color.EMPTY

    def is_valid(self, pos):
        return pos.is_valid()

    def get(self, pos):
        return self.cells[pos.to_index()].piece

    def get_last_move(self):
        return self.state_infos[self.move_count].last_move

    def get_recent_move(self, reverse_idx):
        return self.state_infos[self.move_count - reverse_idx].last_move

    def state_info(self, back_idx=0):
        return self.state_infos[self.move_count - back_idx]

    def p4_count(self, side, p4):
        return self.state_infos[self.move_count].p4_count[side][p4]

    def pass_move_count(self):
        return self.pass_count[Color.BLACK] + self.pass_count[Color.WHITE]

    def pass_move_count_of_side(self, side):
        return self.pass_count[side]

    def side_to_move(self):
        return self.current_side

    def flip_side(self):
        self.current_side = Color.opposite(self.current_side)

    def get_candidate_positions(self):
        return [Pos(x, y) for x in range(self.board_size) for y in range(self.board_size)
                if self.cells[Pos(x, y).to_index()].is_candidate() or self.is_empty(Pos(x, y))]

    def cell(self, pos):
        return self.cells[pos.to_index()]

    def pattern_at(self, pos, dir, rule):
        key = self._get_key_at(pos, dir, rule)
        return PatternConfig.lookup_pattern(rule, key)

    def _get_key_at(self, pos, dir, rule):
        """Matches getKeyAt<R> in board.cpp."""
        x = pos.x + BOARD_BOUNDARY
        y = pos.y + BOARD_BOUNDARY
        half_len = PatternConfig.half_line_len(rule)
        key_width = 2 * half_len + 1  # Total bits for pattern window

        if dir == 0:  # Horizontal
            base_shift = 2 * (x - half_len)
            key = self.bit_key0[y]
            if base_shift < 0:
                key = np.left_shift(key, np.uint64(-base_shift))
            else:
                key = np.right_shift(key, np.uint64(base_shift))
            return key & ((1 << key_width) - 1)  # Mask to pattern width
        elif dir == 1:  # Vertical
            base_shift = 2 * (y - half_len)
            key = self.bit_key1[x]
            if base_shift < 0:
                key = np.left_shift(key, np.uint64(-base_shift))
            else:
                key = np.right_shift(key, np.uint64(base_shift))
            return key & ((1 << key_width) - 1)
        elif dir == 2:  # Diagonal
            diag = x + y
            base_shift = 2 * (diag - half_len)
            key = self.bit_key2[diag]
            if base_shift < 0:
                key = np.left_shift(key, np.uint64(-base_shift))
            else:
                key = np.right_shift(key, np.uint64(base_shift))
            return key & ((1 << key_width) - 1)
        elif dir == 3:  # Anti-diagonal
            anti_diag = FULL_BOARD_SIZE - 1 + x - y
            base_shift = 2 * (anti_diag - half_len)
            key = self.bit_key3[anti_diag]
            if base_shift < 0:
                key = np.left_shift(key, np.uint64(-base_shift))
            else:
                key = np.right_shift(key, np.uint64(base_shift))
            return key & ((1 << key_width) - 1)
        else:
            raise ValueError("Invalid direction")

    def _set_bit_key(self, pos, side):
        x = pos.x + BOARD_BOUNDARY
        y = pos.y + BOARD_BOUNDARY
        shift = np.uint64(2 * x if side == Color.BLACK else 2 * x + 1)
        bit = np.left_shift(np.uint64(1), shift)
        self.bit_key0[y] |= bit
        shift = np.uint64(2 * y if side == Color.BLACK else 2 * y + 1)
        bit = np.left_shift(np.uint64(1), shift)
        self.bit_key1[x] |= bit

        diag = x + y
        if diag < FULL_BOARD_SIZE + FULL_BOARD_SIZE - 1:
            shift = np.uint64(2 * diag if side == Color.BLACK else 2 * diag + 1)
            bit = np.left_shift(np.uint64(1), shift)
            self.bit_key2[diag] |= bit

        anti_diag = FULL_BOARD_SIZE - 1 + x - y
        if 0 <= anti_diag < FULL_BOARD_SIZE + FULL_BOARD_SIZE - 1:
            shift = np.uint64(2 * anti_diag if side == Color.BLACK else 2 * anti_diag + 1)
            bit = np.left_shift(np.uint64(1), shift)
            self.bit_key3[anti_diag] |= bit

    def _flip_bit_key(self, pos, side):
        x = pos.x + BOARD_BOUNDARY
        y = pos.y + BOARD_BOUNDARY
        shift = np.uint64(2 * x if side == Color.BLACK else 2 * x + 1)
        bit = np.left_shift(np.uint64(1), shift)
        self.bit_key0[y] ^= bit
        shift = np.uint64(2 * y if side == Color.BLACK else 2 * y + 1)
        bit = np.left_shift(np.uint64(1), shift)
        self.bit_key1[x] ^= bit
        diag = x + y
        if diag < FULL_BOARD_SIZE + FULL_BOARD_SIZE - 1:
            shift = np.uint64(2 * diag if side == Color.BLACK else 2 * diag + 1)
            bit = np.left_shift(np.uint64(1), shift)
            self.bit_key2[diag] ^= bit
        anti_diag = FULL_BOARD_SIZE - 1 + x - y
        if 0 <= anti_diag < FULL_BOARD_SIZE + FULL_BOARD_SIZE - 1:
            shift = np.uint64(2 * anti_diag if side == Color.BLACK else 2 * anti_diag + 1)
            bit = np.left_shift(np.uint64(1), shift)
            self.bit_key3[anti_diag] ^= bit

    def zobrist_key(self):
        return self.current_zobrist_key

    def ply(self):
        return self.move_count

    def non_pass_move_count(self):
        return self.move_count - self.pass_move_count()