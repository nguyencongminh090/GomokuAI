from typing import Tuple, List, Optional
from .types import Color, Pattern4, Value, Rule, Pattern, CandidateRange, VALUE_ZERO
import time

class StateInfo:
    def __init__(self):
        self.p4_count = {Color.BLACK: {p: 0 for p in Pattern4}, Color.WHITE: {p: 0 for p in Pattern4}}
        self.value_black = VALUE_ZERO
        self.last_move = None
        self.cand_area = set()

    def copy(self):
        """
        Create a deep copy of the StateInfo object.
        """
        new_state = StateInfo()
        new_state.p4_count = {
            Color.BLACK: self.p4_count[Color.BLACK].copy(),
            Color.WHITE: self.p4_count[Color.WHITE].copy()
        }
        new_state.value_black = self.value_black
        new_state.last_move = self.last_move
        new_state.cand_area = self.cand_area.copy()
        return new_state

class Board:
    def __init__(self, board_size: int = 15, cand_range: CandidateRange = CandidateRange.SQUARE4):
        self.size = board_size
        self.cells = [[Color.EMPTY for _ in range(board_size)] for _ in range(board_size)]
        self.move_count = 0
        self.pass_count = {Color.BLACK: 0, Color.WHITE: 0}
        self.current_side = Color.BLACK
        self.last_move = None
        self.last_move_time = time.time()
        self.state_infos = [StateInfo() for _ in range(board_size * board_size * 2 + 1)]
        self.cand_range = cand_range
        self.cand_area_expand_dist = 4 if cand_range == CandidateRange.SQUARE4 else 3
        self.pattern_cache = {}  # Cache for (pos, side, rule) -> Pattern4

    def copy(self):
        """
        Create a deep copy of the Board object.
        """
        new_board = Board(self.size, self.cand_range)
        new_board.cells = [row[:] for row in self.cells]  # Deep copy of cells
        new_board.move_count = self.move_count
        new_board.pass_count = self.pass_count.copy()  # Shallow copy of pass_count
        new_board.current_side = self.current_side
        new_board.last_move = self.last_move
        new_board.last_move_time = self.last_move_time
        new_board.state_infos = [state.copy() for state in self.state_infos]  # Copy each StateInfo
        new_board.cand_area_expand_dist = self.cand_area_expand_dist
        new_board.pattern_cache = {}  # Start with an empty cache
        return new_board

    def new_game(self, rule: Rule = Rule.FREESTYLE):
        self.cells = [[Color.EMPTY for _ in range(self.size)] for _ in range(self.size)]
        self.move_count = 0
        self.pass_count = {Color.BLACK: 0, Color.WHITE: 0}
        self.current_side = Color.BLACK
        self.last_move = None
        self.last_move_time = time.time()
        self.state_infos = [StateInfo() for _ in range(self.size * self.size * 2 + 1)]
        self.pattern_cache.clear()
        self._update_state_info(rule)

    def _get_line(self, pos: Tuple[int, int], direction: Tuple[int, int], side: Color) -> List[Color]:
        row, col = pos
        line = []
        for i in range(-4, 5):
            r, c = row + i * direction[0], col + i * direction[1]
            if 0 <= r < self.size and 0 <= c < self.size:
                line.append(self.cells[r][c])
            else:
                line.append(Color.WALL)
        return line

    def _classify_line(self, line: List[Color], side: Color, rule: Rule) -> Pattern:
        assert len(line) == 9
        mid = 4
        if line[mid] != Color.EMPTY:
            return Pattern.DEAD

        self_color = side
        oppo_color = Color.WHITE if side == Color.BLACK else Color.BLACK
        real_len, full_len = 1, 1
        left_blocked, right_blocked = False, False

        for i in range(mid - 1, -1, -1):
            if line[i] == self_color:
                real_len += 1
                full_len += 1
            elif line[i] == oppo_color or line[i] == Color.WALL:
                left_blocked = True
                break
            else:
                full_len += 1

        for i in range(mid + 1, 9):
            if line[i] == self_color:
                real_len += 1
                full_len += 1
            elif line[i] == oppo_color or line[i] == Color.WALL:
                right_blocked = True
                break
            else:
                full_len += 1

        check_overline = rule != Rule.FREESTYLE and side == Color.BLACK
        if check_overline and real_len >= 6:
            return Pattern.OL
        elif real_len >= 5:
            return Pattern.F5
        elif real_len == 4 and not (left_blocked and right_blocked):
            return Pattern.F4
        elif real_len == 4:
            return Pattern.B4
        elif real_len == 3 and not (left_blocked and right_blocked):
            return Pattern.F3 if full_len >= 6 else Pattern.F3S
        elif real_len == 3:
            return Pattern.B3
        elif real_len == 2 and not (left_blocked and right_blocked):
            if full_len >= 7:
                return Pattern.F2B
            elif full_len == 6:
                return Pattern.F2A
            return Pattern.F2
        elif real_len == 2:
            return Pattern.B2
        elif real_len == 1 and not (left_blocked and right_blocked):
            return Pattern.F1
        elif real_len == 1:
            return Pattern.B1
        return Pattern.DEAD if full_len < 5 else Pattern.DEAD

    def _get_pattern4(self, pos: Tuple[int, int], side: Color, rule: Rule) -> Pattern4:
        cache_key = (pos, side, rule)
        if cache_key in self.pattern_cache:
            return self.pattern_cache[cache_key]

        if not self.is_empty(pos):
            return Pattern4.NONE

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        patterns = [self._classify_line(self._get_line(pos, d, side), side, rule) for d in directions]
        n = {p: patterns.count(p) for p in Pattern}

        if n.get(Pattern.F5, 0) >= 1:
            result = Pattern4.A_FIVE
        elif rule == Rule.RENJU and side == Color.BLACK:
            if n.get(Pattern.OL, 0) >= 1 or n.get(Pattern.F4, 0) + n.get(Pattern.B4, 0) >= 2 or \
               n.get(Pattern.F3, 0) + n.get(Pattern.F3S, 0) >= 2:
                result = Pattern4.FORBID
            else:
                result = self._compute_pattern4(n)
        else:
            result = self._compute_pattern4(n)

        self.pattern_cache[cache_key] = result
        return result

    def _compute_pattern4(self, n: dict) -> Pattern4:
        if n.get(Pattern.B4, 0) >= 2 or n.get(Pattern.F4, 0) >= 1:
            return Pattern4.B_FLEX4
        if n.get(Pattern.B4, 0) >= 1:
            if n.get(Pattern.F3, 0) + n.get(Pattern.F3S, 0) >= 1:
                return Pattern4.C_BLOCK4_FLEX3
            if n.get(Pattern.B3, 0) >= 1 or n.get(Pattern.F2, 0) + n.get(Pattern.F2A, 0) + n.get(Pattern.F2B, 0) >= 1:
                return Pattern4.D_BLOCK4_PLUS
            return Pattern4.E_BLOCK4
        if n.get(Pattern.F3, 0) + n.get(Pattern.F3S, 0) >= 1:
            if n.get(Pattern.F3, 0) + n.get(Pattern.F3S, 0) >= 2:
                return Pattern4.F_FLEX3_2X
            if n.get(Pattern.B3, 0) >= 1 or n.get(Pattern.F2, 0) + n.get(Pattern.F2A, 0) + n.get(Pattern.F2B, 0) >= 1:
                return Pattern4.G_FLEX3_PLUS
            return Pattern4.H_FLEX3
        if n.get(Pattern.B3, 0) >= 2 or (n.get(Pattern.B3, 0) >= 1 and n.get(Pattern.F2, 0) + n.get(Pattern.F2A, 0) + n.get(Pattern.F2B, 0) >= 1):
            return Pattern4.I_BLOCK3_PLUS
        if n.get(Pattern.F2, 0) + n.get(Pattern.F2A, 0) + n.get(Pattern.F2B, 0) >= 2:
            return Pattern4.J_FLEX2_2X
        if n.get(Pattern.B3, 0) >= 1:
            return Pattern4.K_BLOCK3
        if n.get(Pattern.F2, 0) + n.get(Pattern.F2A, 0) + n.get(Pattern.F2B, 0) >= 1:
            return Pattern4.L_FLEX2
        return Pattern4.NONE

    def _get_value_black(self, rule: Rule, p4_black: Pattern4, p4_white: Pattern4) -> Value:
        weights = {
            Pattern4.A_FIVE: 50000, Pattern4.B_FLEX4: 20000, Pattern4.C_BLOCK4_FLEX3: 10000,
            Pattern4.D_BLOCK4_PLUS: 5000, Pattern4.E_BLOCK4: 3000, Pattern4.F_FLEX3_2X: 1000,
            Pattern4.G_FLEX3_PLUS: 500, Pattern4.H_FLEX3: 200, Pattern4.I_BLOCK3_PLUS: 100,
            Pattern4.J_FLEX2_2X: 50, Pattern4.K_BLOCK3: 30, Pattern4.L_FLEX2: 10, Pattern4.NONE: 0,
            Pattern4.FORBID: -10000
        }
        return Value(weights[p4_black] - weights[p4_white])

    def _update_state_info(self, rule: Rule):
        st = self.state_infos[self.move_count]
        st.p4_count = {Color.BLACK: {p: 0 for p in Pattern4}, Color.WHITE: {p: 0 for p in Pattern4}}
        st.value_black = VALUE_ZERO
        st.last_move = self.last_move
        st.cand_area.clear()
        self.pattern_cache.clear()  # Clear cache on state update

        for row in range(self.size):
            for col in range(self.size):
                pos = (row, col)
                if self.cells[row][col] == Color.EMPTY:
                    p4_black = self._get_pattern4(pos, Color.BLACK, rule)
                    p4_white = self._get_pattern4(pos, Color.WHITE, rule)
                    st.p4_count[Color.BLACK][p4_black] += 1
                    st.p4_count[Color.WHITE][p4_white] += 1
                    st.value_black += self._get_value_black(rule, p4_black, p4_white)

        if self.cand_range == CandidateRange.FULL_BOARD:
            self._expand_cand_area(self.size // 2, self.size // 2, self.size // 2, 0)

    def move(self, pos: Tuple[int, int], rule: Rule = Rule.FREESTYLE):
        assert 0 <= pos[0] < self.size and 0 <= pos[1] < self.size
        assert self.cells[pos[0]][pos[1]] == Color.EMPTY

        self.cells[pos[0]][pos[1]] = self.current_side
        self.last_move = pos
        self.last_move_time = time.time()
        self.move_count += 1
        self.state_infos[self.move_count] = StateInfo()
        self.state_infos[self.move_count].last_move = pos
        self._expand_cand_area(pos[0], pos[1], self.cand_area_expand_dist, self.cand_area_expand_dist)
        self._update_state_info(rule)
        self.current_side = Color.WHITE if self.current_side == Color.BLACK else Color.BLACK

    def undo(self, rule: Rule = Rule.FREESTYLE):
        assert self.move_count > 0
        last_pos = self.last_move
        if last_pos is None:
            return
        self.cells[last_pos[0]][last_pos[1]] = Color.EMPTY
        self.last_move = self.state_infos[self.move_count - 1].last_move
        self.move_count -= 1
        self.current_side = Color.WHITE if self.current_side == Color.BLACK else Color.BLACK
        self._update_state_info(rule)

    def _expand_cand_area(self, x: int, y: int, fill_dist: int, line_dist: int):
        area = self.state_infos[self.move_count].cand_area
        for i in range(max(3, fill_dist + 1), line_dist + 1):
            for dir in [(0, 1), (1, 0), (1, 1), (1, -1)]:
                pos = (x + dir[0] * i, y + dir[1] * i)
                if self._is_valid_pos(pos) and self.cells[pos[0]][pos[1]] == Color.EMPTY and pos not in area:
                    area.add(pos)
        for xi in range(-fill_dist, fill_dist + 1):
            for yi in range(-fill_dist, fill_dist + 1):
                pos = (x + xi, y + yi)
                if self._is_valid_pos(pos) and self.cells[pos[0]][pos[1]] == Color.EMPTY and pos not in area:
                    area.add(pos)

    def _is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        return 0 <= pos[0] < self.size and 0 <= pos[1] < self.size

    def side_to_move(self) -> Color:
        return self.current_side

    def p4_count(self, side: Color, pattern: Pattern4) -> int:
        return self.state_infos[self.move_count].p4_count[side][pattern]

    def get_last_move(self) -> Optional[Tuple[int, int]]:
        return self.last_move

    def is_empty(self, pos: Tuple[int, int]) -> bool:
        return self.cells[pos[0]][pos[1]] == Color.EMPTY