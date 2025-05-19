import numpy as np
from enum import IntEnum
from .types import Pos, Color, Rule, Value, Pattern, Pattern4
from .board import Board
from .pattern import PatternConfig
from .constants import FULL_BOARD_CELL_COUNT, MAX_FIND_DIST, DIRECTIONS

# Defense table for F3/F3S patterns (approximates PatternConfig::lookupDefenceTable<R>)
DEFENCE_TABLE = np.zeros(1 << 10, dtype=np.uint8)  # 1024 entries

def init_defence_table():
    """Initialize DEFENCE_TABLE based on Rapfi’s pattern logic."""
    for key in range(1 << 10):
        line = [(key >> (i * 2)) & 3 for i in range(5)]  # 0=empty, 1=self, 2=opp
        opp_count = sum(1 for x in line if x == 2)
        mask = 0
        if opp_count >= 3:
            for i in range(3):
                if i + 2 < 5 and line[i] == 2 and line[i + 1] == 2 and line[i + 2] == 2:
                    if i == 0 and line[0] == 0:  # _XOOO_
                        mask |= 1 << 7  # +3
                    elif i + 3 == 5 and line[4] == 0:  # _OOOX_
                        mask |= 1 << 0  # -4
                    elif i > 0 and line[i - 1] == 0:  # X*OOO_
                        mask |= 1 << (4 + i - 1)  # e.g., -1
                    elif i + 3 < 5 and line[i + 3] == 0:  # _XOOO*_
                        mask |= 1 << (4 + i + 3)  # e.g., +2
        DEFENCE_TABLE[key] = mask

init_defence_table()


class ScoredMove:
    """Matches ScoredMove struct in movegen.h."""
    def __init__(self, pos, score=0):
        self.pos = pos
        self.score = score  # Union with raw_score in Rapfi, simplified here

    def __eq__(self, other):
        return isinstance(other, ScoredMove) and self.pos == other.pos

    def __lt__(self, other):
        return self.pos < other.pos

    def __repr__(self):
        return f"ScoredMove({self.pos}, score={self.score})"

class MoveGenerator:
    """Matches MoveGenerator class in movegen.h/cpp."""
    def __init__(self, rule):
        self.rule = rule

    def generate(self, board: Board, gen_type):
        """Matches generate<Type> in movegen.cpp."""
        moves = []
        if gen_type == GenType.ALL:
            moves = self._generate_all(board)
        elif gen_type == GenType.VCF:
            moves = self._generate_vcf(board, board.side_to_move())
        elif gen_type == GenType.VCF_DEFEND:
            moves = self._generate_vcf_defend(board, Color.opposite(board.side_to_move()))
        elif gen_type & GenType.WINNING:
            moves = self._generate_winning(board)
        elif gen_type & GenType.DEFEND_FIVE:
            moves = self._generate_defend_five(board)
        elif gen_type & GenType.DEFEND_FOUR:
            moves = self._generate_defend_four(board, bool(gen_type & GenType.ALL))
        elif gen_type & GenType.DEFEND_B4F3:
            if self.rule == Rule.FREESTYLE:
                moves = self._generate_b4f3_defence(board, Rule.FREESTYLE)
            elif self.rule == Rule.STANDARD:
                moves = self._generate_b4f3_defence(board, Rule.STANDARD)
            elif self.rule == Rule.RENJU:
                moves = self._generate_b4f3_defence(board, Rule.RENJU)
        return moves

    def _generate_all(self, board: Board):
        """Matches generate<ALL>."""
        moves = []
        for pos in board.get_candidate_positions():
            if board.is_empty(pos):
                moves.append(ScoredMove(pos))
        return moves

    def _generate_vcf(self, board: Board, side: Color):
        """Matches generate<VCF>."""
        moves = []
        for pos in board.get_candidate_positions():
            if self._basic_pattern_filter(board, pos, side, GenType.VCF):
                moves.append(ScoredMove(pos))
        return moves

    def _generate_vcf_defend(self, board: Board, opp_side: Color):
        """Matches generate<VCF_DEFEND> (custom extension in Rapfi)."""
        moves = []
        for pos in board.get_candidate_positions():
            if self._basic_pattern_filter(board, pos, opp_side, GenType.VCF_DEFEND):
                moves.append(ScoredMove(pos))
        return moves

    def _generate_winning(self, board: Board):
        """Matches generate<WINNING>."""
        side = board.side_to_move()
        if board.p4_count(side, Pattern4.A_FIVE):
            return [ScoredMove(self._find_first_pattern4_pos(board, side, Pattern4.A_FIVE))]
        elif board.p4_count(side, Pattern4.B_FLEX4):
            return [ScoredMove(self._find_first_pattern4_pos(board, side, Pattern4.B_FLEX4))]
        return []

    def _generate_defend_five(self, board: Board):
        """Matches generate<DEFEND_FIVE>."""
        opp = Color.opposite(board.side_to_move())
        assert board.p4_count(opp, Pattern4.A_FIVE) > 0
        move = board.state_info().last_pattern4(opp, Pattern4.A_FIVE)
        if board.is_empty(move) and board.cell(move).pattern4[opp] == Pattern4.A_FIVE:
            return [ScoredMove(move)]
        for pos in board.get_candidate_positions():
            if board.cell(pos).pattern4[opp] == Pattern4.A_FIVE:
                return [ScoredMove(pos)]
        return []

    def _generate_defend_four(self, board: Board, include_losing_moves):
        """Matches generate<DEFEND_FOUR> or <DEFEND_FOUR | ALL>."""
        assert board.p4_count(Color.opposite(board.side_to_move()), Pattern4.B_FLEX4) > 0
        moves = self._find_four_defence(board, include_losing_moves)
        moves.sort()
        moves = list(dict.fromkeys(moves))  # Unique moves
        self_side = board.side_to_move()
        moves = [m for m in moves if board.cell(m.pos).pattern4[self_side] < Pattern4.E_BLOCK4]
        return moves

    def _generate_b4f3_defence(self, board: Board, rule):
        """Matches generate<DEFEND_B4F3 | RULE_X>."""
        assert board.p4_count(Color.opposite(board.side_to_move()), Pattern4.C_BLOCK4_FLEX3) > 0
        moves = self._find_b4f3_defence(board, rule)
        moves.sort()
        moves = list(dict.fromkeys(moves))
        self_side = board.side_to_move()
        moves = [m for m in moves if board.cell(m.pos).pattern4[self_side] < Pattern4.E_BLOCK4]
        return moves

    def _basic_pattern_filter(self, board: Board, pos: Pos, side: Color, gen_type):
        """Matches basicPatternFilter<Type> in movegen.cpp."""
        c = board.cell(pos)
        p4 = c.pattern4[side]

        if gen_type & GenType.WINNING and p4 >= Pattern4.B_FLEX4:
            return True

        if gen_type & GenType.VCF:
            if gen_type & GenType.COMB:
                if p4 >= Pattern4.D_BLOCK4_PLUS:
                    return True
            elif gen_type & GenType.RULE_RENJU:
                if p4 >= Pattern4.E_BLOCK4 or (p4 == Pattern4.FORBID and self._is_renju_four(board, pos, side)):
                    return True
            else:
                if p4 >= Pattern4.E_BLOCK4:
                    return True
        return False

    def _is_renju_four(self, board: Board, pos: Pos, side: Color):
        """Helper for Renju-specific four check."""
        for dir in range(4):
            if board.cell(pos).pattern(side, dir) >= Pattern.B4:
                return True
        return False

    def _find_first_pattern4_pos(self, board: Board, side: Color, p4):
        """Matches findFirstPattern4Pos."""
        for pos in board.get_candidate_positions():
            if board.cell(pos).pattern4[side] == p4:
                return pos
        return Pos.NONE

    def _find_four_defence(self, board: Board, include_losing_moves):
        """Matches findFourDefence<IncludeLosingMoves>."""
        opp = Color.opposite(board.side_to_move())
        assert board.p4_count(opp, Pattern4.A_FIVE) == 0
        assert board.p4_count(opp, Pattern4.B_FLEX4) > 0

        moves = []
        last_flex4_pos = board.state_info().last_flex4_attack_move[opp]
        if last_flex4_pos != Pos.NONE:
            attack_cell = board.cell(last_flex4_pos)
            for dir in range(4):
                pattern = attack_cell.pattern(opp, dir)
                if pattern in (Pattern.F3, Pattern.F3S):
                    pos = self._find_f4_pos_in_f3_line(board, last_flex4_pos, dir)
                    if pos != Pos.NONE and board.cell(pos).pattern4[opp] == Pattern4.B_FLEX4:
                        if not include_losing_moves and moves:
                            return moves
                        moves.extend(self._find_f3_line_defence(board, pos, dir, include_losing_moves))
                        if moves and not include_losing_moves:
                            return moves
                elif pattern == Pattern.B3:
                    b4_pos = self._find_b4_in_line(board, last_flex4_pos, dir)
                    if b4_pos != Pos.NONE:
                        if board.cell(b4_pos).pattern4[board.side_to_move()] >= Pattern4.E_BLOCK4:
                            return []  # Losing move
                        moves.append(ScoredMove(b4_pos))
                        for d in range(4):
                            moves.extend(self._find_all_b3_counter_defence(board, b4_pos, d))
                        if not include_losing_moves:
                            return moves
        if not moves:
            moves = self._find_all_pseudo_four_defend_pos(board, opp)
        return moves

    def _find_f3_line_defence(self, board: Board, f4_pos: Pos, dir: int, include_losing_moves):
        """Matches findF3LineDefence in findFourDefence."""
        opp = Color.opposite(board.side_to_move())
        assert board.cell(f4_pos).pattern(opp, dir) == Pattern.F4
        moves = [ScoredMove(f4_pos)]

        pos = f4_pos
        for _ in range(MAX_FIND_DIST):
            pos -= DIRECTIONS[dir]
            idx = pos.to_index()
            if not (0 <= idx < FULL_BOARD_CELL_COUNT) or board.cells[idx].piece == opp:
                continue
            elif board.cells[idx].piece == Color.EMPTY:
                c = board.cell(pos)
                if c.pattern(opp, dir) == Pattern.F4 and (c.pattern4[opp] != Pattern4.FORBID or not board.check_forbidden_point(pos)):
                    moves.append(ScoredMove(pos))
                    return moves[:2] if not include_losing_moves else moves
                break

        pos = f4_pos
        for _ in range(MAX_FIND_DIST):
            pos += DIRECTIONS[dir]
            idx = pos.to_index()
            if not (0 <= idx < FULL_BOARD_CELL_COUNT) or board.cells[idx].piece == opp:
                continue
            elif board.cells[idx].piece == Color.EMPTY:
                c = board.cell(pos)
                if c.pattern(opp, dir) == Pattern.F4 and (c.pattern4[opp] != Pattern4.FORBID or not board.check_forbidden_point(pos)):
                    moves.append(ScoredMove(pos))
                    return moves[:2] if not include_losing_moves else moves
                else:
                    moves.append(ScoredMove(pos))
                    return moves[:3]
        return moves

    def _find_b4_in_line(self, board: Board, f4_pos: Pos, dir: int):
        """Matches findB4InLine in findFourDefence."""
        opp = Color.opposite(board.side_to_move())
        pos = f4_pos
        for _ in range(MAX_FIND_DIST):
            pos -= DIRECTIONS[dir]
            idx = pos.to_index()
            if not (0 <= idx < FULL_BOARD_CELL_COUNT) or board.cells[idx].piece == opp:
                continue
            elif board.cells[idx].piece == Color.EMPTY:
                c = board.cell(pos)
                if c.pattern(opp, dir) == Pattern.B4 or (self.rule == Rule.RENJU and c.pattern4[opp] == Pattern4.FORBID and self._check_renju_f4(board, pos, dir, opp)):
                    if self.rule != Rule.FREESTYLE and not self._check_not_overline_b4(board, f4_pos, pos, dir, opp):
                        return pos
                    return pos
                break

        pos = f4_pos
        for _ in range(MAX_FIND_DIST):
            pos += DIRECTIONS[dir]
            idx = pos.to_index()
            if not (0 <= idx < FULL_BOARD_CELL_COUNT) or board.cells[idx].piece == opp:
                continue
            elif board.cells[idx].piece == Color.EMPTY:
                c = board.cell(pos)
                if c.pattern(opp, dir) == Pattern.B4 or (self.rule == Rule.RENJU and c.pattern4[opp] == Pattern4.FORBID and self._check_renju_f4(board, pos, dir, opp)):
                    if self.rule != Rule.FREESTYLE and not self._check_not_overline_b4(board, f4_pos, pos, dir, opp):
                        return pos
                    return pos
                break
        return Pos.NONE

    def _check_renju_f4(self, board: Board, pos: Pos, dir: int, opp: Color):
        """Matches checkRenjuF4 lambda."""
        board.cells[pos.to_index()].piece = opp
        pattern = board.pattern_at(pos, dir, Rule.STANDARD)[opp]
        board.cells[pos.to_index()].piece = Color.EMPTY
        return pattern >= Pattern.B4

    def _check_not_overline_b4(self, board: Board, f4_pos: Pos, pos: Pos, dir: int, opp: Color):
        """Matches checkNotOverlineB4 lambda."""
        board.flip_side()
        board.make_move(pos, self.rule)
        has_five = board.cell(f4_pos).pattern(opp, dir) == Pattern.F5
        board.undo_move(self.rule)
        board.flip_side()
        return has_five

    def _find_all_b3_counter_defence(self, board: Board, b4_pos: Pos, dir: int):
        """Matches findAllB3CounterDefence."""
        self_side = board.side_to_move()
        opp = Color.opposite(self_side)
        is_pseudo_forbidden_b4 = self.rule == Rule.RENJU and self_side == Color.BLACK and board.cell(b4_pos).pattern4[self_side] == Pattern4.FORBID
        moves = []

        pos = b4_pos
        for _ in range(MAX_FIND_DIST):
            pos -= DIRECTIONS[dir]
            idx = pos.to_index()
            if not (0 <= idx < FULL_BOARD_CELL_COUNT) or board.cells[idx].piece == self_side:
                continue
            elif board.cells[idx].piece == Color.EMPTY and (is_pseudo_forbidden_b4 or board.cell(pos).pattern(self_side, dir) >= Pattern.B3):
                moves.append(ScoredMove(pos))
            else:
                break

        pos = b4_pos
        for _ in range(MAX_FIND_DIST):
            pos += DIRECTIONS[dir]
            idx = pos.to_index()
            if not (0 <= idx < FULL_BOARD_CELL_COUNT) or board.cells[idx].piece == self_side:
                continue
            elif board.cells[idx].piece == Color.EMPTY and (is_pseudo_forbidden_b4 or board.cell(pos).pattern(self_side, dir) >= Pattern.B3):
                moves.append(ScoredMove(pos))
            else:
                break
        return moves

    def _find_all_pseudo_four_defend_pos(self, board: Board, side: Color):
        """Matches findAllPseudoFourDefendPos."""
        moves = []
        for pos in board.get_candidate_positions():
            c = board.cell(pos)
            if c.pattern4[side] >= Pattern4.E_BLOCK4:
                moves.append(ScoredMove(pos))
            elif c.pattern4[side] == Pattern4.FORBID and side == Color.BLACK:
                for dir in range(4):
                    if c.pattern(side, dir) >= Pattern.B4:
                        moves.append(ScoredMove(pos))
                        break
        return moves

    def _find_f4_pos_in_f3_line(self, board: Board, last_flex4_pos: Pos, dir: int):
        """Matches findF4PosInF3Line."""
        opp = Color.opposite(board.side_to_move())
        pos = last_flex4_pos
        for _ in range(MAX_FIND_DIST):
            pos -= DIRECTIONS[dir]
            idx = pos.to_index()
            if not (0 <= idx < FULL_BOARD_CELL_COUNT) or board.cells[idx].piece == opp:
                continue
            elif board.cells[idx].piece == Color.EMPTY and board.cell(pos).pattern(opp, dir) == Pattern.F4 and board.cell(pos).pattern4[opp] == Pattern4.B_FLEX4:
                return pos
            break
        pos = last_flex4_pos
        for _ in range(MAX_FIND_DIST):
            pos += DIRECTIONS[dir]
            idx = pos.to_index()
            if not (0 <= idx < FULL_BOARD_CELL_COUNT) or board.cells[idx].piece == opp:
                continue
            elif board.cells[idx].piece == Color.EMPTY and board.cell(pos).pattern(opp, dir) == Pattern.F4 and board.cell(pos).pattern4[opp] == Pattern4.B_FLEX4:
                return pos
            break
        return Pos.NONE

    def _find_b4f3_defence(self, board: Board, rule):
        """Matches findB4F3Defence<R>."""
        opp = Color.opposite(board.side_to_move())
        assert board.p4_count(opp, Pattern4.A_FIVE) == 0 and board.p4_count(opp, Pattern4.B_FLEX4) == 0
        assert board.p4_count(opp, Pattern4.C_BLOCK4_FLEX3) > 0

        b4f3_pos = board.state_info().last_pattern4(opp, Pattern4.C_BLOCK4_FLEX3)
        c = board.cell(b4f3_pos)
        if c.piece != Color.EMPTY or c.pattern4[opp] != Pattern4.C_BLOCK4_FLEX3:
            b4f3_pos = self._find_first_pattern4_pos(board, opp, Pattern4.C_BLOCK4_FLEX3)

        moves = [ScoredMove(b4f3_pos)]
        b4f3_cell = board.cell(b4f3_pos)
        assert b4f3_cell.piece == Color.EMPTY and b4f3_cell.pattern4[opp] == Pattern4.C_BLOCK4_FLEX3

        for dir in range(4):
            pattern = b4f3_cell.pattern(opp, dir)
            if pattern in (Pattern.F3, Pattern.F3S):
                moves.extend(self._find_f3_line_defence_b4f3(board, b4f3_pos, dir, rule))
            elif pattern == Pattern.B4:
                b4_pos = self._find_b4_in_line_b4f3(board, b4f3_pos, dir, rule)
                if b4_pos != Pos.NONE:
                    if board.cell(b4_pos).pattern4[board.side_to_move()] >= Pattern4.E_BLOCK4:
                        return []
                    moves.append(ScoredMove(b4_pos))
                    for d in range(4):
                        moves.extend(self._find_all_b3_counter_defence_b4f3(board, b4_pos, d, rule))
        return moves

    def _find_b4_in_line_b4f3(self, board: Board, b4f3_pos: Pos, dir: int, rule):
        """Matches findB4InLine in findB4F3Defence<R>."""
        opp = Color.opposite(board.side_to_move())
        pos = b4f3_pos
        for _ in range(MAX_FIND_DIST):
            pos -= DIRECTIONS[dir]
            idx = pos.to_index()
            if not (0 <= idx < FULL_BOARD_CELL_COUNT) or board.cells[idx].piece == opp:
                continue
            elif board.cells[idx].piece == Color.EMPTY:
                c = board.cell(pos)
                if c.pattern(opp, dir) == Pattern.B4 or (rule == Rule.RENJU and c.pattern4[opp] == Pattern4.FORBID and self._check_renju_f4(board, pos, dir, opp)):
                    if rule != Rule.FREESTYLE and not self._check_not_overline_b4_b4f3(board, b4f3_pos, pos, dir, opp):
                        return pos
                    return pos
                break

        pos = b4f3_pos
        for _ in range(MAX_FIND_DIST):
            pos += DIRECTIONS[dir]
            idx = pos.to_index()
            if not (0 <= idx < FULL_BOARD_CELL_COUNT) or board.cells[idx].piece == opp:
                continue
            elif board.cells[idx].piece == Color.EMPTY:
                c = board.cell(pos)
                if c.pattern(opp, dir) == Pattern.B4 or (rule == Rule.RENJU and c.pattern4[opp] == Pattern4.FORBID and self._check_renju_f4(board, pos, dir, opp)):
                    if rule != Rule.FREESTYLE and not self._check_not_overline_b4_b4f3(board, b4f3_pos, pos, dir, opp):
                        return pos
                    return pos
                break
        return Pos.NONE

    def _check_not_overline_b4_b4f3(self, board: Board, b4f3_pos: Pos, pos: Pos, dir: int, opp: Color):
        """Matches checkNotOverlineB4 in findB4F3Defence<R>."""
        board.flip_side()
        board.make_move(pos, self.rule)
        has_five = board.cell(b4f3_pos).pattern(opp, dir) == Pattern.F5
        board.undo_move(self.rule)
        board.flip_side()
        return has_five

    def _find_all_b3_counter_defence_b4f3(self, board: Board, b4_pos: Pos, dir: int, rule):
        """Matches findAllB3CounterDefence in findB4F3Defence<R>."""
        self_side = board.side_to_move()
        opp = Color.opposite(self_side)
        is_pseudo_forbidden_b4 = rule == Rule.RENJU and self_side == Color.BLACK and board.cell(b4_pos).pattern4[self_side] == Pattern4.FORBID
        moves = []

        pos = b4_pos
        for _ in range(MAX_FIND_DIST):
            pos -= DIRECTIONS[dir]
            idx = pos.to_index()
            if not (0 <= idx < FULL_BOARD_CELL_COUNT) or board.cells[idx].piece == self_side:
                continue
            elif board.cells[idx].piece == Color.EMPTY and (is_pseudo_forbidden_b4 or board.cell(pos).pattern(self_side, dir) >= Pattern.B3):
                moves.append(ScoredMove(pos))
            else:
                break

        pos = b4_pos
        for _ in range(MAX_FIND_DIST):
            pos += DIRECTIONS[dir]
            idx = pos.to_index()
            if not (0 <= idx < FULL_BOARD_CELL_COUNT) or board.cells[idx].piece == self_side:
                continue
            elif board.cells[idx].piece == Color.EMPTY and (is_pseudo_forbidden_b4 or board.cell(pos).pattern(self_side, dir) >= Pattern.B3):
                moves.append(ScoredMove(pos))
            else:
                break
        return moves

    def _find_f3_line_defence_b4f3(self, board: Board, f3_pos: Pos, dir: int, rule):
        """Matches findF3LineDefence in findB4F3Defence<R>."""
        opp = Color.opposite(board.side_to_move())
        assert board.cell(f3_pos).pattern(opp, dir) in (Pattern.F3, Pattern.F3S)

        half_len = PatternConfig.half_line_len(rule)
        key = self._get_line_key(board, f3_pos, dir, rule, opp, half_len)
        defence_mask = DEFENCE_TABLE[key & 0x3FF]  # 10-bit mask

        moves = []
        check_renju_defence = rule == Rule.RENJU and opp == Color.BLACK
        left_mask = (defence_mask >> 4) & 0xF
        right_mask = defence_mask & 0xF
        left_renju_defence = Pos.NONE
        right_renju_defence = Pos.NONE
        prev_found = False
        found_left_forbidden = False
        found_right_forbidden = False

        if check_renju_defence:
            board.flip_side()
            board.make_move(f3_pos, Rule.RENJU)

        # Left side (-4 to -1)
        for i in range(4):
            pos = f3_pos - DIRECTIONS[dir] * (i + 1)
            idx = pos.to_index()
            if not (0 <= idx < FULL_BOARD_CELL_COUNT):
                break
            if (left_mask >> (3 - i)) & 1:
                if board.is_empty(pos):
                    moves.append(ScoredMove(pos))
                    prev_found = True
                    if check_renju_defence:
                        found_left_forbidden |= board.check_forbidden_point(pos)
            elif check_renju_defence and prev_found and board.is_empty(pos):
                left_renju_defence = pos
                prev_found = False

        # Right side (+1 to +3)
        prev_found = False
        for i in range(4):
            pos = f3_pos + DIRECTIONS[dir] * (i + 1)
            idx = pos.to_index()
            if not (0 <= idx < FULL_BOARD_CELL_COUNT):
                break
            if (right_mask >> i) & 1:
                if board.is_empty(pos):
                    moves.append(ScoredMove(pos))
                    prev_found = True
                    if check_renju_defence:
                        found_right_forbidden |= board.check_forbidden_point(pos)
            elif check_renju_defence and prev_found and board.is_empty(pos):
                right_renju_defence = pos
                prev_found = False

        if check_renju_defence:
            if found_left_forbidden and right_renju_defence != Pos.NONE and board.is_empty(right_renju_defence):
                moves.append(ScoredMove(right_renju_defence))
            if found_right_forbidden and left_renju_defence != Pos.NONE and board.is_empty(left_renju_defence):
                moves.append(ScoredMove(left_renju_defence))
            board.undo_move(Rule.RENJU)
            board.flip_side()

        return moves

    def _get_line_key(self, board: Board, pos: Pos, dir: int, rule, side: Color, half_len):
        """Simulates getKeyAt<R> for defense table."""
        key = 0
        for i in range(-2, 3):  # 5-position window around pos
            check_pos = pos + DIRECTIONS[dir] * i
            idx = check_pos.to_index()
            if not (0 <= idx < FULL_BOARD_CELL_COUNT):
                bit_value = 0  # Out-of-bounds as empty
            else:
                piece = board.cells[idx].piece
                bit_value = 0 if piece == Color.EMPTY else (1 if piece == side else 2)
            key = (key << 2) | bit_value
        return key

    def validate_opponent_c_move(self, board: Board):
        """Matches validateOpponentCMove."""
        if board.side_to_move() == Color.BLACK:
            return True
        assert board.p4_count(Color.BLACK, Pattern4.C_BLOCK4_FLEX3) > 0
        assert board.p4_count(Color.BLACK, Pattern4.B_FLEX4) == 0

        last_b4f3_pos = board.state_info().last_pattern4(Color.BLACK, Pattern4.C_BLOCK4_FLEX3)
        if not (board.is_empty(last_b4f3_pos) and board.cell(last_b4f3_pos).pattern4[Color.BLACK] == Pattern4.C_BLOCK4_FLEX3):
            last_b4f3_pos = self._find_first_pattern4_pos(board, Color.BLACK, Pattern4.C_BLOCK4_FLEX3)

        board.flip_side()
        board.make_move(last_b4f3_pos, Rule.RENJU)
        has_b_move = board.p4_count(Color.BLACK, Pattern4.B_FLEX4) > 0
        board.undo_move(Rule.RENJU)
        board.flip_side()
        return has_b_move

class GenType(IntEnum):
    """Matches GenType enum in movegen.h."""
    ALL = 0
    WINNING = 1
    DEFEND_FIVE = 2
    DEFEND_FOUR = 3
    DEFEND_B4F3 = 4
    VCF = 5
    VCF_DEFEND = 6
    COMB = 0x100
    RULE_RENJU = 0x200