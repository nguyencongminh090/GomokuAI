import numpy as np
from enum import IntEnum
from .types import Pos, Color, Rule, Value, Depth, Pattern4, Bound
from .board import Board
from .movegen import MoveGenerator, ScoredMove, GenType
from .evaluator import Evaluator
from .config import Config, TT
from .constants import (FULL_BOARD_CELL_COUNT, MAX_PLY, VALUE_ZERO, VALUE_INFINITE, VALUE_NONE, ASPIRATION_DEPTH,
                       AspirationWindow, RAZOR_DEPTH, RazorMargin, RazorReduction, FUTILITY_DEPTH, FutilityMargin,
                       CheckExtension, Block4Flex3Extension, MaxSearchDepth, MAX_DEPTH, MAX_MOVES)

class SearchOptions:
    """Custom options class for search configuration."""
    def __init__(self):
        self.time_limit = 180000  # 2 seconds in milliseconds
        self.max_depth = MaxSearchDepth
        self.start_depth = Depth(1)
        self.multi_pv = 1
        self.disable_opening_query = False

class ABSearchData:
    """Matches ABSearchData in searcher.h."""
    def __init__(self):
        self.multi_pv = 1
        self.pv_idx = 0
        self.root_depth = Depth(0)
        self.root_delta = VALUE_ZERO
        self.root_alpha = -VALUE_INFINITE
        self.singular_root = False
        self.completed_depth = Depth(0)
        self.best_move_changes = 0.0
        self.main_history = np.zeros((Color.COLOR_NB, FULL_BOARD_CELL_COUNT), dtype=np.int32)  # [Color][Pos]
        self.counter_move_history = np.zeros((Color.COLOR_NB, FULL_BOARD_CELL_COUNT, Color.COLOR_NB, FULL_BOARD_CELL_COUNT), dtype=np.int32)  # [Color][PrevPos][Color][Pos]

    def clear_data(self, th):
        self.multi_pv = 1
        self.pv_idx = 0
        self.root_depth = Depth(0)
        self.completed_depth = Depth(0)
        self.best_move_changes = 0.0
        self.singular_root = False
        self.main_history.fill(0)
        self.counter_move_history.fill(0)

class SearchStack:
    """Matches SearchStack struct in searcher.cpp."""
    def __init__(self, ply, static_eval):
        self.ply = ply
        self.static_eval = static_eval
        self.move_count = 0
        self.current_move = Pos.NONE
        self.skip_move = Pos.NONE
        self.tt_pv = False
        self.pv = [Pos.NONE] * (MAX_PLY.value + 1)
        self.num_null_moves = 0
        self.move_p4 = [Pattern4.NONE, Pattern4.NONE]
        self.extra_extension = 0.0
        self.stat_score = 0
        self.db_child_written = False
        self.db_value_depth = -32768  # INT16_MIN
        self.killers = [Pos.NONE, Pos.NONE]

    def update_pv(self, move, child_ss):
        self.pv[0] = move
        i = 0
        while child_ss.pv[i] != Pos.NONE and i < MAX_PLY.value:
            self.pv[i + 1] = child_ss.pv[i]
            i += 1
        while i < MAX_PLY.value:
            self.pv[i + 1] = Pos.NONE
            i += 1

class NodeType(IntEnum):
    """Matches NodeType enum in searcher.h."""
    Root = 0
    PV = 1
    NonPV = 2

class ABSearcher:
    """Matches ABSearcher in searcher.h/cpp."""
    def __init__(self):
        self.previous_time_reduction = 1.0
        self.previous_best_value = VALUE_NONE
        self.reductions = np.zeros((Rule.RULE_NB, MAX_MOVES + 1), dtype=np.float32)  # Fixed: Use MAX_MOVES from constants.py

    def set_memory_limit(self, memory_size_kb):
        TT.resize(memory_size_kb)

    def get_memory_limit(self):
        return TT.hash_size_kb

    def clear(self, pool, clear_all_memory):
        self.previous_time_reduction = 1.0
        self.previous_best_value = VALUE_NONE
        for r in range(Rule.RULE_NB):
            for m in range(1, MAX_MOVES + 1):  # Fixed: Use MAX_MOVES from constants.py
                self.reductions[r][m] = max(0.0, np.log(m) * 0.5 / np.log(pool.size() + 1))
        if clear_all_memory:
            TT.clear()

    def search_main(self, th):
        opts = th.options
        if not opts.disable_opening_query:
            pass  # Placeholder for Opening::probeOpening

        if not th.root_moves:
            if th.non_pass_move_count() == 0:
                th.best_move = th.board.center_pos()
                return
            for pos in th.board.get_candidate_positions():
                if th.board.is_empty(pos):
                    th.best_move = pos
                    return

        if th.board.p4_count(th.board.side_to_move(), Pattern4.A_FIVE):
            th.root_moves[0].value = Config.mate_in(1)
            th.best_move = th.root_moves[0].pv[0]
            return

        TT.inc_generation()
        self.search(th)
        th.best_move = th.root_moves[0].pv[0] if th.root_moves else Pos.NONE

    def search(self, th):
        sd = ABSearchData()
        opts = th.options
        init_value = Evaluator(opts.rule).evaluate(th.board)
        stack_array = [SearchStack(i, init_value if i == 0 else VALUE_NONE) for i in range(MAX_PLY.value + 4)]
        ss = stack_array[2]  # ss[-2]
        best_value = -VALUE_INFINITE
        last_best_move = Pos.NONE
        last_move_change_depth = Depth(0)
        total_best_move_changes = 0.0

        max_depth = min(opts.max_depth, min(MaxSearchDepth, MAX_DEPTH))
        start_depth = min(max_depth, max(Depth(1), opts.start_depth))
        sd.multi_pv = min(opts.multi_pv, len(th.root_moves))

        for sd.root_depth in range(start_depth, max_depth + 1):
            total_best_move_changes *= 0.5

            for rm in th.root_moves:
                rm.previous_value = rm.value
                rm.previous_pv = rm.pv[:]

            for sd.pv_idx in range(sd.multi_pv):
                th.sel_depth = Depth(0)
                delta = Config.next_aspiration_window_delta(th.root_moves[sd.pv_idx].previous_value) if sd.root_depth >= ASPIRATION_DEPTH and AspirationWindow else VALUE_INFINITE
                alpha = max(th.root_moves[sd.pv_idx].previous_value - delta, -VALUE_INFINITE)
                beta = min(th.root_moves[sd.pv_idx].previous_value + delta, VALUE_INFINITE)
                fail_high_cnt = 0

                while True:
                    adjusted_depth = max(1.0, float(sd.root_depth) - fail_high_cnt / 4.0)
                    value = self._search(th, sd, opts.rule, th.board, ss, stack_array, alpha, beta, Depth(adjusted_depth), False, NodeType.Root)
                    if value <= alpha:
                        beta = Value((alpha.value + beta.value) // 2)
                        alpha = max(value - delta, -VALUE_INFINITE)
                        fail_high_cnt = 0
                    elif value >= beta:
                        beta = min(value + delta, VALUE_INFINITE)
                        fail_high_cnt += 1
                    else:
                        break
                    delta = Config.next_aspiration_window_delta(value, delta)

                th.root_moves[sd.pv_idx].value = value
                th.root_moves[sd.pv_idx].pv = ss.pv[:]
                th.root_moves[:sd.multi_pv] = sorted(th.root_moves[:sd.multi_pv], key=lambda rm: -rm.value)

            sd.completed_depth = sd.root_depth
            best_value = th.root_moves[0].value
            if th.root_moves[0].pv[0] != last_best_move:
                last_best_move = th.root_moves[0].pv[0]
                last_move_change_depth = sd.root_depth
                sd.best_move_changes += 1
                total_best_move_changes += 1.0 / float(sd.root_depth)

    def _search(self, th, sd, rule, board: Board, ss, stack_array, alpha, beta, depth, cut_node, node_type):
        pv_node = node_type in (NodeType.PV, NodeType.Root)
        root_node = node_type == NodeType.Root
        depth = max(Depth(0), depth)
        assert pv_node or alpha == beta - Value(1)

        if depth <= Depth(0) and not root_node:
            return Evaluator(rule).evaluate(board, alpha, beta)

        if ss.ply >= MAX_PLY.value:
            return Evaluator(rule).evaluate(board)

        if not root_node and board.pass_move_count() >= board.board_cell_count:
            return VALUE_ZERO

        tt_entry = TT.probe(board.zobrist_key())
        tt_value = tt_entry.value() if tt_entry else VALUE_NONE
        tt_depth = tt_entry.depth() if tt_entry else Depth(-1)
        tt_bound = tt_entry.bound() if tt_entry else Bound.NONE
        tt_move = tt_entry.move() if tt_entry else Pos.NONE
        tt_hit = tt_entry is not None

        if not pv_node and tt_hit and tt_depth >= depth:
            if (tt_bound == Bound.LOWER and tt_value >= beta) or \
               (tt_bound == Bound.UPPER and tt_value <= alpha) or \
               tt_bound == Bound.EXACT:
                return tt_value

        if board.p4_count(board.side_to_move(), Pattern4.A_FIVE):
            return Config.mate_in(ss.ply + 1)

        moves = MoveGenerator(rule).generate(board, GenType.WINNING)
        if moves:
            return Config.mate_in(ss.ply + 1)

        if board.p4_count(Color.opposite(board.side_to_move()), Pattern4.A_FIVE):
            return Config.mated_in(ss.ply)

        static_eval = ss.static_eval if ss.static_eval != VALUE_NONE else Evaluator(rule).evaluate(board)
        improving = ss.ply < 2 or static_eval > (stack_array[ss.ply - 2].static_eval if ss.ply >= 2 else VALUE_NONE)
        skip_quiets = False

        if not root_node and not pv_node and depth <= RAZOR_DEPTH and static_eval + RazorMargin <= alpha:
            new_depth = depth - RazorReduction
            if new_depth <= Depth(0):
                return static_eval
            value = self._search(th, sd, rule, board, ss, stack_array, alpha, beta, new_depth, True, NodeType.NonPV)
            if value <= alpha:
                return value

        if not root_node and depth >= FUTILITY_DEPTH and static_eval + FutilityMargin * float(depth) <= alpha:
            skip_quiets = True

        mg = MoveGenerator(rule)
        moves = mg.generate(board, GenType.ALL)
        if not moves:
            return Config.mated_in(ss.ply)

        best_value = -VALUE_INFINITE
        best_move = Pos.NONE
        move_count = 0
        alpha_orig = alpha

        for move_idx, move in enumerate(moves):
            pos = move.pos
            if root_node and pos in [rm.pv[0] for rm in th.root_moves[:sd.pv_idx]]:
                continue
            if pos == ss.skip_move:
                continue

            gives_check = board.cell(pos).pattern4[board.side_to_move()] >= Pattern4.B_FLEX4
            move_p4 = board.cell(pos).pattern4

            extension = 0.0
            if gives_check:
                extension = max(extension, CheckExtension)
            if move_p4[board.side_to_move()] >= Pattern4.C_BLOCK4_FLEX3:
                extension = max(extension, Block4Flex3Extension)
            new_depth = depth + extension - 1.0

            ss.move_count = move_count + 1
            ss.current_move = pos
            ss.move_p4 = move_p4
            board.make_move(pos, rule)
            th.nodes += 1

            if move_count == 0:
                value = -self._search(th, sd, rule, board, stack_array[ss.ply + 1], stack_array, -beta, -alpha, new_depth, not cut_node, NodeType.PV if pv_node else NodeType.NonPV)
            else:
                reduction = self.reductions[rule][move_count] if not gives_check else 0.0
                reduced_depth = max(new_depth - reduction, Depth(0))
                value = -self._search(th, sd, rule, board, stack_array[ss.ply + 1], stack_array, -(alpha + Value(1)), -alpha, reduced_depth, True, NodeType.NonPV)
                if value > alpha and reduced_depth < new_depth:
                    value = -self._search(th, sd, rule, board, stack_array[ss.ply + 1], stack_array, -(alpha + Value(1)), -alpha, new_depth, not cut_node, NodeType.NonPV)
                if value > alpha and pv_node:
                    value = -self._search(th, sd, rule, board, stack_array[ss.ply + 1], stack_array, -beta, -alpha, new_depth, not cut_node, NodeType.PV)

            board.undo_move(rule)
            move_count += 1

            if value > best_value:
                best_value = value
                best_move = pos
                if value > alpha:
                    alpha = value
                    if pv_node:
                        ss.update_pv(pos, stack_array[ss.ply + 1])
                    if value >= beta:
                        if not gives_check:
                            sd.main_history[board.side_to_move()][pos.to_index()] += int(depth * depth)
                            for prev_move in moves[:move_idx]:
                                sd.main_history[board.side_to_move()][prev_move.pos.to_index()] -= int(depth * depth)
                        break

        bound = Bound.EXACT if best_value > alpha_orig else (Bound.LOWER if best_value >= beta else Bound.UPPER)
        TT.store(board.zobrist_key(), best_value, bound, depth, best_move, static_eval)
        return best_value

    def _vcf_search(self, th, sd, rule, board: Board, ss, stack_array, alpha, beta, depth=Depth(0), node_type=NodeType.NonPV):
        pv_node = node_type in (NodeType.PV, NodeType.Root)
        if ss.ply >= MAX_PLY.value:
            return Evaluator(rule).evaluate(board)

        tt_entry = TT.probe(board.zobrist_key())
        tt_value = tt_entry.value() if tt_entry else VALUE_NONE
        tt_depth = tt_entry.depth() if tt_entry else Depth(-1)
        tt_bound = tt_entry.bound() if tt_entry else Bound.NONE
        tt_move = tt_entry.move() if tt_entry else Pos.NONE
        tt_hit = tt_entry is not None

        if not pv_node and tt_hit and tt_depth >= depth:
            if (tt_bound == Bound.LOWER and tt_value >= beta) or \
               (tt_bound == Bound.UPPER and tt_value <= alpha) or \
               tt_bound == Bound.EXACT:
                return tt_value

        moves = MoveGenerator(rule).generate(board, GenType.VCF)
        if not moves:
            return Evaluator(rule).evaluate(board, alpha, beta)

        best_value = -VALUE_INFINITE
        best_move = Pos.NONE
        alpha_orig = alpha

        for move in moves:
            pos = move.pos
            board.make_move(pos, rule)
            value = -self._vcf_search(th, sd, rule, board, stack_array[ss.ply + 1], stack_array, -beta, -alpha, depth, NodeType.PV if pv_node else NodeType.NonPV)
            board.undo_move(rule)

            if value > best_value:
                best_value = value
                best_move = pos
                if value > alpha:
                    alpha = value
                    if pv_node:
                        ss.update_pv(pos, stack_array[ss.ply + 1])
                    if value >= beta:
                        break

        bound = Bound.EXACT if best_value > alpha_orig else (Bound.LOWER if best_value >= beta else Bound.UPPER)
        TT.store(board.zobrist_key(), best_value, bound, depth, best_move, VALUE_NONE)
        return best_value

    def _vcf_defend(self, th, sd, rule, board: Board, ss, stack_array, alpha, beta, depth=Depth(0), node_type=NodeType.NonPV):
        pv_node = node_type in (NodeType.PV, NodeType.Root)
        if ss.ply >= MAX_PLY.value:
            return Evaluator(rule).evaluate(board)

        tt_entry = TT.probe(board.zobrist_key())
        tt_value = tt_entry.value() if tt_entry else VALUE_NONE
        tt_depth = tt_entry.depth() if tt_entry else Depth(-1)
        tt_bound = tt_entry.bound() if tt_entry else Bound.NONE
        tt_move = tt_entry.move() if tt_entry else Pos.NONE
        tt_hit = tt_entry is not None

        if not pv_node and tt_hit and tt_depth >= depth:
            if (tt_bound == Bound.LOWER and tt_value >= beta) or \
               (tt_bound == Bound.UPPER and tt_value <= alpha) or \
               tt_bound == Bound.EXACT:
                return tt_value

        moves = MoveGenerator(rule).generate(board, GenType.VCF_DEFEND)
        if not moves:
            return Evaluator(rule).evaluate(board, alpha, beta)

        best_value = -VALUE_INFINITE
        best_move = Pos.NONE
        alpha_orig = alpha

        for move in moves:
            pos = move.pos
            board.make_move(pos, rule)
            value = -self._vcf_defend(th, sd, rule, board, stack_array[ss.ply + 1], stack_array, -beta, -alpha, depth, NodeType.PV if pv_node else NodeType.NonPV)
            board.undo_move(rule)

            if value > best_value:
                best_value = value
                best_move = pos
                if value > alpha:
                    alpha = value
                    if pv_node:
                        ss.update_pv(pos, stack_array[ss.ply + 1])
                    if value >= beta:
                        break

        bound = Bound.EXACT if best_value > alpha_orig else (Bound.LOWER if best_value >= beta else Bound.UPPER)
        TT.store(board.zobrist_key(), best_value, bound, depth, best_move, VALUE_NONE)
        return best_value

    def _pick_best_thread(self, threads):
        best_value = -VALUE_INFINITE
        best_thread = threads.main()
        for th in threads:
            if th.root_moves and th.root_moves[0].value > best_value:
                best_value = th.root_moves[0].value
                best_thread = th
        return best_thread

searcher = ABSearcher()