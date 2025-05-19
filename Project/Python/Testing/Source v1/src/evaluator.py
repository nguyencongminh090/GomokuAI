import numpy as np
from .types import Pos, Color, Rule, Value, Pattern4
from .board import Board
from .config import Config
from .constants import VALUE_INFINITE, VALUE_EVAL_MIN, VALUE_EVAL_MAX

class Evaluator:
    """Matches Evaluator class in eval.h/cpp (classical implementation)."""
    def __init__(self, rule):
        """Matches constructor with Rule parameter."""
        self.rule = rule

    def evaluate(self, board: Board, alpha=Value(-VALUE_INFINITE), beta=Value(VALUE_INFINITE)):
        """Matches evaluate<R> in eval.cpp."""
        if board.ply() > 0:
            if self.rule == Rule.FREESTYLE:
                return self._evaluate(board, alpha, beta, Rule.FREESTYLE)
            elif self.rule == Rule.STANDARD:
                return self._evaluate(board, alpha, beta, Rule.STANDARD)
            elif self.rule == Rule.RENJU:
                return self._evaluate(board, alpha, beta, Rule.RENJU)
            else:
                raise ValueError("Unknown rule")
        else:
            return self._evaluate_empty(board, alpha, beta)

    def _evaluate(self, board: Board, alpha, beta, rule):
        """Matches evaluate<R> for non-empty boards in eval.cpp."""
        assert board.ply() > 0
        self_side = board.side_to_move()
        st0 = board.state_info()  # Current state
        st1 = board.state_info(1)  # Previous state

        # Average basic evaluation from current and previous states
        basic_eval = Value((self._evaluate_basic(st0, self_side) + self._evaluate_basic(st1, self_side)) // 2)
        threat_eval = self._evaluate_threat(st0, self_side, rule)

        # Combine and clamp evaluation
        eval = Value(basic_eval + threat_eval)
        eval = Value(np.clip(eval.value, VALUE_EVAL_MIN.value, VALUE_EVAL_MAX.value))
        return eval

    def _evaluate_empty(self, board: Board, alpha, beta):
        """Matches evaluate() empty board case in eval.cpp."""
        self_side = board.side_to_move()
        st = board.state_info()
        basic_eval = self._evaluate_basic(st, self_side)
        threat_eval = self._evaluate_threat(st, self_side, self.rule)
        eval = Value(basic_eval + threat_eval)
        return Value(np.clip(eval.value, VALUE_EVAL_MIN.value, VALUE_EVAL_MAX.value))

    def _evaluate_basic(self, st, self_side):
        """Matches evaluateBasic in eval.cpp."""
        return st.value_black if self_side == Color.BLACK else Value(-st.value_black.value)

    def _evaluate_threat(self, st, self_side, rule):
        """Matches evaluateThreat<R> in eval.cpp."""
        mask = self._make_threat_mask(st, self_side)
        return Value(Config.EVALS_THREAT[rule][self_side][mask])

    def _make_threat_mask(self, st, self_side):
        """Matches makeThreatMask in eval.cpp."""
        oppo = Color.opposite(self_side)
        mask = np.uint32(0)
        mask |= (1 if st.p4_count(oppo, Pattern4.A_FIVE) > 0 else 0) << 0  # oppoFive
        mask |= (1 if st.p4_count(self_side, Pattern4.B_FLEX4) > 0 else 0) << 1  # selfFlexFour
        mask |= (1 if st.p4_count(oppo, Pattern4.B_FLEX4) > 0 else 0) << 2  # oppoFlexFour
        mask |= (1 if (st.p4_count(self_side, Pattern4.D_BLOCK4_PLUS) + st.p4_count(self_side, Pattern4.C_BLOCK4_FLEX3)) > 0 else 0) << 3  # selfFourPlus
        mask |= (1 if st.p4_count(self_side, Pattern4.E_BLOCK4) > 0 else 0) << 4  # selfFour
        mask |= (1 if (st.p4_count(self_side, Pattern4.G_FLEX3_PLUS) + st.p4_count(self_side, Pattern4.F_FLEX3_2X)) > 0 else 0) << 5  # selfThreePlus
        mask |= (1 if st.p4_count(self_side, Pattern4.H_FLEX3) > 0 else 0) << 6  # selfThree
        mask |= (1 if (st.p4_count(oppo, Pattern4.D_BLOCK4_PLUS) + st.p4_count(oppo, Pattern4.C_BLOCK4_FLEX3)) > 0 else 0) << 7  # oppoFourPlus
        mask |= (1 if st.p4_count(oppo, Pattern4.E_BLOCK4) > 0 else 0) << 8  # oppoFour
        mask |= (1 if (st.p4_count(oppo, Pattern4.G_FLEX3_PLUS) + st.p4_count(oppo, Pattern4.F_FLEX3_2X)) > 0 else 0) << 9  # oppoThreePlus
        mask |= (1 if st.p4_count(oppo, Pattern4.H_FLEX3) > 0 else 0) << 10  # oppoThree
        return mask

    def init_empty_board(self):
        pass  # No-op in classical evaluator

    def before_move(self, board: Board, pos: Pos):
        pass  # No-op in classical evaluator

    def after_move(self, board: Board, pos: Pos):
        pass  # No-op in classical evaluator

    def before_undo(self, board: Board, pos: Pos):
        pass  # No-op in classical evaluator

    def after_undo(self, board: Board, pos: Pos):
        pass  # No-op in classical evaluator

    def after_pass(self, board: Board):
        pass  # No-op in classical evaluator

    def after_undo_pass(self, board: Board):
        pass  # No-op in classical evaluator