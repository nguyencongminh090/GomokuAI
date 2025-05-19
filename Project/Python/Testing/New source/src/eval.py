from typing import Tuple, Dict, Optional
from .types import Color, Pattern4, Value, Rule, VALUE_EVAL_MIN, VALUE_EVAL_MAX
from .board import Board
import logging


logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, board_size: int, board: Board = None, rule: Rule = Rule.FREESTYLE):
        self.board_size = board_size
        self.board = board
        self.rule = rule
        self.weights = {
            Pattern4.A_FIVE: Value(100000),
            Pattern4.B_FLEX4: Value(50000),
            Pattern4.C_BLOCK4_FLEX3: Value(20000),
            Pattern4.D_BLOCK4_PLUS: Value(10000),
            Pattern4.E_BLOCK4: Value(5000),
            Pattern4.F_FLEX3_2X: Value(3000),
            Pattern4.G_FLEX3_PLUS: Value(1000),
            Pattern4.H_FLEX3: Value(500),
            Pattern4.I_BLOCK3_PLUS: Value(200),
            Pattern4.J_FLEX2_2X: Value(100),
            Pattern4.K_BLOCK3: Value(50),
            Pattern4.L_FLEX2: Value(20),
            Pattern4.NONE: Value(0),
            Pattern4.FORBID: Value(-10000)
        }

    def _make_threat_mask(self, board: Board, side: Color) -> int:
        opposite = Color.WHITE if side == Color.BLACK else Color.BLACK
        self_counts = {p: board.p4_count(side, p) for p in Pattern4}
        opp_counts = {p: board.p4_count(opposite, p) for p in Pattern4}
        mask = 0
        mask |= 0b1 & -int(opp_counts[Pattern4.A_FIVE] > 0)
        mask |= 0b10 & -int(self_counts[Pattern4.B_FLEX4] > 0)
        mask |= 0b100 & -int(opp_counts[Pattern4.B_FLEX4] > 0)
        mask |= 0b1000 & -int(self_counts[Pattern4.F_FLEX3_2X] > 0)
        mask |= 0b10000 & -int(opp_counts[Pattern4.F_FLEX3_2X] > 0)
        return mask

    def evaluate(self, board: Board, side: Color) -> Value:
        ai_counts = {p: board.p4_count(side, p) for p in Pattern4}
        opp = Color.WHITE if side == Color.BLACK else Color.BLACK
        opp_counts = {p: board.p4_count(opp, p) for p in Pattern4}

        ai_score = sum(self.weights.get(p, Value(0)) * count for p, count in ai_counts.items())
        opp_score = sum(self.weights.get(p, Value(0)) * count for p, count in opp_counts.items())

        threat_eval = Value(0)
        if opp_counts[Pattern4.A_FIVE] > 0:
            threat_eval -= Value(100000)
        else:
            if opp_counts[Pattern4.B_FLEX4] > 0:
                threat_eval -= Value(50000) * opp_counts[Pattern4.B_FLEX4]
            if opp_counts[Pattern4.F_FLEX3_2X] > 0:
                threat_eval -= Value(25000) * opp_counts[Pattern4.F_FLEX3_2X]

        if ai_counts[Pattern4.A_FIVE] > 0:
            threat_eval += Value(100000)
        else:
            if ai_counts[Pattern4.B_FLEX4] > 0:
                threat_eval += Value(50000) * ai_counts[Pattern4.B_FLEX4]
            if ai_counts[Pattern4.F_FLEX3_2X] > 0:
                threat_eval += Value(20000) * ai_counts[Pattern4.F_FLEX3_2X]

        total_score = ai_score - opp_score + threat_eval
        logger.debug(f"Eval for {side}: ai_score={ai_score}, opp_score={opp_score}, threat_eval={threat_eval}, total={total_score}")
        return Value(max(VALUE_EVAL_MIN, min(VALUE_EVAL_MAX, total_score)))