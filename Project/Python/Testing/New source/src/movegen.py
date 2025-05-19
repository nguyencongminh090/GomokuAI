from typing import List, Tuple, Optional
from .types import Color, Pattern4, Rule
from .board import Board
import logging

logger = logging.getLogger(__name__)

class GenType:
    """
    Constants representing different move generation types.
    """
    WINNING = 1
    VCF = 2
    VCT = 4
    VC2 = 8
    TRIVIAL = 16
    ALL = WINNING | VCF | VCT | VC2 | TRIVIAL

class ScoredMove:
    """
    Represents a move with an associated score for prioritization.
    """
    def __init__(self, pos: Tuple[int, int], score: float = 0.0):
        self.pos = pos
        self.score = score
        self.policy = 0.0

    @staticmethod
    def ScoreComparator(a: 'ScoredMove', b: 'ScoredMove') -> bool:
        return a.score > b.score

    def __repr__(self):
        return f"ScoredMove({self.pos}, score={self.score})"

class MovePicker:
    """
    Generates and prioritizes moves based on heuristics and patterns.
    """
    def __init__(self, rule: Rule, board: Board, tt_move: Optional[Tuple[int, int]] = None, use_normalized_policy: bool = False):
        self.board = board
        self.rule = rule
        self.tt_move = tt_move
        self.use_normalized_policy = use_normalized_policy
        self.normalized_policy_temp = 1.0
        self.cur_move = []
        self._generate_moves()
        self.cur_move.sort(key=lambda m: m.score, reverse=True)
        logger.debug(f"MovePicker init: moves={[m.pos for m in self.cur_move]}")

    def _generate_moves(self):
        """
        Generate moves in a prioritized order based on critical patterns.
        """
        self_side = self.board.side_to_move()
        oppo = Color.WHITE if self_side == Color.BLACK else Color.BLACK
        self.cur_move = []

        # Priority 1: Block opponent's A_FIVE
        if self.board.p4_count(oppo, Pattern4.A_FIVE) > 0:
            self._generate_defend_five_moves()
        # Priority 2: Form own A_FIVE
        elif self.board.p4_count(self_side, Pattern4.A_FIVE) > 0:
            self._generate_winning_moves()
        # Priority 3: Block opponent's B_FLEX4
        elif self.board.p4_count(oppo, Pattern4.B_FLEX4) > 0:
            self._generate_defend_four_moves()
        # Priority 4: Form own B_FLEX4
        elif self.board.p4_count(self_side, Pattern4.B_FLEX4) > 0:
            self._generate_winning_moves()
        # Priority 5: Other moves
        else:
            self._generate_all_moves()

        if not self.cur_move:
            logger.warning(f"No moves generated, falling back to all moves")
            self._generate_all_moves()

    def _generate_winning_moves(self):
        """
        Generate moves that form winning patterns (A_FIVE or B_FLEX4).
        """
        self_side = self.board.side_to_move()
        pos = self._find_first_pattern4_pos(Pattern4.A_FIVE, self_side)
        if not pos:
            pos = self._find_first_pattern4_pos(Pattern4.B_FLEX4, self_side)
        if pos:
            score = 100000 if pos == Pattern4.A_FIVE else 50000
            self.cur_move.append(ScoredMove(pos, score=score))
            logger.debug(f"Winning move added: {pos}")

    def _generate_defend_five_moves(self):
        """
        Generate moves that block opponent's A_FIVE.
        """
        oppo = Color.WHITE if self.board.side_to_move() == Color.BLACK else Color.BLACK
        pos = self._find_first_pattern4_pos(Pattern4.A_FIVE, oppo)
        if pos:
            self.cur_move.append(ScoredMove(pos, score=75000))
            logger.debug(f"Defend five move added: {pos}")

    def _generate_defend_four_moves(self):
        """
        Generate moves that block opponent's B_FLEX4.
        """
        self.cur_move = self._generate_defense_moves(Pattern4.B_FLEX4, include_losing=False)
        logger.debug(f"Defend four moves: {[m.pos for m in self.cur_move]}")

    def _generate_all_moves(self):
        """
        Generate all possible moves with heuristic scoring.
        """
        self.cur_move = []
        last_move = self.board.get_last_move()
        oppo = Color.WHITE if self.board.side_to_move() == Color.BLACK else Color.BLACK
        threat_priority = {
            Pattern4.A_FIVE: 100000,
            Pattern4.B_FLEX4: 50000,
            Pattern4.C_BLOCK4_FLEX3: 20000,
            Pattern4.D_BLOCK4_PLUS: 10000,
            Pattern4.E_BLOCK4: 5000,
            Pattern4.F_FLEX3_2X: 3000,
            Pattern4.G_FLEX3_PLUS: 1000,
            Pattern4.H_FLEX3: 500,
            Pattern4.I_BLOCK3_PLUS: 200,
            Pattern4.J_FLEX2_2X: 100,
            Pattern4.K_BLOCK3: 50,
            Pattern4.L_FLEX2: 20,
        }
        for row in range(self.board.size):
            for col in range(self.board.size):
                pos = (row, col)
                if self.board.is_empty(pos):
                    p4_self = self.board._get_pattern4(pos, self.board.side_to_move(), self.rule)
                    p4_oppo = self.board._get_pattern4(pos, oppo, self.rule)
                    score = threat_priority.get(p4_self, 0)
                    if p4_oppo in threat_priority:
                        score += threat_priority[p4_oppo] * 3  # Triple weight for defense
                    if last_move:
                        distance = max(abs(row - last_move[0]), abs(col - last_move[1]))
                        score += 100 / (distance + 1)  # Proximity bonus
                    self.cur_move.append(ScoredMove(pos, score=score))
        logger.debug(f"All moves generated: {len(self.cur_move)} moves")

    def _find_first_pattern4_pos(self, pattern: Pattern4, side: Color = None) -> Optional[Tuple[int, int]]:
        """
        Find the first position that forms the specified pattern for the given side.
        """
        side = side or self.board.side_to_move()
        for row in range(self.board.size):
            for col in range(self.board.size):
                pos = (row, col)
                if self.board.is_empty(pos):
                    p4 = self.board._get_pattern4(pos, side, self.rule)
                    if p4 == pattern:
                        logger.debug(f"Found {pattern} at {pos} for {side}")
                        return pos
        logger.debug(f"No {pattern} found for {side}")
        return None

    def _generate_defense_moves(self, pattern: Pattern4, include_losing: bool) -> List[ScoredMove]:
        """
        Generate moves that block opponent's specified pattern.
        """
        moves = []
        oppo = Color.WHITE if self.board.side_to_move() == Color.BLACK else Color.BLACK
        threat_scores = {
            Pattern4.A_FIVE: 100000,
            Pattern4.B_FLEX4: 50000,
            Pattern4.C_BLOCK4_FLEX3: 20000,
            Pattern4.D_BLOCK4_PLUS: 10000,
            Pattern4.E_BLOCK4: 5000,
        }
        if self.board.p4_count(oppo, pattern) > 0:
            for row in range(self.board.size):
                for col in range(self.board.size):
                    pos = (row, col)
                    if self.board.is_empty(pos):
                        board_copy = self.board.copy()
                        board_copy.move(pos)
                        new_oppo_count = board_copy.p4_count(oppo, pattern)
                        if new_oppo_count < self.board.p4_count(oppo, pattern):
                            score = threat_scores.get(pattern, 0) * 3
                            moves.append(ScoredMove(pos, score=score))
        logger.debug(f"Defense moves for {pattern}: {[m.pos for m in moves]}")
        return moves

    def __call__(self) -> Optional[Tuple[int, int]]:
        """
        Return the next move from the generated list.
        """
        if self.cur_move:
            move = self.cur_move.pop(0).pos
            logger.debug(f"Returning move: {move}")
            return move
        logger.error("No moves available, should not happen")
        return None