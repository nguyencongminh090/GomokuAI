import numpy as np
from .types import Pos, Color, Rule, Value, Depth, Bound, Pattern4, Pattern
from .constants import (FULL_BOARD_CELL_COUNT, VALUE_ZERO, VALUE_INFINITE, VALUE_EVAL_MAX, VALUE_EVAL_MIN, VALUE_NONE,
                       THREAT_MASK_SIZE, AspirationWindowMinDelta, AspirationWindowMaxDelta)  # Added missing imports

class Hash:
    """Matches Hash namespace in config.h."""
    zobrist = np.zeros((Color.COLOR_NB, FULL_BOARD_CELL_COUNT), dtype=np.uint64)

    @staticmethod
    def init():
        """Matches Hash::init in config.cpp."""
        np.random.seed(0)
        for c in range(Color.COLOR_NB):
            for p in range(FULL_BOARD_CELL_COUNT):
                Hash.zobrist[c][p] = np.uint64(np.random.randint(0, 2**64, dtype=np.uint64))  # Specify uint64 dtype

class TTEntry:
    """Matches TTEntry struct in config.h."""
    def __init__(self, key=0, value=Value(20000+1), depth=Depth(-1), bound=Bound.NONE, move=Pos.NONE, static_eval=Value(20000+1)):
        self.key = key & 0xFFFF  # 16-bit key part
        self.value16 = np.int16(value.value)
        self.depth = np.int16(depth.value)
        self.bound_gen = (bound << 6) | (TT.generation & 0x3F)
        self.move = move
        self.static_eval16 = np.int16(static_eval.value)

    def value(self):
        return Value(self.value16)

    def depth(self):
        return Depth(self.depth)

    def bound(self):
        return Bound((self.bound_gen >> 6) & 0x3)

    def generation(self):
        return self.bound_gen & 0x3F

    def move(self):
        return self.move

    def static_eval(self):
        return Value(self.static_eval16)

class TT:
    """Matches TT namespace in config.h/cpp."""
    generation = 0
    hash_size_kb = 16384  # Default 16 MB
    entry_count = 0
    table = None

    @staticmethod
    def init():
        TT.resize(TT.hash_size_kb)

    @staticmethod
    def resize(memory_size_kb):
        from .board import FULL_BOARD_CELL_COUNT  # Deferred import
        TT.hash_size_kb = memory_size_kb
        TT.entry_count = (memory_size_kb * 1024) // 16  # ~16 bytes per TTEntry
        TT.table = np.array([TTEntry() for _ in range(TT.entry_count)], dtype=object)
        TT.clear()

    @staticmethod
    def clear():
        for i in range(TT.entry_count):
            TT.table[i] = TTEntry()

    @staticmethod
    def probe(key):
        index = key % TT.entry_count
        entry = TT.table[index]
        if entry.key == (key >> 48) & 0xFFFF:  # Upper 16 bits
            return entry
        return None

    @staticmethod
    def store(key, value, bound, depth, move, static_eval):
        index = key % TT.entry_count
        entry = TT.table[index]
        if not entry.key or entry.generation() != TT.generation or depth >= entry.depth() or bound == Bound.EXACT:
            TT.table[index] = TTEntry(key, value, depth, bound, move, static_eval)

    @staticmethod
    def inc_generation():
        TT.generation = (TT.generation + 1) & 0x3F

class Config:
    """Matches Config namespace in config.h/cpp."""
    EVALS_THREAT = {
        Rule.FREESTYLE: {
            Color.BLACK: np.zeros(THREAT_MASK_SIZE, dtype=np.int16),
            Color.WHITE: np.zeros(THREAT_MASK_SIZE, dtype=np.int16)
        },
        Rule.STANDARD: {
            Color.BLACK: np.zeros(THREAT_MASK_SIZE, dtype=np.int16),
            Color.WHITE: np.zeros(THREAT_MASK_SIZE, dtype=np.int16)
        },
        Rule.RENJU: {
            Color.BLACK: np.zeros(THREAT_MASK_SIZE, dtype=np.int16),
            Color.WHITE: np.zeros(THREAT_MASK_SIZE, dtype=np.int16)
        }
    }

    PATTERN_SCORE = {
        Rule.FREESTYLE: {
            Pattern.DEAD: VALUE_ZERO,
            Pattern.B1: Value(2),
            Pattern.F1: Value(5),
            Pattern.B2: Value(10),
            Pattern.F2: Value(20),
            Pattern.F2A: Value(30),
            Pattern.F2B: Value(50),
            Pattern.B3: Value(100),
            Pattern.F3: Value(200),
            Pattern.F3S: Value(300),
            Pattern.B4: Value(500),
            Pattern.F4: Value(1000),
            Pattern.F5: VALUE_EVAL_MAX - Value(1),
            Pattern.OL: Value(-VALUE_EVAL_MAX + Value(1))
        },
        Rule.STANDARD: {
            Pattern.DEAD: VALUE_ZERO,
            Pattern.B1: Value(2),
            Pattern.F1: Value(5),
            Pattern.B2: Value(10),
            Pattern.F2: Value(20),
            Pattern.F2A: Value(30),
            Pattern.F2B: Value(50),
            Pattern.B3: Value(100),
            Pattern.F3: Value(200),
            Pattern.F3S: Value(300),
            Pattern.B4: Value(500),
            Pattern.F4: Value(1000),
            Pattern.F5: VALUE_EVAL_MAX - Value(1),
            Pattern.OL: Value(-VALUE_EVAL_MAX + Value(1))
        },
        Rule.RENJU: {
            Pattern.DEAD: VALUE_ZERO,
            Pattern.B1: Value(2),
            Pattern.F1: Value(5),
            Pattern.B2: Value(10),
            Pattern.F2: Value(20),
            Pattern.F2A: Value(30),
            Pattern.F2B: Value(50),
            Pattern.B3: Value(100),
            Pattern.F3: Value(200),
            Pattern.F3S: Value(300),
            Pattern.B4: Value(500),
            Pattern.F4: Value(1000),
            Pattern.F5: VALUE_EVAL_MAX - Value(1),
            Pattern.OL: Value(-VALUE_EVAL_MAX + Value(1)),
            Pattern.FORBID: VALUE_ZERO
        }
    }

    @staticmethod
    def init():
        Hash.init()
        TT.init()
        Config._init_threat_table()

    @staticmethod
    def _init_threat_table():
        """Matches initialization of EVALS_THREAT in config.cpp."""
        for rule in [Rule.FREESTYLE, Rule.STANDARD, Rule.RENJU]:
            for side in [Color.BLACK, Color.WHITE]:
                table = Config.EVALS_THREAT[rule][side]
                for mask in range(THREAT_MASK_SIZE):
                    oppo = Color.opposite(side)
                    value = Value(0)
                    if mask & (1 << 0):  # oppoFive
                        value = Value(-10000)
                    elif mask & (1 << 1):  # selfFlexFour
                        value = Value(1000)
                    elif mask & (1 << 2):  # oppoFlexFour
                        value = Value(-1000)
                    elif mask & (1 << 3):  # selfFourPlus
                        value = Value(500)
                    elif mask & (1 << 4):  # selfFour
                        value = Value(300)
                    elif mask & (1 << 5):  # selfThreePlus
                        value = Value(100)
                    elif mask & (1 << 6):  # selfThree
                        value = Value(50)
                    elif mask & (1 << 7):  # oppoFourPlus
                        value = Value(-500)
                    elif mask & (1 << 8):  # oppoFour
                        value = Value(-300)
                    elif mask & (1 << 9):  # oppoThreePlus
                        value = Value(-100)
                    elif mask & (1 << 10):  # oppoThree
                        value = Value(-50)

                    # Combination logic (approximated from eval.cpp)
                    if mask & (1 << 0):  # oppoFive dominates
                        if mask & (1 << 1):
                            value = Value(-9500)  # oppoFive + selfFlexFour
                        elif mask & (1 << 2):
                            value = Value(-11000)  # oppoFive + oppoFlexFour
                        elif mask & (1 << 7):
                            value = Value(-10500)  # oppoFive + oppoFourPlus
                    elif mask & (1 << 2) and mask & (1 << 7):
                        value = Value(-1500)  # oppoFlexFour + oppoFourPlus
                    elif mask & (1 << 1) and mask & (1 << 3):
                        value = Value(1500)  # selfFlexFour + selfFourPlus

                    table[mask] = np.int16(value)

    @staticmethod
    def get_score(rule, side, pcode_black, pcode_white):
        """Matches Config::getScore in config.cpp."""
        pcode = pcode_black if side == Color.BLACK else pcode_white
        if rule == Rule.RENJU and side == Color.BLACK and pcode >= Pattern.FORBID:
            return Value(0)
        return Config.PATTERN_SCORE[rule].get(pcode, Value(0))

    @staticmethod
    def get_value_black(rule, pcode_black, pcode_white):
        """Matches Config::getValueBlack in config.cpp."""
        black_score = Config.get_score(rule, Color.BLACK, pcode_black, pcode_white)
        white_score = Config.get_score(rule, Color.WHITE, pcode_black, pcode_white)
        value_black = Value(black_score.value - white_score.value)
        return Value(np.clip(value_black.value, VALUE_EVAL_MIN.value, VALUE_EVAL_MAX.value))

    @staticmethod
    def mate_in(ply):
        return Value(VALUE_EVAL_MAX - ply)
    
    @staticmethod
    def mated_in(ply):
        return Value(-VALUE_EVAL_MAX + ply)

    @staticmethod
    def next_aspiration_window_delta(value, prev_delta=VALUE_ZERO):
        abs_value = abs(value.value)
        base = min(max(AspirationWindowMinDelta.value, abs_value // 8), AspirationWindowMaxDelta.value)
        return Value(base + prev_delta.value // 2)

# Initialize config
Config.init()