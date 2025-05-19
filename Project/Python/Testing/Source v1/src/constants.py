from .types import Depth, Value

# Board-specific constants from board.h
MAX_BOARD_SIZE = 20
BOARD_BOUNDARY = 4
FULL_BOARD_SIZE = MAX_BOARD_SIZE + 2 * BOARD_BOUNDARY  # 28
FULL_BOARD_CELL_COUNT = FULL_BOARD_SIZE * FULL_BOARD_SIZE  # 784
FULL_BOARD_START = BOARD_BOUNDARY * FULL_BOARD_SIZE + BOARD_BOUNDARY  # 116
FULL_BOARD_END = FULL_BOARD_START + MAX_BOARD_SIZE * MAX_BOARD_SIZE - 1  # 515

# Search-specific constants from config.h
MaxSearchDepth = 256
MAX_PLY = Value(MaxSearchDepth + 32)
MAX_MOVES = 256
MAX_DEPTH = Depth(MaxSearchDepth + MAX_PLY.value)

# Value constants from config.h
VALUE_ZERO = Value(0)
VALUE_INFINITE = Value(20000)
VALUE_EVAL_MAX = Value(VALUE_INFINITE.value - MAX_PLY.value - 1)
VALUE_EVAL_MIN = Value(-VALUE_EVAL_MAX.value)
VALUE_NONE = Value(VALUE_INFINITE.value + 1)

# Tuning constants from config.h
ASPIRATION_DEPTH = Depth(4)
AspirationWindow = True
AspirationWindowMinDelta = Value(16)
AspirationWindowMaxDelta = Value(1024)

RAZOR_DEPTH = Depth(4)
RazorMargin = Value(300)
RazorReduction = Depth(2)

FUTILITY_DEPTH = Depth(7)
FutilityMargin = Value(100)

CheckExtension = 1.0
Block4Flex3Extension = 0.5

StatScoreMax = 16384
MaxHistoryScore = 16384

THREAT_MASK_SIZE = 1 << 11  # 2048 entries

# Movement constants from movegen.h
MAX_FIND_DIST = 4
DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]