"""
Position and Direction related definitions for the Gomoku engine,
based on Rapfi's pos.h.
"""
import math
from enum import IntEnum, Enum
from functools import total_ordering # For comparison operators

# -------------------------------------------------
# Board size & limits (from pos.h)

FULL_BOARD_SIZE: int = 32
FULL_BOARD_CELL_COUNT: int = FULL_BOARD_SIZE * FULL_BOARD_SIZE
BOARD_BOUNDARY: int = 5
MAX_BOARD_SIZE: int = FULL_BOARD_SIZE - 2 * BOARD_BOUNDARY
MAX_MOVES: int = MAX_BOARD_SIZE * MAX_BOARD_SIZE + 1 # Max board cells + 1 for PASS

# -------------------------------------------------

class Direction(IntEnum):
    """
    Direction represents one of the eight line directions on the board.
    Values are integer offsets.
    """
    UP = -FULL_BOARD_SIZE
    LEFT = -1
    DOWN = FULL_BOARD_SIZE
    RIGHT = 1

    UP_LEFT = UP + LEFT
    UP_RIGHT = UP + RIGHT
    DOWN_LEFT = DOWN + LEFT
    DOWN_RIGHT = DOWN + RIGHT

# Array of 4 main unique directions for iteration (e.g., for pattern checking)
# Rapfi: constexpr Direction DIRECTION[] = {RIGHT, DOWN, UP_RIGHT, DOWN_RIGHT};
DIRECTIONS = (Direction.RIGHT, Direction.DOWN, Direction.UP_RIGHT, Direction.DOWN_RIGHT)

# -------------------------------------------------
@total_ordering
class Pos:
    """
    Represents a move coordinate on board.
    Internally stores a single integer `_pos` similar to Rapfi.
    Coordinates (x,y) are relative to the playable board area (0-indexed).
    The internal `_pos` includes `BOARD_BOUNDARY`.
    """
    _pos: int

    def __init__(self, x_or_val: int, y: int | None = None):
        """
        Initializes a Pos object.
        - Pos(x, y): Creates a Pos from 0-indexed board coordinates.
        - Pos(raw_value): Creates a Pos from its internal integer representation.
        """
        if y is not None: # Constructor from x, y coordinates
            # Assumes x and y are 0-indexed for the MAX_BOARD_SIZE area
            # Rapfi: _pos(((y + BOARD_BOUNDARY) << 5) | (x + BOARD_BOUNDARY))
            # FULL_BOARD_SIZE (32) means 5 bits for shift
            self._pos = ((y + BOARD_BOUNDARY) << 5) | (x_or_val + BOARD_BOUNDARY)
        else: # Constructor from raw _pos value
            self._pos = x_or_val

    @property
    def x(self) -> int:
        """Returns the 0-indexed X coordinate on the playable board."""
        # Rapfi: (_pos & 31) - BOARD_BOUNDARY
        return (self._pos & (FULL_BOARD_SIZE - 1)) - BOARD_BOUNDARY

    @property
    def y(self) -> int:
        """Returns the 0-indexed Y coordinate on the playable board."""
        # Rapfi: (_pos >> 5) - BOARD_BOUNDARY
        return (self._pos >> 5) - BOARD_BOUNDARY

    def __int__(self) -> int:
        """Returns the internal integer representation of the Pos."""
        return self._pos

    def is_valid_raw(self) -> bool:
        """
        Checks if the raw _pos value is within the full board array bounds,
        including PASS. Corresponds to Rapfi's `Pos::valid()`.
        """
        # Rapfi: _pos >= PASS._pos && _pos < FULL_BOARD_END._pos
        return Pos.PASS._pos <= self._pos < Pos.FULL_BOARD_END._pos

    def move_index(self) -> int:
        """
        Returns a 0-indexed integer suitable for indexing flat arrays
        representing the MAX_BOARD_SIZE playable area.
        Pos.PASS maps to MAX_BOARD_SIZE * MAX_BOARD_SIZE.
        """
        if 0 <= self.x < MAX_BOARD_SIZE and 0 <= self.y < MAX_BOARD_SIZE:
            return self.y * MAX_BOARD_SIZE + self.x
        if self == Pos.PASS:
            return MAX_BOARD_SIZE * MAX_BOARD_SIZE # Index for PASS (MAX_MOVES - 1)
        raise ValueError(f"move_index() called on invalid Pos (not on board or PASS): {self}")

    def is_on_board(self, board_width: int, board_height: int) -> bool:
        """Checks if the Pos (x,y) is within the given board dimensions."""
        # Rapfi: Pos::isInBoard
        cur_x, cur_y = self.x, self.y
        return 0 <= cur_x < board_width and 0 <= cur_y < board_height

    @staticmethod
    def chebyshev_distance(p1: 'Pos', p2: 'Pos') -> int:
        """Calculates the Chebyshev distance (max coordinate difference) between two Pos."""
        # Rapfi: Pos::distance
        if p1._pos <= Pos.NONE._pos or p2._pos <= Pos.NONE._pos: # Using internal _pos for PASS/NONE check
            return -1 # Or raise error, as distance is ill-defined
        x_dist = abs(p1.x - p2.x)
        y_dist = abs(p1.y - p2.y)
        return max(x_dist, y_dist)

    @staticmethod
    def line_distance(p1: 'Pos', p2: 'Pos') -> int:
        """
        Calculates distance if p1 and p2 are on the same line (horizontal, vertical, diagonal).
        Returns Chebyshev distance if on a line, otherwise FULL_BOARD_SIZE (as a sentinel for "not on line").
        Returns -1 if either Pos is NONE or PASS (before NONE).
        """
        if p1._pos <= Pos.NONE._pos or p2._pos <= Pos.NONE._pos:
            return -1

        x_delta = p1.x - p2.x
        y_delta = p1.y - p2.y

        if x_delta == 0: # Vertical line
            return abs(y_delta)
        elif y_delta == 0: # Horizontal line
            return abs(x_delta)
        # For diagonal, abs(x_delta) must equal abs(y_delta)
        elif abs(x_delta) == abs(y_delta): # Diagonal line
            return abs(x_delta) # or abs(y_delta), they are the same
        else:
            return FULL_BOARD_SIZE # Not on a line, using Rapfi's sentinel

    def __repr__(self) -> str:
        if self._pos == Pos.NONE._pos: # Compare with class attribute's _pos
            return "Pos.NONE"
        if self._pos == Pos.PASS._pos:
            return "Pos.PASS"
        return f"Pos({self.x}, {self.y} [raw:{self._pos}])"

    def __hash__(self) -> int:
        return self._pos

    def __eq__(self, other) -> bool:
        if isinstance(other, Pos):
            return self._pos == other._pos
        return False

    def __lt__(self, other) -> bool:
        # Comparison is based on the internal _pos value, like in C++
        if isinstance(other, Pos):
            return self._pos < other._pos
        return NotImplemented

    def __add__(self, offset: Direction | int) -> 'Pos':
        """Adds a Direction (offset) or an integer to the Pos."""
        if isinstance(offset, Direction):
            return Pos(self._pos + offset.value)
        elif isinstance(offset, int):
            return Pos(self._pos + offset)
        return NotImplemented

    def __sub__(self, offset: Direction | int) -> 'Pos':
        """Subtracts a Direction (offset) or an integer from the Pos."""
        if isinstance(offset, Direction):
            return Pos(self._pos - offset.value)
        elif isinstance(offset, int):
            return Pos(self._pos - offset)
        return NotImplemented

    def __iadd__(self, offset: Direction | int) -> 'Pos':
        """In-place addition of a Direction (offset) or an integer."""
        if isinstance(offset, Direction):
            self._pos += offset.value
            return self
        elif isinstance(offset, int):
            self._pos += offset
            return self
        return NotImplemented

    def __isub__(self, offset: Direction | int) -> 'Pos':
        """In-place subtraction of a Direction (offset) or an integer."""
        if isinstance(offset, Direction):
            self._pos -= offset.value
            return self
        elif isinstance(offset, int):
            self._pos -= offset
            return self
        return NotImplemented

# --- Special Pos instances (must be defined after class definition) ---
# Raw _pos values are taken directly from Rapfi's definitions.
Pos.NONE = Pos(0)    # Corresponds to (0,0) in Pos(x,y) if BOARD_BOUNDARY made them 0,0 with raw 0.
                     # Rapfi: Pos::NONE {0}. The coords would be (-B,-B) if 0 is the raw val for (0,0).
                     # If Pos::NONE is _pos=0, then x=(-5), y=(-5). This matches Rapfi's behavior
                     # for operations like p1 <= Pos::NONE.

Pos.PASS = Pos(-1)   # Rapfi: Pos::PASS {-1}. x=26, y=-6 if interpreted via x(),y().
                     # It's a sentinel value.

Pos.FULL_BOARD_START = Pos(0) # Rapfi: Pos::FULL_BOARD_START {0}. Same as Pos.NONE's raw value.
Pos.FULL_BOARD_END = Pos(FULL_BOARD_CELL_COUNT) # Raw value.

# -------------------------------------------------

class TransformType(Enum):
    """Represents one of the eight board symmetries for a square board."""
    IDENTITY = 0    # (x, y) -> (x, y)
    ROTATE_90 = 1   # (x, y) -> (y, s - x)
    ROTATE_180 = 2  # (x, y) -> (s - x, s - y)
    ROTATE_270 = 3  # (x, y) -> (s - y, x)
    FLIP_X = 4      # (x, y) -> (x, s - y) (Flip over horizontal mid-line)
    FLIP_Y = 5      # (x, y) -> (s - x, y) (Flip over vertical mid-line)
    FLIP_XY = 6     # (x, y) -> (y, x)       (Flip over main diagonal y=x)
    FLIP_YX = 7     # (x, y) -> (s - y, s - x) (Flip over anti-diagonal y=s-x)
    TRANS_NB = 8

def is_rectangle_transform(t: TransformType) -> bool:
    """Checks if a transform type is applicable to a non-square rectangle without changing dimensions."""
    return t in (TransformType.IDENTITY, TransformType.ROTATE_180,
                  TransformType.FLIP_X, TransformType.FLIP_Y)

def apply_transform(pos: Pos, size_x: int, transform_type: TransformType, size_y: int | None = None) -> Pos:
    """
    Applies a geometric transform to a Pos.
    - If size_y is None, assumes a square board (size_x by size_x).
    - For rectangular boards (size_y is not None), only dimension-preserving transforms are applied.
      Other transforms will return the original Pos.
    """
    if pos == Pos.PASS or pos == Pos.NONE: # Transformations don't apply to PASS/NONE
        return pos

    is_square = (size_y is None) or (size_x == size_y)
    if size_y is None:
        size_y = size_x # Square board

    x, y = pos.x, pos.y
    # s is 'size - 1' for the relevant dimension
    sx, sy = size_x - 1, size_y - 1

    if is_square:
        s = sx # size_x - 1 == size_y - 1
        if transform_type == TransformType.IDENTITY:   return Pos(x, y)
        elif transform_type == TransformType.ROTATE_90:  return Pos(y, s - x)
        elif transform_type == TransformType.ROTATE_180: return Pos(s - x, s - y)
        elif transform_type == TransformType.ROTATE_270: return Pos(s - y, x)
        elif transform_type == TransformType.FLIP_X:     return Pos(x, s - y)
        elif transform_type == TransformType.FLIP_Y:     return Pos(s - x, y)
        elif transform_type == TransformType.FLIP_XY:    return Pos(y, x)
        elif transform_type == TransformType.FLIP_YX:    return Pos(s - y, s - x)
        else:
            raise ValueError(f"Unknown transform type: {transform_type}")
    else: # Rectangular board
        if transform_type == TransformType.IDENTITY:    return Pos(x, y)
        elif transform_type == TransformType.ROTATE_180:  return Pos(sx - x, sy - y)
        elif transform_type == TransformType.FLIP_X:      return Pos(x, sy - y)
        elif transform_type == TransformType.FLIP_Y:      return Pos(sx - x, y)
        else:
            # For non-square preserving transforms on a rectangle, Rapfi returns original.
            return Pos(x, y)

# -------------------------------------------------
# Direction ranges (tuples of integer offsets)
# These correspond to the int16_t values stored in C++ Direction arrays.
# arithmetic with Direction enum members results in integers.

RANGE_LINE2 = tuple(d.value if isinstance(d, Direction) else int(d) for d in [
    Direction.UP_LEFT * 2, Direction.UP * 2, Direction.UP_RIGHT * 2,
    Direction.UP_LEFT, Direction.UP, Direction.UP_RIGHT,
    Direction.LEFT * 2, Direction.LEFT,
    Direction.RIGHT * 2, Direction.RIGHT,
    Direction.DOWN_LEFT, Direction.DOWN, Direction.DOWN_RIGHT,
    Direction.DOWN_LEFT * 2, Direction.DOWN * 2, Direction.DOWN_RIGHT * 2,
])

RANGE_SQUARE2 = tuple(d.value if isinstance(d, Direction) else int(d) for d in [
    Direction.UP_LEFT * 2, Direction.UP_LEFT + Direction.UP, Direction.UP * 2, Direction.UP_RIGHT + Direction.UP, Direction.UP_RIGHT * 2,
    Direction.UP_LEFT + Direction.LEFT, Direction.UP_LEFT, Direction.UP, Direction.UP_RIGHT, Direction.UP_RIGHT + Direction.RIGHT,
    Direction.LEFT * 2, Direction.LEFT,
    Direction.RIGHT, Direction.RIGHT * 2,
    Direction.DOWN_LEFT + Direction.LEFT, Direction.DOWN_LEFT, Direction.DOWN, Direction.DOWN_RIGHT, Direction.DOWN_RIGHT + Direction.RIGHT,
    Direction.DOWN_LEFT * 2, Direction.DOWN_LEFT + Direction.DOWN, Direction.DOWN * 2, Direction.DOWN_RIGHT + Direction.DOWN, Direction.DOWN_RIGHT * 2,
])

RANGE_SQUARE2_LINE3 = tuple(d.value if isinstance(d, Direction) else int(d) for d in [
    Direction.UP_LEFT * 3, Direction.UP * 3, Direction.UP_RIGHT * 3,
    Direction.UP_LEFT * 2, Direction.UP_LEFT + Direction.UP, Direction.UP * 2, Direction.UP_RIGHT + Direction.UP, Direction.UP_RIGHT * 2,
    Direction.UP_LEFT + Direction.LEFT, Direction.UP_LEFT, Direction.UP, Direction.UP_RIGHT, Direction.UP_RIGHT + Direction.RIGHT,
    Direction.LEFT * 3, Direction.LEFT * 2, Direction.LEFT,
    Direction.RIGHT, Direction.RIGHT * 2, Direction.RIGHT * 3,
    Direction.DOWN_LEFT + Direction.LEFT, Direction.DOWN_LEFT, Direction.DOWN, Direction.DOWN_RIGHT, Direction.DOWN_RIGHT + Direction.RIGHT,
    Direction.DOWN_LEFT * 2, Direction.DOWN_LEFT + Direction.DOWN, Direction.DOWN * 2, Direction.DOWN_RIGHT + Direction.DOWN, Direction.DOWN_RIGHT * 2,
    Direction.DOWN_LEFT * 3, Direction.DOWN * 3, Direction.DOWN_RIGHT * 3,
])

RANGE_SQUARE3 = tuple(d.value if isinstance(d, Direction) else int(d) for d in [
    Direction.UP_LEFT * 3, Direction.UP_LEFT * 2 + Direction.UP, Direction.UP_LEFT + Direction.UP * 2, Direction.UP * 3, Direction.UP_RIGHT + Direction.UP * 2, Direction.UP_RIGHT * 2 + Direction.UP, Direction.UP_RIGHT * 3,
    Direction.UP_LEFT * 2 + Direction.LEFT, Direction.UP_LEFT * 2, Direction.UP_LEFT + Direction.UP, Direction.UP * 2, Direction.UP_RIGHT + Direction.UP, Direction.UP_RIGHT * 2, Direction.UP_RIGHT * 2 + Direction.RIGHT,
    Direction.UP_LEFT + Direction.LEFT * 2, Direction.UP_LEFT + Direction.LEFT, Direction.UP_LEFT, Direction.UP, Direction.UP_RIGHT, Direction.UP_RIGHT + Direction.RIGHT, Direction.UP_RIGHT + Direction.RIGHT * 2,
    Direction.LEFT * 3, Direction.LEFT * 2, Direction.LEFT,
    Direction.RIGHT, Direction.RIGHT * 2, Direction.RIGHT * 3,
    Direction.DOWN_LEFT + Direction.LEFT * 2, Direction.DOWN_LEFT + Direction.LEFT, Direction.DOWN_LEFT, Direction.DOWN, Direction.DOWN_RIGHT, Direction.DOWN_RIGHT + Direction.RIGHT, Direction.DOWN_RIGHT + Direction.RIGHT * 2,
    Direction.DOWN_LEFT * 2 + Direction.LEFT, Direction.DOWN_LEFT * 2, Direction.DOWN_LEFT + Direction.DOWN, Direction.DOWN * 2, Direction.DOWN_RIGHT + Direction.DOWN, Direction.DOWN_RIGHT * 2, Direction.DOWN_RIGHT * 2 + Direction.RIGHT,
    Direction.DOWN_LEFT * 3, Direction.DOWN_LEFT * 2 + Direction.DOWN, Direction.DOWN_LEFT + Direction.DOWN * 2, Direction.DOWN * 3, Direction.DOWN_RIGHT + Direction.DOWN * 2, Direction.DOWN_RIGHT * 2 + Direction.DOWN, Direction.DOWN_RIGHT * 3,
])

RANGE_LINE4 = tuple(d.value if isinstance(d, Direction) else int(d) for d in [
    Direction.UP_LEFT * 4, Direction.UP * 4, Direction.UP_RIGHT * 4,
    Direction.UP_LEFT * 3, Direction.UP * 3, Direction.UP_RIGHT * 3,
    Direction.UP_LEFT * 2, Direction.UP * 2, Direction.UP_RIGHT * 2,
    Direction.UP_LEFT, Direction.UP, Direction.UP_RIGHT,
    Direction.LEFT * 4, Direction.LEFT * 3, Direction.LEFT * 2, Direction.LEFT,
    Direction.RIGHT, Direction.RIGHT * 2, Direction.RIGHT * 3, Direction.RIGHT * 4,
    Direction.DOWN_LEFT, Direction.DOWN, Direction.DOWN_RIGHT,
    Direction.DOWN_LEFT * 2, Direction.DOWN * 2, Direction.DOWN_RIGHT * 2,
    Direction.DOWN_LEFT * 3, Direction.DOWN * 3, Direction.DOWN_RIGHT * 3,
    Direction.DOWN_LEFT * 4, Direction.DOWN * 4, Direction.DOWN_RIGHT * 4,
])

RANGE_SQUARE2_LINE4 = tuple(d.value if isinstance(d, Direction) else int(d) for d in [
    Direction.UP_LEFT * 4, Direction.UP * 4, Direction.UP_RIGHT * 4,
    Direction.UP_LEFT * 3, Direction.UP * 3, Direction.UP_RIGHT * 3,
    Direction.UP_LEFT * 2, Direction.UP_LEFT + Direction.UP, Direction.UP * 2, Direction.UP_RIGHT + Direction.UP, Direction.UP_RIGHT * 2,
    Direction.UP_LEFT + Direction.LEFT, Direction.UP_LEFT, Direction.UP, Direction.UP_RIGHT, Direction.UP_RIGHT + Direction.RIGHT,
    Direction.LEFT * 4, Direction.LEFT * 3, Direction.LEFT * 2, Direction.LEFT,
    Direction.RIGHT, Direction.RIGHT * 2, Direction.RIGHT * 3, Direction.RIGHT * 4,
    Direction.DOWN_LEFT + Direction.LEFT, Direction.DOWN_LEFT, Direction.DOWN, Direction.DOWN_RIGHT, Direction.DOWN_RIGHT + Direction.RIGHT,
    Direction.DOWN_LEFT * 2, Direction.DOWN_LEFT + Direction.DOWN, Direction.DOWN * 2, Direction.DOWN_RIGHT + Direction.DOWN, Direction.DOWN_RIGHT * 2,
    Direction.DOWN_LEFT * 3, Direction.DOWN * 3, Direction.DOWN_RIGHT * 3,
    Direction.DOWN_LEFT * 4, Direction.DOWN * 4, Direction.DOWN_RIGHT * 4,
])

RANGE_SQUARE3_LINE4 = tuple(d.value if isinstance(d, Direction) else int(d) for d in [
    Direction.UP_LEFT * 4, Direction.UP * 4, Direction.UP_RIGHT * 4,
    Direction.UP_LEFT * 3, Direction.UP_LEFT * 2 + Direction.UP, Direction.UP_LEFT + Direction.UP * 2, Direction.UP * 3, Direction.UP_RIGHT + Direction.UP * 2, Direction.UP_RIGHT * 2 + Direction.UP, Direction.UP_RIGHT * 3,
    Direction.UP_LEFT * 2 + Direction.LEFT, Direction.UP_LEFT * 2, Direction.UP_LEFT + Direction.UP, Direction.UP * 2, Direction.UP_RIGHT + Direction.UP, Direction.UP_RIGHT * 2, Direction.UP_RIGHT * 2 + Direction.RIGHT,
    Direction.UP_LEFT + Direction.LEFT * 2, Direction.UP_LEFT + Direction.LEFT, Direction.UP_LEFT, Direction.UP, Direction.UP_RIGHT, Direction.UP_RIGHT + Direction.RIGHT, Direction.UP_RIGHT + Direction.RIGHT * 2,
    Direction.LEFT * 4, Direction.LEFT * 3, Direction.LEFT * 2, Direction.LEFT,
    Direction.RIGHT, Direction.RIGHT * 2, Direction.RIGHT * 3, Direction.RIGHT * 4,
    Direction.DOWN_LEFT + Direction.LEFT * 2, Direction.DOWN_LEFT + Direction.LEFT, Direction.DOWN_LEFT, Direction.DOWN, Direction.DOWN_RIGHT, Direction.DOWN_RIGHT + Direction.RIGHT, Direction.DOWN_RIGHT + Direction.RIGHT * 2,
    Direction.DOWN_LEFT * 2 + Direction.LEFT, Direction.DOWN_LEFT * 2, Direction.DOWN_LEFT + Direction.DOWN, Direction.DOWN * 2, Direction.DOWN_RIGHT + Direction.DOWN, Direction.DOWN_RIGHT * 2, Direction.DOWN_RIGHT * 2 + Direction.RIGHT,
    Direction.DOWN_LEFT * 3, Direction.DOWN_LEFT * 2 + Direction.DOWN, Direction.DOWN_LEFT + Direction.DOWN * 2, Direction.DOWN * 3, Direction.DOWN_RIGHT + Direction.DOWN * 2, Direction.DOWN_RIGHT * 2 + Direction.DOWN, Direction.DOWN_RIGHT * 3,
    Direction.DOWN_LEFT * 4, Direction.DOWN * 4, Direction.DOWN_RIGHT * 4,
])

RANGE_SQUARE4 = tuple(d.value if isinstance(d, Direction) else int(d) for d in [
    Direction.UP_LEFT * 4, Direction.UP_LEFT * 3 + Direction.UP, Direction.UP_LEFT * 2 + Direction.UP * 2, Direction.UP_LEFT + Direction.UP * 3, Direction.UP * 4, Direction.UP_RIGHT + Direction.UP * 3, Direction.UP_RIGHT * 2 + Direction.UP * 2, Direction.UP_RIGHT * 3 + Direction.UP, Direction.UP_RIGHT * 4,
    Direction.UP_LEFT * 3 + Direction.LEFT, Direction.UP_LEFT * 3, Direction.UP_LEFT * 2 + Direction.UP, Direction.UP_LEFT + Direction.UP * 2, Direction.UP * 3, Direction.UP_RIGHT + Direction.UP * 2, Direction.UP_RIGHT * 2 + Direction.UP, Direction.UP_RIGHT * 3, Direction.UP_RIGHT * 3 + Direction.RIGHT,
    Direction.UP_LEFT * 2 + Direction.LEFT * 2, Direction.UP_LEFT * 2 + Direction.LEFT, Direction.UP_LEFT * 2, Direction.UP_LEFT + Direction.UP, Direction.UP * 2, Direction.UP_RIGHT + Direction.UP, Direction.UP_RIGHT * 2, Direction.UP_RIGHT * 2 + Direction.RIGHT, Direction.UP_RIGHT * 2 + Direction.RIGHT * 2,
    Direction.UP_LEFT + Direction.LEFT * 3, Direction.UP_LEFT + Direction.LEFT * 2, Direction.UP_LEFT + Direction.LEFT, Direction.UP_LEFT, Direction.UP, Direction.UP_RIGHT, Direction.UP_RIGHT + Direction.RIGHT, Direction.UP_RIGHT + Direction.RIGHT * 2, Direction.UP_RIGHT + Direction.RIGHT * 3,
    Direction.LEFT * 4, Direction.LEFT * 3, Direction.LEFT * 2, Direction.LEFT,
    Direction.RIGHT, Direction.RIGHT * 2, Direction.RIGHT * 3, Direction.RIGHT * 4,
    Direction.DOWN_LEFT + Direction.LEFT * 3, Direction.DOWN_LEFT + Direction.LEFT * 2, Direction.DOWN_LEFT + Direction.LEFT, Direction.DOWN_LEFT, Direction.DOWN, Direction.DOWN_RIGHT, Direction.DOWN_RIGHT + Direction.RIGHT, Direction.DOWN_RIGHT + Direction.RIGHT * 2, Direction.DOWN_RIGHT + Direction.RIGHT * 3,
    Direction.DOWN_LEFT * 2 + Direction.LEFT * 2, Direction.DOWN_LEFT * 2 + Direction.LEFT, Direction.DOWN_LEFT * 2, Direction.DOWN_LEFT + Direction.DOWN, Direction.DOWN * 2, Direction.DOWN_RIGHT + Direction.DOWN, Direction.DOWN_RIGHT * 2, Direction.DOWN_RIGHT * 2 + Direction.RIGHT, Direction.DOWN_RIGHT * 2 + Direction.RIGHT * 2,
    Direction.DOWN_LEFT * 3 + Direction.LEFT, Direction.DOWN_LEFT * 3, Direction.DOWN_LEFT * 2 + Direction.DOWN, Direction.DOWN_LEFT + Direction.DOWN * 2, Direction.DOWN * 3, Direction.DOWN_RIGHT + Direction.DOWN * 2, Direction.DOWN_RIGHT * 2 + Direction.DOWN, Direction.DOWN_RIGHT * 3, Direction.DOWN_RIGHT * 3 + Direction.RIGHT,
    Direction.DOWN_LEFT * 4, Direction.DOWN_LEFT * 3 + Direction.DOWN, Direction.DOWN_LEFT * 2 + Direction.DOWN * 2, Direction.DOWN_LEFT + Direction.DOWN * 3, Direction.DOWN * 4, Direction.DOWN_RIGHT + Direction.DOWN * 3, Direction.DOWN_RIGHT * 2 + Direction.DOWN * 2, Direction.DOWN_RIGHT * 3 + Direction.DOWN, Direction.DOWN_RIGHT * 4,
])


if __name__ == '__main__':
    # Test Pos
    p1 = Pos(3, 4) # (x,y) coordinates
    print(f"p1: {p1}, x={p1.x}, y={p1.y}, int_val={int(p1)}")
    p_raw = Pos(int(p1)) # from raw value
    assert p1 == p_raw

    p_none = Pos.NONE
    p_pass = Pos.PASS
    print(f"p_none: {p_none}, x={p_none.x}, y={p_none.y}") # Expect x=-5, y=-5 for raw _pos=0
    print(f"p_pass: {p_pass}, x={p_pass.x}, y={p_pass.y}") # Expect x=26, y=-6 for raw _pos=-1

    assert Pos(0,0).x == 0 and Pos(0,0).y == 0
    assert Pos(MAX_BOARD_SIZE-1, MAX_BOARD_SIZE-1).x == MAX_BOARD_SIZE-1

    print(f"p1 + Direction.UP: {p1 + Direction.UP}")
    p1_copy = Pos(p1.x, p1.y)
    p1_copy += Direction.RIGHT
    print(f"p1_copy after += Direction.RIGHT: {p1_copy}")
    assert p1_copy.x == p1.x + 1

    print(f"Is p1 valid_raw? {p1.is_valid_raw()}")
    print(f"Is p_pass valid_raw? {p_pass.is_valid_raw()}")
    assert Pos.PASS.is_valid_raw()
    assert Pos.NONE.is_valid_raw()
    assert Pos(FULL_BOARD_CELL_COUNT -1).is_valid_raw()
    assert not Pos(FULL_BOARD_CELL_COUNT).is_valid_raw()


    board_center = Pos(MAX_BOARD_SIZE // 2, MAX_BOARD_SIZE // 2)
    print(f"Board center: {board_center}")
    print(f"Board center move_index: {board_center.move_index()}")
    print(f"Pos.PASS move_index: {Pos.PASS.move_index()}")
    assert Pos.PASS.move_index() == MAX_BOARD_SIZE * MAX_BOARD_SIZE

    # Test distance
    p2 = Pos(board_center.x, board_center.y + 2)
    p3 = Pos(board_center.x + 2, board_center.y + 2)
    print(f"Distance center to p2 (Chebychev): {Pos.chebyshev_distance(board_center, p2)}")
    assert Pos.chebyshev_distance(board_center, p2) == 2
    print(f"Distance p2 to p3 (Chebychev): {Pos.chebyshev_distance(p2, p3)}")
    assert Pos.chebyshev_distance(p2, p3) == 2

    # Test line_distance
    print(f"Line distance center to p2 (Vertical): {Pos.line_distance(board_center, p2)}")
    assert Pos.line_distance(board_center, p2) == 2
    p_horiz = Pos(board_center.x + 3, board_center.y)
    print(f"Line distance center to p_horiz (Horizontal): {Pos.line_distance(board_center, p_horiz)}")
    assert Pos.line_distance(board_center, p_horiz) == 3
    p_diag = Pos(board_center.x + 4, board_center.y + 4)
    print(f"Line distance center to p_diag (Diagonal): {Pos.line_distance(board_center, p_diag)}")
    assert Pos.line_distance(board_center, p_diag) == 4
    p_off_line = Pos(board_center.x + 1, board_center.y + 2)
    print(f"Line distance center to p_off_line (Off line): {Pos.line_distance(board_center, p_off_line)}")
    assert Pos.line_distance(board_center, p_off_line) == FULL_BOARD_SIZE

    # Test transforms
    test_pos = Pos(2, 3)
    board_s = 15 # Example square board size for transforms (0-14 coords)
    print(f"Original Pos: {test_pos} for board size {board_s}")
    transformed_identity = apply_transform(test_pos, board_s, TransformType.IDENTITY)
    assert transformed_identity == test_pos
    transformed_r90 = apply_transform(test_pos, board_s, TransformType.ROTATE_90)
    assert transformed_r90.x == 3 and transformed_r90.y == (board_s - 1 - 2)
    print(f"Transform ROTATE_90: {transformed_r90}")


    # Test rectangular transforms
    rect_pos = Pos(2,3)
    size_x_rect, size_y_rect = 10, 8
    print(f"Original Pos: {rect_pos} for rect board ({size_x_rect}x{size_y_rect})")
    transformed_rect_flip_x = apply_transform(rect_pos, size_x_rect, TransformType.FLIP_X, size_y_rect)
    assert transformed_rect_flip_x.x == 2 and transformed_rect_flip_x.y == (size_y_rect - 1 - 3)
    print(f"Rect Transform FLIP_X: {transformed_rect_flip_x}")
    # Test a non-rectangle-preserving transform on a rectangle (should return original)
    transformed_rect_p_rot90 = apply_transform(rect_pos, size_x_rect, TransformType.ROTATE_90, size_y_rect)
    assert transformed_rect_p_rot90 == rect_pos
    print(f"Rect Transform ROTATE_90 (should be original): {transformed_rect_p_rot90}")

    print(f"DIRECTIONS tuple: {DIRECTIONS}")
    print(f"A sample from RANGE_SQUARE2[0]: {RANGE_SQUARE2[0]} (type: {type(RANGE_SQUARE2[0])})")
    assert isinstance(RANGE_SQUARE2[0], int)

    # Hashing and equality
    pos_set = {Pos(1,1), Pos(1,2), Pos(1,1)}
    print(f"Set of Pos: {pos_set}")
    assert len(pos_set) == 2
    assert Pos(1,1) == Pos(1,1)
    assert Pos(1,1) != Pos(1,0)

    # Check total_ordering (based on _pos values)
    assert Pos(0,0) < Pos(1,0) # x=0,y=0 vs x=1,y=0
    assert Pos(MAX_BOARD_SIZE-1, MAX_BOARD_SIZE-1) > Pos(0,0)
    print("pos.py tests completed.")