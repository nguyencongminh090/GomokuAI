"""
Zobrist hashing utilities for the Gomoku engine, based on Rapfi's hash.h and hash.cpp.
"""

# Relative imports for package structure
from .types import HashKey, Color, SIDE_NB
from .pos import FULL_BOARD_CELL_COUNT
from .utils import PRNG

# Seed for Zobrist key generation (from hash.cpp)
ZOBRIST_SEED: int = 0xa52ca39782739747

# Global Zobrist tables
zobrist_table: list[list[HashKey]] = []
zobrist_side: list[HashKey] = []

_UINT64_MASK = (1 << 64) - 1

def lc_hash(x: int) -> HashKey:
    result = (x * 6364136223846793005) + 1442695040888963407
    return result & _UINT64_MASK

def init_zobrist(seed: int = ZOBRIST_SEED) -> None:
    global zobrist_table, zobrist_side
    prng = PRNG(seed)

    zobrist_table = [[0] * FULL_BOARD_CELL_COUNT for _ in range(SIDE_NB)]
    for i in range(FULL_BOARD_CELL_COUNT):
        zobrist_table[Color.BLACK.value][i] = prng()
        zobrist_table[Color.WHITE.value][i] = prng()

    zobrist_side = [0] * SIDE_NB
    zobrist_side[Color.BLACK.value] = prng()
    zobrist_side[Color.WHITE.value] = prng()

init_zobrist()


if __name__ == '__main__':
    # This block will only execute correctly if run as a module:
    # python -m src.hash_utils
    # OR if the path manipulation for direct script running is added back
    # For simplicity with packages, prefer `python -m ...`

    print("--- Zobrist Hashing Tests (running as part of src package) ---")

    # Since init_zobrist() is called at module level, tables should be populated.
    # We might need to re-import other modules if they also rely on being part of the package
    # for their own internal imports if this __main__ block were more complex.
    # For now, accessing globals defined in this file is fine.

    if not zobrist_table or not zobrist_side:
        print("Error: Zobrist tables not initialized!")
        # This might happen if Color, SIDE_NB etc. were not available due to import issues
        # during the initial module-level call to init_zobrist().
        # Re-running init here for standalone test via -m might be an option if needed,
        # but ideally, the module-level init works because all its dependencies are met.
        # init_zobrist() # Re-call for test if module-level one failed due to import context

    print(f"Zobrist table dimensions: {len(zobrist_table)} x {len(zobrist_table[0]) if zobrist_table else 0}")
    print(f"Expected: {SIDE_NB} x {FULL_BOARD_CELL_COUNT}") # These should be available from import
    assert len(zobrist_table) == SIDE_NB
    if SIDE_NB > 0:
        assert len(zobrist_table[0]) == FULL_BOARD_CELL_COUNT

    print(f"Zobrist side table length: {len(zobrist_side)}")
    assert len(zobrist_side) == SIDE_NB

    print(f"Sample Zobrist key (BLACK, pos 0): {zobrist_table[Color.BLACK.value][0]:016x}")
    print(f"Sample Zobrist key (WHITE, pos 0): {zobrist_table[Color.WHITE.value][0]:016x}")
    print(f"Sample Zobrist side key (BLACK): {zobrist_side[Color.BLACK.value]:016x}")

    assert zobrist_table[Color.BLACK.value][0] != zobrist_table[Color.WHITE.value][0]
    assert zobrist_table[Color.BLACK.value][0] != zobrist_table[Color.BLACK.value][1]
    assert zobrist_side[Color.BLACK.value] != zobrist_side[Color.WHITE.value]

    print("LC Hash of 12345:", hex(lc_hash(12345)))

    old_black_0 = zobrist_table[Color.BLACK.value][0]
    init_zobrist(seed=ZOBRIST_SEED + 1) # Test re-init
    new_black_0 = zobrist_table[Color.BLACK.value][0]
    assert old_black_0 != new_black_0, "Zobrist keys did not change with new seed."
    print("Re-initialized with different seed, keys changed as expected.")

    init_zobrist(ZOBRIST_SEED) # Restore
    assert zobrist_table[Color.BLACK.value][0] == old_black_0

    print("hash_utils.py tests completed.")