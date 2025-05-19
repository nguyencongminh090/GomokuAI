"""
Manages the search stack, storing information for each ply in the search tree.
Based on Rapfi's searchstack.h.
"""
from typing import List, Optional, TYPE_CHECKING

from .types import Value, Pattern4, Color, SIDE_NB # Import basic types
from .pos import Pos # Import Pos

# Max search depth for allocating stack space (can be from config)
from . import config as engine_config # To get MAX_SEARCH_DEPTH potentially

if TYPE_CHECKING:
    # For type hints if Board and StateInfo are used directly in StackEntry,
    # though Rapfi's Stack mostly stores simpler data.
    # from .board import Board
    # from .board_utils import StateInfo
    pass

# Conceptual MAX_PLY from Rapfi (often around 60-128 for practical search)
# This should be consistent with the maximum depth the search can reach.
# Using engine_config.MAX_SEARCH_DEPTH for consistency.
MAX_PLY_STACK = engine_config.MAX_SEARCH_DEPTH + 10 # Add some buffer

class StackEntry:
    """
    Represents the information stored for a single ply in the search stack.
    Mirrors the 'Stack' struct in Rapfi's searchstack.h (which is ss, ss+1 etc.).
    """
    def __init__(self):
        self.ply: int = 0 # Current ply in the search tree (0 for root)

        # PV line for this node. pv[0] is the best move from this node.
        # pv is terminated by Pos.NONE.
        # Max PV length can be MAX_PLY_STACK.
        self.pv: List[Pos] = [Pos.NONE] * MAX_PLY_STACK # Pre-allocate for efficiency

        self.current_move: Pos = Pos.NONE # Move currently being searched from this node
        self.skip_move: Pos = Pos.NONE    # Move to be excluded in singular extension search

        # Killer moves: good quiet moves that caused a beta cutoff at this ply
        # Stored as [Pos, Pos] in Rapfi (ss->killers[0], ss->killers[1])
        self.killers: List[Pos] = [Pos.NONE, Pos.NONE]

        self.static_eval: int = Value.VALUE_ZERO.value # Static evaluation of the position at this node
        self.stat_score: int = 0 # Statistical score for LMR, from history table

        # tt_pv: True if this node or one of its ancestors is part of a PV line
        #        found in the TT, or if it's on the current search's PV.
        self.tt_pv: bool = False

        # move_count: Number of moves searched so far from this node
        self.move_count: int = 0

        # move_p4[color_val]: Pattern4 type of current_move for BLACK and WHITE
        self.move_p4: List[Pattern4] = [Pattern4.NONE] * SIDE_NB

        # db_value_depth: Depth of a value retrieved from database for the PV from this node.
        # Used to compare if a TT entry is deeper/better than a DB entry.
        # INT16_MIN in Rapfi.
        self.db_value_depth: int = -32768 # Smallest int16

        # num_null_moves: Number of consecutive null moves made to reach this ply.
        self.num_null_moves: int = 0
        
        # extra_extension: Accumulation of non-1.0 extensions (e.g., singular extensions)
        self.extra_extension: float = 0.0

        # db_child_written: Flag used in database interaction.
        # True if any child node of the current node has written to the database.
        self.db_child_written: bool = False


    def reset(self):
        """Resets the entry to a default state, typically for reuse."""
        # self.pv = [Pos.NONE] * MAX_PLY_STACK # Already list of NONEs if not modified
        for i in range(len(self.pv)): # Explicitly reset PV
            self.pv[i] = Pos.NONE
            
        self.current_move = Pos.NONE
        self.skip_move = Pos.NONE
        self.killers = [Pos.NONE, Pos.NONE]
        self.static_eval = Value.VALUE_ZERO.value
        self.stat_score = 0
        self.tt_pv = False
        self.move_count = 0
        self.move_p4 = [Pattern4.NONE] * SIDE_NB
        self.db_value_depth = -32768
        self.num_null_moves = 0
        self.extra_extension = 0.0
        self.db_child_written = False
        # self.ply is set by the manager

    def update_pv(self, best_move: Pos, next_stack_entry: 'StackEntry'):
        """
        Updates the PV for this node.
        Assumes `best_move` is the best move found from this node,
        and `next_stack_entry.pv` contains the PV from the child node after making `best_move`.
        """
        self.pv[0] = best_move
        idx = 1
        # Copy PV from child, ensuring we don't exceed MAX_PLY_STACK in self.pv
        while next_stack_entry.pv[idx-1] != Pos.NONE and idx < MAX_PLY_STACK:
            self.pv[idx] = next_stack_entry.pv[idx-1]
            idx += 1
        if idx < MAX_PLY_STACK: # Terminate with Pos.NONE if space allows
            self.pv[idx] = Pos.NONE


    def is_killer(self, move: Pos) -> bool:
        """Checks if the given move is one of the killer moves for this ply."""
        return move == self.killers[0] or move == self.killers[1]

    def add_killer(self, move: Pos):
        """Adds a move as a killer move, shifting existing ones if different."""
        if move != Pos.NONE and move != self.killers[0]:
            self.killers[1] = self.killers[0]
            self.killers[0] = move


class SearchStackManager:
    """
    Manages a list of StackEntry objects, providing access like an array (ss, ss+1).
    """
    def __init__(self, max_depth: int = MAX_PLY_STACK, initial_static_eval: int = Value.VALUE_ZERO.value):
        # Pre-allocate stack entries. +1 for ss[0] at root, up to ss[max_depth]
        self.stack: List[StackEntry] = [StackEntry() for _ in range(max_depth + 1)]
        for i, entry in enumerate(self.stack):
            entry.ply = i # Set the ply for each entry

        # Initialize the root stack entry (ss[0] or self.stack[0])
        # Rapfi: StackArray(MAX_PLY, initValue) -> StackArray sets ss[0].staticEval = -initValue
        # The sign flip for static_eval at root (ss[0]) depends on how the search loop starts.
        # If search starts with -search(..., -beta, -alpha), then ss[0].static_eval can be positive.
        # For now, initialize based on the initial_static_eval provided.
        # Rapfi's StackArray constructor: (ss-1)->staticEval = initValue; (ss-1) used as base.
        # And ss[0].staticEval = -initValue for the first actual search ply.
        # This needs careful alignment with the search function's perspective.
        # For now, let's assume `initial_static_eval` is from current player's view.
        # The search often uses -(ss-1).staticEval for current node's eval.
        # If stack[0] represents ply 0 (root node from current player's view),
        # then stack[0].static_eval would be `initial_static_eval`.
        # stack[-1] concept (ss-1) would be tricky.
        # Let's assume stack[0] is before any move is made for the search call.
        if self.stack:
            self.stack[0].static_eval = initial_static_eval
            # Other root initializations if necessary (e.g., tt_pv for root)
            self.stack[0].tt_pv = True


    def get_entry(self, ply: int) -> StackEntry:
        """Gets the StackEntry for a given ply."""
        if 0 <= ply < len(self.stack):
            return self.stack[ply]
        else:
            # This case should ideally be prevented by MAX_PLY_STACK checks in search.
            # If it happens, it means search depth exceeded allocation.
            raise IndexError(f"Search stack ply {ply} out of bounds (max: {len(self.stack)-1})")

    def root_stack_entry(self) -> StackEntry:
        """Returns the stack entry for the root of the search (ply 0)."""
        return self.stack[0]

    def reset_ply_for_search(self, ply: int):
        """
        Resets entries from 'ply' onwards, useful when starting a new search iteration
        or exploring a new branch where history beyond the current path is irrelevant.
        Rapfi doesn't explicitly have a "reset_ply_for_search" at this manager level;
        it's usually handled by the search loop overwriting ss+1, ss+2 data.
        This can be a helper if needed.
        """
        for i in range(ply, len(self.stack)):
            self.stack[i].reset()
            self.stack[i].ply = i # Re-affirm ply


if __name__ == '__main__':
    print("--- Search Stack Tests ---")

    # Test StackEntry
    se = StackEntry()
    se.ply = 5
    se.current_move = Pos(1,1)
    se.add_killer(Pos(2,2))
    se.add_killer(Pos(3,3)) # (3,3) becomes killer[0], (2,2) becomes killer[1]
    print(f"StackEntry: ply={se.ply}, current_move={se.current_move}, killers={se.killers}")
    assert se.killers[0] == Pos(3,3)
    assert se.killers[1] == Pos(2,2)
    assert se.is_killer(Pos(3,3))
    assert not se.is_killer(Pos(1,1))

    se_child = StackEntry()
    se_child.pv[0] = Pos(4,4)
    se_child.pv[1] = Pos(5,5)
    # pv[2] is Pos.NONE by default

    se.update_pv(Pos(3,3), se_child)
    print(f"PV for se after update: {[p for p in se.pv if p != Pos.NONE]}")
    assert se.pv[0] == Pos(3,3)
    assert se.pv[1] == Pos(4,4)
    assert se.pv[2] == Pos(5,5)
    assert se.pv[3] == Pos.NONE

    se.reset()
    print(f"StackEntry after reset: ply={se.ply}, killers={se.killers}, pv[0]={se.pv[0]}")
    assert se.killers[0] == Pos.NONE
    assert se.pv[0] == Pos.NONE
    # Ply is not reset by StackEntry.reset(), but by SearchStackManager.reset_ply_for_search

    # Test SearchStackManager
    stack_manager = SearchStackManager(max_depth=10, initial_static_eval=100)
    root_entry = stack_manager.root_stack_entry()
    print(f"Root entry: ply={root_entry.ply}, static_eval={root_entry.static_eval}, tt_pv={root_entry.tt_pv}")
    assert root_entry.ply == 0
    assert root_entry.static_eval == 100
    assert root_entry.tt_pv is True

    entry_ply3 = stack_manager.get_entry(3)
    entry_ply3.current_move = Pos(8,8)
    print(f"Entry at ply 3: ply={entry_ply3.ply}, current_move={entry_ply3.current_move}")
    assert entry_ply3.ply == 3
    
    stack_manager.reset_ply_for_search(2) # Resets entries from ply 2 onwards
    entry_ply3_after_reset = stack_manager.get_entry(3)
    print(f"Entry at ply 3 after reset from ply 2: current_move={entry_ply3_after_reset.current_move}")
    assert entry_ply3_after_reset.current_move == Pos.NONE
    assert entry_ply3_after_reset.ply == 3 # Ply number should be preserved

    # Check if entry at ply 1 (before reset point) is untouched
    entry_ply1 = stack_manager.get_entry(1)
    entry_ply1.current_move = Pos(1,1) # Modify before reset
    stack_manager.reset_ply_for_search(2)
    assert stack_manager.get_entry(1).current_move == Pos(1,1)


    print("Search stack tests completed.")