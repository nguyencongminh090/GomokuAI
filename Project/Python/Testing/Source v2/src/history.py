"""
History heuristic tables for move ordering in search.
Based on Rapfi's history.h.
"""

from typing import List, Tuple, Generic, TypeVar, Dict, cast # Added cast

from .types import Color, Score, Value, SIDE_NB
from .pos import Pos, MAX_MOVES # MAX_MOVES for indexing by Pos.move_index()

# Generic type for history table values
T = TypeVar('T')

# Max value for history scores to prevent overflow if using fixed-size integers,
# Python ints are arbitrary precision, but good to have a conceptual limit for scaling.
HISTORY_MAX_SCORE = 32000 # Similar to Value.VALUE_MATE range but for heuristics

class HistoryTable(Generic[T]):
    """
    Generic base for history tables.
    Manages a table and provides basic operations like clear, update, get.
    In Rapfi, History is a template class.
    """
    def __init__(self, num_indices1: int, num_indices2: int, default_value: T):
        """
        Initializes a 2D history table.
        Example: table[index1][index2]
        """
        self.default_value: T = default_value
        self.table: List[List[T]] = [
            [default_value for _ in range(num_indices2)] for _ in range(num_indices1)
        ]

    def init(self, value: T) -> None:
        """Initializes all entries in the table to a specific value."""
        num_idx1 = len(self.table)
        if num_idx1 == 0: return # Nothing to init
        num_idx2 = len(self.table[0])
        
        self.table = [[value for _ in range(num_idx2)] for _ in range(num_idx1)]


    def get(self, idx1: int, idx2: int) -> T:
        """Gets the history value for the given indices."""
        try:
            return self.table[idx1][idx2]
        except IndexError:
            return self.default_value 


    def update(self, idx1: int, idx2: int, bonus: int, max_value: int = HISTORY_MAX_SCORE) -> None:
        """
        Updates the history score by adding a bonus using Rapfi's formula.
        Ensures the score does not exceed max_value (or go below -max_value).
        Assumes table stores integer-like scores.
        """
        try:
            # Ensure current_score is treated as int for calculation
            # If T is not int (e.g. CounterMovePair), this update logic is not suitable
            # This method is primarily for integer-scored history tables.
            if not isinstance(self.table[idx1][idx2], (int, float)): # Check if arithmetic is possible
                 # print(f"Warning: HistoryTable value at ({idx1}, {idx2}) is not numeric for update. Type: {type(self.table[idx1][idx2])}")
                return

            current_score = cast(int, self.table[idx1][idx2]) 
            
            abs_bonus = abs(bonus)
            if max_value == 0: # Avoid division by zero, ensure it's a positive scaling factor
                effective_max_val_for_decay = 1 
            else:
                effective_max_val_for_decay = abs(max_value) # Use abs for safety if max_value could be negative

            # Rapfi formula: h[key] += bonus - h[key] * abs(bonus) / Max;
            # Ensure integer division for the decay term if that's the C++ behavior
            decay_term = (current_score * abs_bonus) // effective_max_val_for_decay
            update_val = bonus - decay_term
            new_score = current_score + update_val
            
            # Cap at max_value and some reasonable minimum (e.g., -max_value)
            self.table[idx1][idx2] = cast(T, max(-max_value, min(new_score, max_value)))

        except IndexError:
            # print(f"Warning: HistoryTable update index out of bounds: ({idx1}, {idx2})")
            pass
        # TypeError should be caught by the isinstance check above for numeric types


class MainHistory(HistoryTable[Score]): 
    """
    Main history table, indexed by [color][to_pos.move_index()].
    Stores Score (int).
    """
    def __init__(self):
        super().__init__(SIDE_NB, MAX_MOVES, default_value=Score(0))

    def get_score(self, color: Color, move_pos: Pos) -> Score:
        return self.get(color.value, move_pos.move_index())

    def update_score(self, color: Color, move_pos: Pos, bonus: int):
        self.update(color.value, move_pos.move_index(), bonus, HISTORY_MAX_SCORE)


CounterMovePair = Tuple[Pos, Pos] 

class CounterMoveHistory(HistoryTable[CounterMovePair]):
    """
    Countermove history table.
    Indexed by opponent's previous move: table[0][opponent_prev_move.move_index()]
    Stores a CounterMovePair (tuple of two good counter moves).
    """
    def __init__(self):
        super().__init__(1, MAX_MOVES, default_value=(Pos.NONE, Pos.NONE))

    def get_counter_moves(self, opponent_prev_move: Pos) -> CounterMovePair:
        return self.get(0, opponent_prev_move.move_index())

    def update_counter_moves(self, opponent_prev_move: Pos, new_good_counter: Pos):
        """Updates the counter moves for opponent_prev_move with new_good_counter."""
        # This method does not use the numeric self.update from HistoryTable.
        # It directly manipulates the CounterMovePair.
        try:
            current_pair = self.table[0][opponent_prev_move.move_index()]
            if new_good_counter != Pos.NONE and new_good_counter != current_pair[0]:
                self.table[0][opponent_prev_move.move_index()] = (new_good_counter, current_pair[0])
        except IndexError:
            # print(f"Warning: CounterMoveHistory update index out of bounds for move: {opponent_prev_move}")
            pass


if __name__ == '__main__':
    print("--- History Heuristics Tests ---")

    main_hist = MainHistory()
    main_hist.init(Score(0)) 

    move1 = Pos(7,7)
    move2 = Pos(7,8)

    print(f"Initial score for B at {move1}: {main_hist.get_score(Color.BLACK, move1)}")
    assert main_hist.get_score(Color.BLACK, move1) == 0

    main_hist.update_score(Color.BLACK, move1, 100)
    print(f"Score for B at {move1} after +100: {main_hist.get_score(Color.BLACK, move1)}")
    assert main_hist.get_score(Color.BLACK, move1) == 100

    main_hist.update_score(Color.BLACK, move1, 50)
    # Expected: 100 + (50 - (100*50)//32000) = 100 + (50 - 0) = 150
    print(f"Score for B at {move1} after +50: {main_hist.get_score(Color.BLACK, move1)}")
    assert main_hist.get_score(Color.BLACK, move1) == 150
    
    main_hist.init(Score(0))
    main_hist.update_score(Color.WHITE, move2, HISTORY_MAX_SCORE // 2) 
    assert main_hist.get_score(Color.WHITE, move2) == HISTORY_MAX_SCORE // 2

    main_hist.update_score(Color.WHITE, move2, HISTORY_MAX_SCORE // 2) 
    # Expected: 16000 + (16000 - (16000*16000)//32000) = 16000 + (16000 - 8000) = 24000
    print(f"Score for W at {move2} after two +16000 bonuses: {main_hist.get_score(Color.WHITE, move2)}")
    assert main_hist.get_score(Color.WHITE, move2) == 24000

    cm_hist = CounterMoveHistory()
    # Default value is (Pos.NONE, Pos.NONE), so init isn't strictly needed if constructor works
    # cm_hist.init((Pos.NONE, Pos.NONE)) 

    opponent_move = Pos(5,5)
    counter1 = Pos(6,6)
    counter2 = Pos(4,4)

    print(f"Initial counters for opp_move {opponent_move}: {cm_hist.get_counter_moves(opponent_move)}")
    assert cm_hist.get_counter_moves(opponent_move) == (Pos.NONE, Pos.NONE)

    cm_hist.update_counter_moves(opponent_move, counter1)
    print(f"Counters after adding {counter1}: {cm_hist.get_counter_moves(opponent_move)}")
    assert cm_hist.get_counter_moves(opponent_move) == (counter1, Pos.NONE)

    cm_hist.update_counter_moves(opponent_move, counter2)
    print(f"Counters after adding {counter2}: {cm_hist.get_counter_moves(opponent_move)}")
    assert cm_hist.get_counter_moves(opponent_move) == (counter2, counter1)
    
    cm_hist.update_counter_moves(opponent_move, counter2)
    print(f"Counters after re-adding {counter2}: {cm_hist.get_counter_moves(opponent_move)}")
    assert cm_hist.get_counter_moves(opponent_move) == (counter2, counter1)

    print("History heuristics tests completed.")