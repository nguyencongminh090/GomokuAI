"""
Time control and management for the search algorithm.
Based on Rapfi's timecontrol.h and parts of searcher.cpp / ABSearcher.
"""
import time
from typing import NamedTuple, Optional

from .utils import now, Time # Using Time = int (milliseconds) from utils
from .types import Value # For previousBestValue
from . import config as engine_config # For time management options

# To avoid circular dependency if SearchOptions is defined in search.py later
class SearchOptionsPlaceholder(NamedTuple):
    turn_time: Time = 0
    match_time: Time = 0
    time_left: Time = 0
    inc_time: Time = 0 # Increment per move
    moves_to_go: int = 0 # Moves left to next time control, or estimated for game

class TimeInfo(NamedTuple):
    """Holds game timing information."""
    ply: int
    moves_left_in_game: int # Estimated moves remaining in the whole game

class StopConditions(NamedTuple):
    """Parameters to check if search should stop, from ABSearcher::search."""
    current_search_depth: int
    last_best_move_change_depth: int
    current_best_value: Value # Actually int
    previous_search_best_value: Value # Actually int
    previous_time_reduction_factor: float
    avg_best_move_changes_this_iter: float


class TimeControl:
    """
    Manages thinking time for a move.
    """
    def __init__(self):
        self.start_time: Time = 0         # Time when search for current move started
        self.allocated_time: Time = 0     # Time allocated for this move based on settings
        self.optimal_time: Time = 0       # Ideal time to spend
        self.maximum_time: Time = 0       # Absolute maximum time to spend
        self.is_time_limited: bool = True # Whether any time limit is active

        # From Rapfi's ABSearcher, kept per game
        self.previous_best_value: int = Value.VALUE_NONE.value
        self.previous_time_reduction_factor: float = 1.0

    def init(self, turn_time_ms: Time, match_time_ms: Time, time_left_ms: Time,
             time_info: TimeInfo, inc_time_ms: Time = 0, moves_to_go: int = 0):
        """
        Initializes time control for the current move.
        Args:
            turn_time_ms: Total time for this turn (if fixed per turn).
            match_time_ms: Total match time (if used).
            time_left_ms: Time remaining on the clock for the current player.
            time_info: TimeInfo struct with ply and estimated moves left in game.
            inc_time_ms: Increment per move.
            moves_to_go: Moves until next time control (e.g., X moves in Y minutes).
                         If 0, it's sudden death or based on estimated game length.
        """
        self.start_time = now()
        self.is_time_limited = True

        if turn_time_ms > 0: # Fixed time per turn
            self.optimal_time = turn_time_ms
            self.maximum_time = turn_time_ms
        elif time_left_ms > 0: # Time control based on remaining time
            # Estimate moves remaining in the current time control period
            # If moves_to_go is not given, use estimated moves_left_in_game
            if moves_to_go <= 0:
                # Rapfi uses MatchSpace, MatchSpaceMin. Default moves_to_go is often around 20-30 for midgame.
                # Let's use a simpler heuristic if moves_to_go is not provided:
                # Divide remaining time by a fraction of estimated remaining game moves.
                # Rapfi's logic is more complex, involving MatchSpace.
                # moves_to_go_effective = max(1, time_info.moves_left_in_game // 2) # Simple heuristic
                
                # Rapfi's TimeCtrl::init logic for moves_to_go:
                # `moves = MovesToGo == 0 ? int(MatchSpace - ply() / 12.0f) : MovesToGo;`
                # `moves = std::max({int(MatchSpaceMin), moves, NumMovesTilNextTC});`
                # Assuming similar logic if moves_to_go is not directly provided:
                # This uses an estimate based on game progress.
                # For simplicity, if moves_to_go is not passed, assume a portion of game_moves_left.
                # Let's say, if sudden death, allocate for N moves.
                moves_to_go_effective = max(1, moves_to_go if moves_to_go > 0 else int(engine_config.MATCH_SPACE))
            else:
                moves_to_go_effective = max(1, moves_to_go)

            # Add increment to time_left before dividing
            time_for_control_period = time_left_ms + (moves_to_go_effective * inc_time_ms)
            
            self.allocated_time = time_for_control_period // moves_to_go_effective
            
            # Subtract a small reservation for communication delay
            self.allocated_time -= engine_config.TURN_TIME_RESERVED
            self.allocated_time = max(1, self.allocated_time) # Ensure at least 1ms

            # Optimal time is usually a fraction of allocated time (e.g., 50-80%)
            self.optimal_time = self.allocated_time // 2 # Simple: spend half
            
            # Maximum time can be more, e.g., up to 2-3x optimal, but not exceeding allocated_time
            # or a certain fraction of total time_left.
            self.maximum_time = self.allocated_time
            # Rapfi has more complex maximum_time calculation, can be higher if time is abundant.
            # `Maximum = std::min(Allocated * 3, TimeLeft - Reserved);`
            self.maximum_time = min(self.allocated_time * 3, time_left_ms - engine_config.TURN_TIME_RESERVED)
            self.maximum_time = max(self.optimal_time, self.maximum_time) # Max must be >= Optimal
            self.maximum_time = max(1, self.maximum_time)


        else: # No time limits (e.g., analysis mode or infinite time)
            self.is_time_limited = False
            self.optimal_time = Time(3_600_000) # 1 hour (effectively very long)
            self.maximum_time = Time(3_600_000 * 24) # 1 day
            self.allocated_time = self.optimal_time

        # Ensure optimal is not more than maximum
        self.optimal_time = min(self.optimal_time, self.maximum_time)
        self.optimal_time = max(1, self.optimal_time) # Ensure at least 1ms

    def elapsed(self) -> Time:
        """Returns time elapsed since search started for this move."""
        return now() - self.start_time

    def is_time_up(self, check_optimal: bool = False) -> bool:
        """Checks if allocated/optimal time is up."""
        if not self.is_time_limited:
            return False
        
        target_time = self.optimal_time if check_optimal else self.maximum_time
        return self.elapsed() >= target_time

    def check_stop(self, conditions: StopConditions, current_time_reduction_factor: float) -> bool:
        """
        More advanced check to see if search should stop based on stability and time.
        This mimics part of ABSearcher's time management logic in the iterative deepening loop.
        Updates `current_time_reduction_factor` (passed by reference in C++ via pointer/ref).
        Since Python passes objects by assignment, this function will return the new factor.
        Returns: (should_stop, new_time_reduction_factor)
        """
        if not self.is_time_limited:
            return False, current_time_reduction_factor

        elapsed_ms = self.elapsed()
        new_reduction_factor = current_time_reduction_factor # Start with current

        # Rapfi's TimeCtl::checkStop logic is complex. Simplified version:
        # 1. If past maximum, definitely stop.
        if elapsed_ms >= self.maximum_time:
            return True, new_reduction_factor

        # 2. Calculate a dynamic "stop_time" based on optimal time and stability.
        #    If best move hasn't changed for a while, can spend more time (reduction_factor < 1).
        #    If best move changes often, spend less (reduction_factor > 1).
        
        # Heuristic for stability (from Rapfi's time management)
        # How many iterations since best move changed
        stable_depth_count = conditions.current_search_depth - conditions.last_best_move_change_depth
        
        # Adjust reduction factor based on stability
        # If stable, reduce the factor (spend more time relative to optimal)
        # If unstable (avg_best_move_changes > threshold), increase factor (spend less)
        if stable_depth_count >= 2 and conditions.avg_best_move_changes_this_iter < 0.5:
            new_reduction_factor *= (1.0 - engine_config.BESTMOVE_STABLE_REDUCTION_SCALE * stable_depth_count)
        elif conditions.avg_best_move_changes_this_iter > 1.5 : # Very unstable
             new_reduction_factor *= 1.25
        
        # Consider previous search value vs current
        # If value is falling, might want to search more (decrease reduction factor)
        if conditions.current_best_value < conditions.previous_search_best_value - 50: # 50 cp drop
            new_reduction_factor *= 0.9
        
        # Apply previous reduction factor with some decay
        new_reduction_factor = (new_reduction_factor + 
                                conditions.previous_time_reduction_factor**engine_config.BESTMOVE_STABLE_PREV_REDUCTION_POW
                               ) / 2.0
        
        new_reduction_factor = max(0.25, min(new_reduction_factor, 2.0)) # Bounds for factor

        stop_time = self.optimal_time * new_reduction_factor
        
        # Advanced stop: if nearing maximum_time or a large fraction of allocated_time
        if elapsed_ms >= self.maximum_time * engine_config.ADVANCED_STOP_RATIO:
            return True, new_reduction_factor
            
        if elapsed_ms >= stop_time:
            return True, new_reduction_factor
            
        return False, new_reduction_factor


# ... (rest of time_control.py up to if __name__ == '__main__') ...

if __name__ == '__main__':
    print("--- TimeControl Tests ---")
    tc = TimeControl()

    # Test 1: Fixed time per turn
    print("\nTest 1: Fixed time per turn (5s)")
    game_time_info1 = TimeInfo(ply=10, moves_left_in_game=30) 
    tc.init(turn_time_ms=5000, match_time_ms=0, time_left_ms=0, time_info=game_time_info1)
    print(f"Optimal: {tc.optimal_time}ms, Maximum: {tc.maximum_time}ms, Allocated: {tc.allocated_time}ms")
    assert tc.optimal_time == 5000
    assert tc.maximum_time == 5000
    assert tc.is_time_limited

    # Test 2: Time left, moves to go specified
    print("\nTest 2: Time left (60s), 20 moves to go, 2s increment")
    game_time_info2 = TimeInfo(ply=20, moves_left_in_game=40)
    tc.init(turn_time_ms=0, match_time_ms=180000, time_left_ms=60000, 
            time_info=game_time_info2, inc_time_ms=2000, moves_to_go=20)
    print(f"Optimal: {tc.optimal_time}ms, Maximum: {tc.maximum_time}ms, Allocated: {tc.allocated_time}ms")
    assert tc.allocated_time == 4970
    assert tc.optimal_time == 2485
    assert tc.maximum_time == 14910 
    assert tc.is_time_limited

    # Test 3: No time limit (analysis)
    print("\nTest 3: No time limit")
    tc.init(turn_time_ms=0, match_time_ms=0, time_left_ms=0, time_info=game_time_info1)
    print(f"Optimal: {tc.optimal_time}ms, Maximum: {tc.maximum_time}ms")
    assert not tc.is_time_limited
    assert tc.optimal_time > 1000000 

    # Test 4: is_time_up
    print("\nTest 4: is_time_up checks")
    tc.init(turn_time_ms=100, match_time_ms=0, time_left_ms=0, time_info=game_time_info1) 
    tc.start_time = now() 
    time.sleep(0.05) 
    print(f"Elapsed: {tc.elapsed()}ms")
    assert not tc.is_time_up(check_optimal=True) 
    assert not tc.is_time_up(check_optimal=False) 
    time.sleep(0.06) 
    print(f"Elapsed: {tc.elapsed()}ms")
    assert tc.is_time_up(check_optimal=True)
    assert tc.is_time_up(check_optimal=False)

    # Test 5: check_stop basic call
    print("\nTest 5: check_stop basic call")
    tc.init(turn_time_ms=0, match_time_ms=0, time_left_ms=30000, 
            time_info=TimeInfo(ply=10, moves_left_in_game=20), inc_time_ms=0, moves_to_go=10)
    print(f"Optimal: {tc.optimal_time}ms, Maximum: {tc.maximum_time}ms, Allocated: {tc.allocated_time}ms")
    
    stop_cond = StopConditions(
        current_search_depth=5,
        last_best_move_change_depth=5, 
        current_best_value=100,                 # Pass as int
        previous_search_best_value=80,          # Pass as int
        previous_time_reduction_factor=1.0,
        avg_best_move_changes_this_iter=0.2
    )
    reduction_factor = 1.0
    tc.start_time = now()
    time.sleep(0.1) 
    should_stop, new_factor = tc.check_stop(stop_cond, reduction_factor)
    print(f"Elapsed: {tc.elapsed()}ms. Should stop: {should_stop}, New factor: {new_factor:.3f}")
    assert not should_stop 

    time.sleep(tc.optimal_time / 1000.0 * 0.8) 
    should_stop, new_factor = tc.check_stop(stop_cond, new_factor) 
    print(f"Elapsed: {tc.elapsed()}ms. Should stop: {should_stop}, New factor: {new_factor:.3f}")
    
    time.sleep(tc.optimal_time / 1000.0 * 0.4) 
    should_stop, new_factor = tc.check_stop(stop_cond, new_factor)
    print(f"Elapsed: {tc.elapsed()}ms. Should stop: {should_stop}, New factor: {new_factor:.3f}")
    
    print("TimeControl tests completed.")