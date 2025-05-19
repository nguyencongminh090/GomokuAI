"""
Transposition Table (TT) for the Gomoku engine.
Based on Rapfi's hashtable.h.
"""
import sys
from typing import List, Optional, Tuple

from .types import HashKey, Value, Depth, Bound, stored_value_to_search_value, search_value_to_stored_value
from .pos import Pos
from .utils import is_power_of_two, floor_power_of_two

# Default number of entries per TT bucket (cluster)
TT_CLUSTER_SIZE = 4 # As in Rapfi's TTEntryCluster

class TTEntry:
    """
    Represents a single entry in the Transposition Table.
    """
    # Rapfi uses bitfields for some members to save space.
    # Python objects have more overhead, so direct bitfield packing isn't done
    # unless we use ctypes or similar. We'll store them as regular attributes.

    def __init__(self):
        self.key: HashKey = 0           # Lower 32 bits of the Zobrist key (upper 32 used for indexing)
        self.move: Pos = Pos.NONE       # Best move or refutation move
        self.value: int = Value.VALUE_NONE.value # Stored value (needs ply adjustment via search_value_to_stored_value)
        self.eval_value: int = Value.VALUE_NONE.value # Static evaluation of the position
        
        # Depth is float in types.py, but stored as int in TT (scaled)
        # Rapfi: depth: 8 (for -3 to 252, scaled from float)
        # Let's store as int, representing Rapfi's scaled depth.
        # Actual Depth is Depth(stored_depth_val / DEPTH_STEP + DEPTH_NONE) approx.
        # For now, treat as integer depth plies.
        self.depth: int = 0 # Stored as integer plies remaining or absolute depth
        
        self.generation: int = 0 # 8 bits in Rapfi, cycle counter for TT entry age
        self.bound: Bound = Bound.BOUND_NONE # 2 bits in Rapfi
        self.pv_node: bool = False # 1 bit in Rapfi (is_pv)

    def is_valid(self, key_to_check: HashKey, current_generation: int) -> bool:
        """Checks if the entry is valid for the given key and generation."""
        # Key check: Rapfi stores only lower 32 bits of key.
        # Assumes key_to_check is the full Zobrist key.
        # The upper part of key_to_check is used for bucket index.
        # The lower part (self.key) must match the lower part of key_to_check.
        # For simplicity, if we store full key, we can just compare.
        # If we follow Rapfi's split:
        # key32 = key_to_check & 0xFFFFFFFF
        # return self.key == key32 and self.generation == current_generation

        # For now, let's assume self.key might store the full key or a significant part.
        # If self.key is 0, it's an empty/invalid entry.
        if self.key == 0: # Unused entry
            return False
            
        # Check if the stored key matches the Zobrist key we're probing with.
        # Rapfi's TT indexing uses `key >> 32` for bucket, `key & 0xFFFFFFFF` for stored tag.
        # For now, we assume full key is stored or checked appropriately.
        # This check depends on how TT.probe will use the full key.
        # Let's assume for now: if self.key is set, it should match the probe key.
        # This will be refined when TT.probe is implemented. For now, key matching logic is simplified.
        
        # Crucial: check generation to ensure it's not an old entry from a previous search/state.
        return self.generation == current_generation


class TranspositionTable:
    """
    Manages TT entries.
    """
    # Rapfi: TTEntry* table = nullptr; uint64_t entryCount = 0; uint8_t generation = 0;
    # In Python, table will be a list of lists (buckets of TTEntry).
    
    def __init__(self, size_kb: int = 0): # size_kb = 0 means it might be set later
        self.table: List[List[TTEntry]] = []
        self.num_buckets: int = 0 # Was entryCount in Rapfi, but it's num clusters/buckets
        self.generation: int = 0  # uint8_t in Rapfi, wraps around 0-255
        self.hash_mask: int = 0   # Used for bucket indexing: (key >> key_shift) & hash_mask

        # Rapfi uses key >> 32 for index. This implies key_shift = 32.
        # This detail is important for how Zobrist keys are split for index vs. tag.
        self.key_shift_for_index: int = 32 # How many lower bits of key are for tag

        if size_kb > 0:
            self.resize(size_kb)

    def resize(self, size_kb: int) -> None:
        """Resizes the TT to approximately size_kb kilobytes."""
        if size_kb == 0:
            self.table = []
            self.num_buckets = 0
            self.hash_mask = 0
            return

        # Estimate entry size. TTEntry in Python will be larger than C++ bitfield struct.
        # Let's make a rough estimate. A Python object might be ~50-100 bytes.
        # A TTEntryCluster (4 entries) in C++ is sizeof(TTEntry)*4 + padding.
        # sizeof(TTEntry) in Rapfi (key32,move16,val16,eval16,dep8,gen8,bound2,pv1)
        # approx (4+2+2+2+1+1+0.25+0.125) = ~10.5 bytes per entry. Cluster ~42 bytes.
        # Python: key(8)+move(obj_ref_or_int)+value(int)+eval(int)+depth(int)+gen(int)+bound(obj)+pv(bool)
        # Assume one TTEntry object is roughly 64 bytes for estimation.
        # One bucket (cluster) of TT_CLUSTER_SIZE entries: TT_CLUSTER_SIZE * 64 bytes.
        bytes_per_bucket = TT_CLUSTER_SIZE * 64 # Rough estimate
        
        target_bytes = size_kb * 1024
        if target_bytes < bytes_per_bucket: # Ensure at least one bucket
            num_b = 1
        else:
            num_b = target_bytes // bytes_per_bucket

        # Number of buckets should be a power of two for efficient masking
        if num_b > 0 and not is_power_of_two(num_b):
            self.num_buckets = floor_power_of_two(num_b)
        elif num_b == 0 : # If target_bytes was too small for even one estimated bucket
            self.num_buckets = 0 # Or a minimum like 1 if size_kb > 0
        else:
            self.num_buckets = num_b


        if self.num_buckets == 0 and size_kb > 0: # If resize was requested but results in 0 buckets
            self.num_buckets = 1 # Ensure at least one bucket if TT is enabled
        
        if self.num_buckets > 0:
            self.hash_mask = self.num_buckets - 1
            self.table = [[TTEntry() for _ in range(TT_CLUSTER_SIZE)] for _ in range(self.num_buckets)]
            self.generation = 0 # Reset generation on resize
            # print(f"TT resized. Buckets: {self.num_buckets}, Total entries: {self.num_buckets * TT_CLUSTER_SIZE}", file=sys.stderr)
        else:
            self.table = []
            self.hash_mask = 0
            # print("TT disabled or size too small.", file=sys.stderr)


    def clear(self) -> None:
        """Clears all entries in the TT by resetting them."""
        # A faster clear is to just increment generation if other data can be stale.
        # But for a full clear, reinitialize entries.
        for bucket in self.table:
            for i in range(TT_CLUSTER_SIZE):
                bucket[i] = TTEntry() # Create new, default entries
        self.generation = 0

    def inc_generation(self) -> None:
        """Increments the generation counter, effectively aging out old entries."""
        self.generation = (self.generation + 1) & 0xFF # Wrap around 255 (like uint8_t)

    def _get_bucket_index(self, key: HashKey) -> int:
        """Calculates the bucket index from the full Zobrist key."""
        if self.num_buckets == 0:
            return 0 # Should not be called if TT is empty
        # Rapfi uses upper bits for index: key >> 32
        # Then masks with num_buckets - 1 (hash_mask)
        return (key >> self.key_shift_for_index) & self.hash_mask
    
    def _get_key_tag(self, key: HashKey) -> HashKey:
        """Gets the part of the key stored as a tag in TTEntry."""
        # Rapfi uses lower 32 bits: key & 0xFFFFFFFF
        # For now, our TTEntry.key might store more.
        # This depends on TTEntry.is_valid implementation detail.
        # Let's assume TTEntry.key will store this tag.
        tag_mask = (1 << self.key_shift_for_index) - 1
        return key & tag_mask


    def probe(self, key: HashKey, current_search_ply: int) -> \
            Tuple[bool, int, int, bool, Bound, Pos, Depth]: # Changed Value return types to int
        """
        Probes the TT for an entry matching the key.
        Returns: (hit, value_as_int, eval_value_as_int, is_pv, bound, move, depth)
        Value and depth are adjusted from stored TT format.
        """
        if not self.table or self.num_buckets == 0:
            # Ensure Pos.NONE and Value.VALUE_NONE are their respective types if needed by caller
            # but here, the int value of VALUE_NONE is fine.
            return False, Value.VALUE_NONE.value, Value.VALUE_NONE.value, False, Bound.BOUND_NONE, Pos.NONE, 0.0

        bucket_idx = self._get_bucket_index(key)
        key_tag_to_match = self._get_key_tag(key)
        bucket = self.table[bucket_idx]
        best_entry: Optional[TTEntry] = None

        for entry in bucket:
            if entry.key == key_tag_to_match and entry.generation == self.generation:
                best_entry = entry
                break 
        
        if best_entry:
            search_val_int = stored_value_to_search_value(best_entry.value, current_search_ply)
            eval_val_int = best_entry.eval_value # Already stored as int
            search_depth = float(best_entry.depth)

            return (True,
                    search_val_int, # Return as int
                    eval_val_int,   # Return as int
                    best_entry.pv_node,
                    best_entry.bound,
                    best_entry.move,
                    search_depth)
        
        return False, Value.VALUE_NONE.value, Value.VALUE_NONE.value, False, Bound.BOUND_NONE, Pos.NONE, 0.0


    def store(self, key: HashKey, value: int, eval_value: int, is_pv: bool,
              bound: Bound, move: Pos, depth: int, current_search_ply: int):
        """
        Stores an entry in the TT.
        `value` and `depth` are search values/depths and will be converted for storage.
        `depth` here is integer plies (e.g. remaining depth).
        """
        if not self.table or self.num_buckets == 0:
            return

        bucket_idx = self._get_bucket_index(key)
        key_tag_to_store = self._get_key_tag(key)
        bucket = self.table[bucket_idx]

        # Convert search value and depth to storable format
        # `value` is already an int, `search_value_to_stored_value` expects an int.
        stored_val = search_value_to_stored_value(value, current_search_ply) # REMOVED Value(value) cast

        stored_depth = max(0, min(int(depth), 255)) 

        replace_idx = -1
        min_quality_score = float('inf') 

        for i, entry in enumerate(bucket):
            if entry.key == 0 or entry.generation != self.generation: 
                replace_idx = i
                break 

            entry_quality = entry.depth * 100 
            if entry.pv_node: entry_quality += 10
            if entry.bound == Bound.BOUND_EXACT: entry_quality += 5
            
            if entry_quality < min_quality_score :
                min_quality_score = entry_quality
                replace_idx = i
        
        if replace_idx != -1: 
            if bucket[replace_idx].key == 0 or bucket[replace_idx].generation != self.generation or \
               stored_depth >= bucket[replace_idx].depth:
                entry_to_write = bucket[replace_idx]
                entry_to_write.key = key_tag_to_store
                entry_to_write.move = move
                entry_to_write.value = stored_val
                entry_to_write.eval_value = eval_value
                entry_to_write.depth = stored_depth
                entry_to_write.generation = self.generation
                entry_to_write.bound = bound
                entry_to_write.pv_node = is_pv
            # else: new entry is shallower than all existing current-gen entries, don't store.
        # else: Should not happen if loop always finds a replace_idx if bucket not empty.
        # If bucket was full of high-quality entries, this logic might discard good new entries.
        # A common strategy is "always replace" if a slot was chosen, or
        # "replace if deeper OR (same_depth AND (new_is_pv_or_exact AND old_is_not))"

    def hash_size_kb(self) -> int:
        """Estimates current TT size in KB based on number of buckets."""
        if self.num_buckets == 0: return 0
        bytes_per_bucket_rough = TT_CLUSTER_SIZE * 64 # Must match resize() estimate
        return (self.num_buckets * bytes_per_bucket_rough) // 1024

if __name__ == '__main__':
    print("--- HashTable Tests ---")
    tt = TranspositionTable()
    print(f"Initial TT size: {tt.hash_size_kb()} KB, Buckets: {tt.num_buckets}")

    tt.resize(1024) # 1MB TT
    print(f"TT size after resize(1024): {tt.hash_size_kb()} KB, Buckets: {tt.num_buckets}")
    assert tt.num_buckets > 0
    initial_generation = tt.generation

    key1: HashKey = 0x123456789ABCDEF0
    val1_int: int = Value.VALUE_MATE.value - 10 # mate_in(10) as int
    eval_val1_int: int = 100
    depth1: int = 15 # plies
    ply1: int = 5

    tt.store(key1, val1_int, eval_val1_int, True, Bound.BOUND_EXACT, Pos(1,1), depth1, ply1)
    print(f"Stored key {key1:x} with value {val1_int}, depth {depth1}")

    # Probe result will now have ints for value and eval_value
    hit, p_val_int, p_eval_int, p_pv, p_bound, p_move, p_depth = tt.probe(key1, ply1)
    
    # Use the get_score_name helper from wincheck.py if you want named output for tests
    # For now, just print the int values.
    print(f"Probe key {key1:x}: hit={hit}, val={p_val_int}, eval={p_eval_int}, pv={p_pv}, bound={p_bound.name}, move={p_move}, depth={p_depth}")
    
    assert hit
    assert p_val_int == val1_int 
    assert p_eval_int == eval_val1_int
    assert p_pv
    assert p_bound == Bound.BOUND_EXACT
    assert p_move == Pos(1,1)
    assert int(p_depth) == depth1

    key2: HashKey = 0xFEDCBA9876543210
    hit_miss, _, _, _, _, _, _ = tt.probe(key2, ply1)
    print(f"Probe key {key2:x}: hit={hit_miss}")
    assert not hit_miss

    tt.inc_generation()
    print(f"TT generation incremented to: {tt.generation}")
    assert tt.generation == (initial_generation + 1) & 0xFF

    hit_after_gen_inc, _, _, _, _, _, _ = tt.probe(key1, ply1)
    print(f"Probe key {key1:x} after generation increment: hit={hit_after_gen_inc}")
    assert not hit_after_gen_inc 

    val2_int = Value.VALUE_MATE.value - 12
    tt.store(key1, val2_int, eval_val1_int + 50, True, Bound.BOUND_LOWER, Pos(2,2), depth1 + 2, ply1)
    hit_new_gen, p_val_new, p_eval_new, _, p_bound_new, p_move_new, p_depth_new = tt.probe(key1, ply1)
    print(f"Probe key {key1:x} with new generation: hit={hit_new_gen}, val={p_val_new}, eval={p_eval_new}, bound={p_bound_new.name}, move={p_move_new}, depth={p_depth_new}")
    assert hit_new_gen
    assert p_val_new == val2_int
    assert p_eval_new == eval_val1_int + 50
    assert p_bound_new == Bound.BOUND_LOWER
    assert p_move_new == Pos(2,2)
    assert int(p_depth_new) == depth1 + 2

    tt.clear()
    print(f"TT cleared. Generation: {tt.generation}")
    assert tt.generation == 0
    hit_after_clear, _, _, _, _, _, _ = tt.probe(key1, ply1)
    print(f"Probe key {key1:x} after clear: hit={hit_after_clear}")
    assert not hit_after_clear
    
    print("HashTable tests completed.")