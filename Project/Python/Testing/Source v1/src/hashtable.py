from .types import Pos, Value, Bound, Depth
import numpy as np
import multiprocessing as mp
import pickle
from io import BytesIO

class TTEntry:
    def __init__(self, key=0, value=0, eval=0, pv=False, bound=Bound.NONE, move=Pos.NONE, depth=0, generation=0):
        self.value16 = max(-32768, min(32767, value))
        self.eval16 = max(-32768, min(32767, eval))
        move_idx = (move.x * 15 + move.y + 1) if move != Pos.NONE else 0
        self.pv_bound_best16 = (int(pv) << 15) | (bound << 13) | (move_idx & 0x3FF)
        self.depth8 = max(0, min(255, depth + Depth.DEPTH_LOWER_BOUND))
        self.generation8 = generation % 256
        self.data = [self.value16 | (self.eval16 << 16), self.pv_bound_best16 | (self.depth8 << 16) | (self.generation8 << 24)]
        self.key32 = (key & 0xFFFFFFFF) ^ self.data[0] ^ self.data[1]

    def get_value(self):
        return self.value16

    def get_eval(self):
        return self.eval16

    def get_pv(self):
        return bool(self.pv_bound_best16 >> 15)

    def get_bound(self):
        return (self.pv_bound_best16 >> 13) & 0x3

    def get_move(self):
        move_idx = self.pv_bound_best16 & 0x3FF
        return Pos.NONE if move_idx == 0 else Pos((move_idx - 1) // 15, (move_idx - 1) % 15)

    def get_depth(self):
        return self.depth8 - Depth.DEPTH_LOWER_BOUND

    def get_generation(self):
        return self.generation8

class HashTable:
    ENTRIES_PER_BUCKET = 5

    def __init__(self, size_mb=16):
        self.size_kb = size_mb * 1024
        self.num_buckets = max(1, self.size_kb * 1024 // (self.ENTRIES_PER_BUCKET * 12))
        self.table = [[] for _ in range(self.num_buckets)]
        self.generation = 0

    def resize(self, size_mb):
        self.size_kb = size_mb * 1024
        new_num_buckets = max(1, self.size_kb * 1024 // (self.ENTRIES_PER_BUCKET * 12))
        if new_num_buckets != self.num_buckets:
            self.num_buckets = new_num_buckets
            self.table = [[] for _ in range(self.num_buckets)]
        self.generation = 0

    def clear(self):
        def clear_chunk(start, end):
            for i in range(start, end):
                self.table[i] = []

        num_threads = min(mp.cpu_count(), self.num_buckets)
        chunk_size = self.num_buckets // num_threads
        processes = []
        for i in range(num_threads):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_threads - 1 else self.num_buckets
            p = mp.Process(target=clear_chunk, args=(start, end))
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
        self.generation = 0

    def inc_generation(self):
        self.generation = (self.generation + 1) % 256

    def _index(self, key):
        """Maps a 64-bit key to a bucket index, ensuring it stays within bounds."""
        return (key * self.num_buckets) % self.num_buckets  # Safer than >> 32

    def first_entry(self, key):
        return self._index(key)

    def prefetch(self, key):
        pass

    def probe(self, hash_key, ply):
        bucket_idx = self._index(hash_key)
        key32 = hash_key & 0xFFFFFFFF
        for entry in self.table[bucket_idx]:
            if entry.key32 == key32:
                return (Value.stored_to_search(entry.get_value(), ply),
                        entry.get_eval(),
                        entry.get_pv(),
                        entry.get_bound(),
                        entry.get_move(),
                        entry.get_depth(),
                        True)
        return None, None, None, None, None, None, False

    def store(self, hash_key, value, eval, is_pv, bound, move, depth, ply):
        bucket_idx = self._index(hash_key)
        key32 = hash_key & 0xFFFFFFFF
        bucket = self.table[bucket_idx]

        replace_idx = -1
        for i, entry in enumerate(bucket):
            if entry.key32 == key32:
                if bound != Bound.EXACT and depth + 2 < entry.get_depth():
                    return
                replace_idx = i
                break
            elif replace_idx == -1 or (entry.depth8 - (self.generation - entry.generation8)) < \
                     (bucket[replace_idx].depth8 - (self.generation - bucket[replace_idx].generation8)):
                replace_idx = i

        new_entry = TTEntry(key32, value, eval, is_pv, bound, move, depth, self.generation)
        if replace_idx != -1:
            bucket[replace_idx] = new_entry
        elif len(bucket) < self.ENTRIES_PER_BUCKET:
            bucket.append(new_entry)
        else:
            bucket[replace_idx] = new_entry

    def dump(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({'num_buckets': self.num_buckets, 'generation': self.generation, 'table': self.table}, f)

    def load(self, filename):
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.num_buckets = data['num_buckets']
                self.generation = data['generation']
                self.table = data['table']
                self.size_kb = self.num_buckets * self.ENTRIES_PER_BUCKET * 12 // 1024
            return True
        except:
            return False

    def hash_usage(self):
        cnt = 0
        test_cnt = min(1024, self.num_buckets)
        for i in range(test_cnt):
            for entry in self.table[i]:
                if entry.generation8 == self.generation:
                    cnt += 1
        return (cnt * 1000) // (self.ENTRIES_PER_BUCKET * test_cnt)

    def hash_size_kb(self):
        return self.num_buckets * self.ENTRIES_PER_BUCKET * 12 // 1024