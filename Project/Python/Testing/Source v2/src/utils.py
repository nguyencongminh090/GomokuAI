"""
Utility functions for the Gomoku engine, based on Rapfi's utils.h and utils.cpp.
"""
import time
import math
import os
import pathlib
import random
import sys
from typing import List, Tuple, TypeVar, Sequence, Any, Generator, Iterable 

# -------------------------------------------------
# Chrono time type
# Time represents a time point/period value in milliseconds
Time = int  # Alias for int, representing milliseconds

def now() -> Time:
    """Returns the current time in milliseconds since an arbitrary epoch (monotonic)."""
    # time.monotonic_ns() returns nanoseconds
    return int(time.monotonic_ns() / 1_000_000)

# -------------------------------------------------
# Math/Template helper functions

T = TypeVar('T')

def array_size(arr: Sequence[Any]) -> int:
    """Returns the size of a sequence (like a list or tuple)."""
    return len(arr)

def power(base: T, exponent: int) -> T:
    """Calculates base to the power of exponent."""
    # Python's ** operator is generally preferred and efficient.
    if not isinstance(exponent, int) or exponent < 0:
        raise ValueError("Exponent must be a non-negative integer")
    return base ** exponent

def is_power_of_two(x: int) -> bool:
    """Checks if x is a power of two."""
    if x <= 0:
        return False
    return (x & (x - 1)) == 0

def floor_log2(x: int) -> int:
    """Calculates the floor of log base 2 of x."""
    if x <= 0:
        raise ValueError("Input must be positive for log2")
    if x == 1:
        return 0
    # math.log2 returns float, int() truncates.
    # int.bit_length() - 1 is a more direct way for positive integers.
    return x.bit_length() - 1

def floor_power_of_two(x: int) -> int:
    """Returns the nearest power of two less than or equal to x."""
    if x <= 0:
        return 0 # Or raise error, depending on desired behavior for non-positives
    # Rapfi's bit manipulation method
    # x |= x >> 1
    # x |= x >> 2
    # ...
    # return x ^ (x >> 1)
    # A simpler way in Python for positive integers:
    if x == 0: return 0 # Should already be caught by x <= 0
    return 1 << (x.bit_length() - 1)


def combine_number(n: int, m: int) -> int:
    """
    Calculates combinations with repetition: (n + m - 1)! / ((n - 1)! * m!).
    This is also equivalent to math.comb(n + m - 1, m) or math.comb(n + m - 1, n - 1).
    """
    if n < 1 or m < 0: # n typically items to choose from (>=1), m items to choose (>=0)
        raise ValueError("n must be >= 1 and m >= 0 for combinations with repetition")
    if m == 0:
        return 1
    # Uses math.comb for H(n, k) = C(n+k-1, k)
    return math.comb(n + m - 1, m)

# -------------------------------------------------
# String helper functions

def trim_inplace(s: str) -> str:
    """Trims leading/trailing whitespace from a string."""
    # Python strings are immutable, so "inplace" modification isn't direct.
    # This function returns the trimmed string.
    return s.strip()

def upper_inplace(s: str) -> str:
    """Converts a string to uppercase."""
    return s.upper()

def replace_all(s: str, old_sub: str, new_sub: str) -> str:
    """Replaces all occurrences of old_sub with new_sub in s."""
    return s.replace(old_sub, new_sub)

def split_str_view(s: str, delims: str = "\n", include_empty: bool = False) -> List[str]:
    """
    Splits a string by any character in delims.
    Equivalent to Rapfi's split for std::string_view.
    Python's re.split can handle multiple delimiters naturally.
    If delims is a single character, str.split() is simpler.
    """
    if not s:
        return [""] if include_empty and not s else []

    import re
    # Construct a regex pattern for splitting: '[<delims>]'
    # Using re.escape on delims in case they contain special regex characters
    pattern = f"[{re.escape(delims)}]"
    parts = re.split(pattern, s)

    if include_empty:
        return parts
    else:
        return [part for part in parts if part]


def time_text(t_ms: Time) -> str:
    """Formats time in milliseconds to a human-readable string (ms, s, min, h)."""
    if t_ms < 0: t_ms = 0 # Treat negative times as 0 for display
    if t_ms < 10_000:  # Up to 10s
        return f"{t_ms}ms"
    elif t_ms < 1_000_000:  # Up to 1000s (approx 16 min)
        return f"{t_ms // 1000}s"
    elif t_ms < 3_600_000 * 10: # Up to 10h (original was 360_000_000 ms = 100h)
        return f"{t_ms // 60_000}min"
    else:
        return f"{t_ms // 3_600_000}h"

def nodes_text(nodes: int) -> str:
    """Formats node count to a human-readable string (K, M, G, T)."""
    if nodes < 0: nodes = 0
    if nodes < 10_000:
        return str(nodes)
    elif nodes < 10_000_000:  # 10K to 10M-1
        return f"{nodes // 1000}K"
    elif nodes < 10_000_000_000: # 10M to 10G-1
        return f"{nodes // 1_000_000}M"
    elif nodes < 10_000_000_000_000: # 10G to 10T-1
        return f"{nodes // 1_000_000_000}G"
    else:
        return f"{nodes // 1_000_000_000_000}T"

def speed_text(nodes_per_second: int) -> str:
    """Formats speed (nodes/sec) to a human-readable string (raw, K, M)."""
    if nodes_per_second < 0: nodes_per_second = 0
    if nodes_per_second < 100_000:
        return str(nodes_per_second)
    elif nodes_per_second < 100_000_000: # 100K to 100M-1
        return f"{nodes_per_second // 1000}K"
    else:
        return f"{nodes_per_second // 1_000_000}M"

# -------------------------------------------------
# Container helpers

Elem = TypeVar('Elem')
Container = TypeVar('Container', bound=Sequence)

def contains(container: Iterable[Elem], element: Elem) -> bool: # Changed from Container[Elem] to Iterable[Elem]
    """Checks if a container contains an element."""
    # Python's 'in' operator is the idiomatic way.
    return element in container

# MultiDimNativeArray and MDNativeArray are for C-style fixed-size multi-dimensional arrays.
# In Python, lists of lists are typically used. For performance-critical numeric arrays,
# NumPy is the standard. We'll skip direct translation of these for now unless a clear need arises.

# -------------------------------------------------
# Fast Pseudo Random Number Generator (PRNG)

class PRNG:
    """
    PRNG class based on SplitMix64.
    See <https://xoroshiro.di.unimi.it/splitmix64.c>
    Python integers have arbitrary precision, so we'll need to mask to 64 bits.
    """
    _UINT64_MASK = (1 << 64) - 1

    def __init__(self, seed: int | None = None):
        if seed is None:
            seed = now() # Default seed to current time in ms
        self._x: int = seed & PRNG._UINT64_MASK

    def __call__(self) -> int:
        self._x = (self._x + 0x9e3779b97f4a7c15) & PRNG._UINT64_MASK
        z = self._x
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
        z &= PRNG._UINT64_MASK
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb
        z &= PRNG._UINT64_MASK
        return (z ^ (z >> 31)) & PRNG._UINT64_MASK

    @staticmethod
    def min_val() -> int:
        return 0

    @staticmethod
    def max_val() -> int:
        return PRNG._UINT64_MASK

# -------------------------------------------------
# String encoding conversion
# Python 3 strings are natively Unicode (UTF-8 is a common encoding).
# For file I/O, explicit encoding can be specified.
# Console encoding can be more complex and platform-dependent.
# sys.stdin.encoding and sys.stdout.encoding can give hints.

def legacy_file_cp_to_utf8(s_bytes: bytes, legacy_codepage: str | None = None) -> str:
    """
    Converts bytes (presumably from a legacy file) from a specified codepage to a UTF-8 string.
    In Rapfi, Config::DatabaseLegacyFileCodePage (int) is used. Python uses string names for codecs.
    If legacy_codepage is None, it might try a system default (locale.getpreferredencoding()).
    """
    # This is a simplification. Windows GetACP() is complex.
    # A mapping from Windows Code Page IDs to Python codec names would be needed for direct port.
    # For now, user must provide a valid Python codec name.
    if legacy_codepage is None:
        # This is a guess; Python's default might be UTF-8 or locale-dependent.
        # On Windows, locale.getpreferredencoding(False) often gives the ANSI codepage.
        import locale
        legacy_codepage = locale.getpreferredencoding(False)
        if legacy_codepage is None or legacy_codepage.lower() == 'utf-8': # fallback
             legacy_codepage = "cp1252" # Common Western Windows codepage as a fallback

    try:
        return s_bytes.decode(legacy_codepage, errors='replace')
    except LookupError: # Unknown encoding
        # Fallback to trying common encodings or just returning as is with a warning
        print(f"Warning: Unknown legacy codepage '{legacy_codepage}'. Trying 'utf-8'.", file=sys.stderr)
        return s_bytes.decode('utf-8', errors='replace')


def console_cp_to_utf8(s_bytes: bytes) -> str:
    """Converts bytes from console input codepage to UTF-8 string."""
    # sys.stdin.encoding *should* be the console's input encoding.
    encoding = sys.stdin.encoding if sys.stdin.encoding else 'utf-8'
    try:
        return s_bytes.decode(encoding, errors='replace')
    except Exception as e:
        print(f"Error decoding from console encoding '{encoding}': {e}", file=sys.stderr)
        return s_bytes.decode('utf-8', errors='replace') # Fallback


def utf8_to_console_cp(utf8_str: str) -> bytes:
    """Converts a UTF-8 string to bytes using the console output codepage."""
    # sys.stdout.encoding *should* be the console's output encoding.
    encoding = sys.stdout.encoding if sys.stdout.encoding else 'utf-8'
    try:
        return utf8_str.encode(encoding, errors='replace')
    except Exception as e:
        print(f"Error encoding to console encoding '{encoding}': {e}", file=sys.stderr)
        return utf8_str.encode('utf-8', errors='replace') # Fallback


# -------------------------------------------------
# File system-related helpers
# Python's pathlib is generally preferred for modern path operations.
# It handles Unicode paths well on most systems.

def path_from_console_string(path_str: str) -> pathlib.Path:
    """
    Converts a string (assumed from console) to a pathlib.Path object.
    Python 3 generally handles Unicode paths well, so direct conversion is often fine.
    The C++ version deals with Windows-specific MultiByteToWideChar.
    """
    # In Python 3, strings are Unicode. pathlib.Path handles them.
    # If path_str truly came from console input that wasn't UTF-8 and Python
    # didn't decode it correctly via sys.stdin.encoding, problems could arise earlier.
    return pathlib.Path(path_str)

def path_to_console_string(path_obj: pathlib.Path) -> str:
    """
    Converts a pathlib.Path object to a string suitable for console output.
    This is mostly `str(path_obj)`, as Python's print will use sys.stdout.encoding.
    """
    return str(path_obj)

def list_all_files_in_dir_recursively(
    dirpath: str | pathlib.Path,
    extensions: List[str] | None = None
) -> List[str]:
    """
    Lists all files in a directory recursively, optionally filtered by extensions.
    Extensions should include the dot, e.g., [".txt", ".log"].
    Returns a list of absolute path strings.
    """
    base_path = pathlib.Path(dirpath)
    filenames: List[str] = []
    if not base_path.is_dir():
        return []

    # Ensure extensions start with a dot for consistent matching
    processed_extensions: List[str] | None = None
    if extensions:
        processed_extensions = [ext if ext.startswith('.') else '.' + ext for ext in extensions]
        processed_extensions = [ext.lower() for ext in processed_extensions]


    for p_obj in base_path.rglob("*"): # recursive glob
        if p_obj.is_file():
            if processed_extensions:
                if p_obj.suffix.lower() in processed_extensions:
                    filenames.append(str(p_obj.resolve()))
            else:
                filenames.append(str(p_obj.resolve()))
    return filenames

def make_file_list_from_path_list(
    paths: List[str],
    extensions: List[str] | None = None
) -> List[str]:
    """
    Creates a flat list of filenames from a list of paths (which can be files or dirs).
    If a path is a directory, its files are recursively added (filtered by extensions).
    """
    filenames: List[str] = []
    for path_str in paths:
        p_obj = pathlib.Path(path_str)
        if p_obj.is_dir():
            filenames.extend(list_all_files_in_dir_recursively(p_obj, extensions))
        elif p_obj.is_file():
            # If it's a file, check extension if provided
            if extensions:
                processed_extensions = [ext if ext.startswith('.') else '.' + ext for ext in extensions]
                processed_extensions = [ext.lower() for ext in processed_extensions]
                if p_obj.suffix.lower() in processed_extensions:
                    filenames.append(str(p_obj.resolve()))
            else:
                filenames.append(str(p_obj.resolve()))
        # else: path does not exist or is not a file/dir, ignore.
    return filenames


def ensure_dir(dirpath: str | pathlib.Path, raise_exception: bool = True) -> bool:
    """
    Ensures a directory exists. Creates it if it doesn't.
    Returns True if directory exists (or was created successfully).
    If raise_exception is True, raises an OSError on failure. Otherwise, returns False.
    """
    path_obj = pathlib.Path(dirpath)
    if path_obj.exists():
        if path_obj.is_dir():
            return True
        else: # Exists but is a file
            if raise_exception:
                raise FileExistsError(f"Path '{dirpath}' exists but is not a directory.")
            return False

    try:
        path_obj.mkdir(parents=True, exist_ok=True) # exist_ok=True handles race conditions
        return True
    except OSError as e:
        if raise_exception:
            raise e
        return False

# -------------------------------------------------
# Engine Version (Placeholders - can be set in config.py or here)
ENGINE_MAJOR_VERSION = 0
ENGINE_MINOR_VERSION = 1
ENGINE_REVISION_VERSION = 0

def get_version_numbers() -> Tuple[int, int, int]:
    """Returns the engine major/minor/revision version numbers."""
    return (ENGINE_MAJOR_VERSION, ENGINE_MINOR_VERSION, ENGINE_REVISION_VERSION)

def get_version_info() -> str:
    """Returns the engine version information string."""
    major, minor, rev = get_version_numbers()
    # Assuming ENGINE_NAME is defined, perhaps in config.py
    try:
        from .. import config
        name = config.ENGINE_NAME
    except ImportError:
        name = "PyGomokuEngine"
    return f"{name} version {major}.{minor}.{rev}"

def get_engine_info() -> str:
    """Returns general engine information."""
    try:
        from .. import config
        name = config.ENGINE_NAME
        author = config.ENGINE_AUTHOR
    except ImportError:
        name = "PyGomokuEngine"
        author = "AI Agent & User"
    return f"{name} by {author}"


if __name__ == '__main__':
    print("--- Time utils ---")
    start_time_ms = now()
    time.sleep(0.05) # Sleep for 50 ms
    end_time_ms = now()
    print(f"Time elapsed: {end_time_ms - start_time_ms}ms")
    print(f"time_text(50): {time_text(50)}")
    print(f"time_text(15000): {time_text(15000)}")
    print(f"time_text(120000): {time_text(120000)}")
    print(f"time_text(7200000): {time_text(7200000)}")

    print("\n--- Math utils ---")
    print(f"power(2, 10): {power(2, 10)}")
    print(f"is_power_of_two(16): {is_power_of_two(16)}")
    print(f"is_power_of_two(15): {is_power_of_two(15)}")
    print(f"floor_log2(1023): {floor_log2(1023)}")
    print(f"floor_power_of_two(1000): {floor_power_of_two(1000)}")
    print(f"combine_number(5, 3): {combine_number(5, 3)} (H(5,3) = C(5+3-1, 3) = C(7,3)=35)") # C(n+k-1, k)

    print("\n--- String utils ---")
    test_str = "  Hello, World!  "
    print(f"trim_inplace('{test_str}'): '{trim_inplace(test_str)}'")
    print(f"upper_inplace('hello'): '{upper_inplace('hello')}'")
    print(f"replace_all('one two one three', 'one', 'ONE'): '{replace_all('one two one three', 'one', 'ONE')}'")
    print(f"split_str_view('a,b; c|d', ',;|'): {split_str_view('a,b; c|d', ',;|')}")
    print(f"split_str_view('a,,b', ',', include_empty=True): {split_str_view('a,,b', ',', include_empty=True)}")
    print(f"nodes_text(1234567): {nodes_text(1234567)}")
    print(f"speed_text(1234567): {speed_text(1234567)}")

    print("\n--- PRNG ---")
    prng = PRNG(12345)
    print(f"PRNG output 1: {prng()}")
    print(f"PRNG output 2: {prng()}")
    print(f"PRNG output 3: {prng()}")

    print("\n--- Filesystem utils ---")
    test_dir = pathlib.Path("./test_utils_dir")
    ensure_dir(test_dir)
    print(f"Ensured directory: {test_dir.resolve()}")
    # Create some test files
    with open(test_dir / "file1.txt", "w") as f: f.write("test")
    with open(test_dir / "file2.log", "w") as f: f.write("log")
    sub_dir = test_dir / "subdir"
    ensure_dir(sub_dir)
    with open(sub_dir / "file3.txt", "w") as f: f.write("sub test")

    print(f"Files in '{test_dir}' (txt): {list_all_files_in_dir_recursively(test_dir, ['.txt'])}")
    print(f"All files in '{test_dir}': {list_all_files_in_dir_recursively(test_dir)}")

    path_list = [str(test_dir / "file1.txt"), str(sub_dir)]
    print(f"Files from list '{path_list}' (txt): {make_file_list_from_path_list(path_list, ['.txt'])}")

    # Cleanup test files/dirs
    os.remove(sub_dir / "file3.txt")
    os.rmdir(sub_dir)
    os.remove(test_dir / "file1.txt")
    os.remove(test_dir / "file2.log")
    os.rmdir(test_dir)
    print("Filesystem utils tests completed and cleaned up.")

    print("\n--- Version ---")
    print(get_version_info())
    print(get_engine_info())