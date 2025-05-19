# parallel_test.py

import time
from concurrent.futures import ProcessPoolExecutor

def cpu_bound_task(n):
    """A CPU-bound task that performs a large number of computations."""
    total = 0
    for i in range(1, 1000000):
        total += i * i
    return total + n

def serial_execution(tasks):
    """Execute tasks serially."""
    results = []
    start_time = time.time()
    for task in tasks:
        result = cpu_bound_task(task)
        results.append(result)
    end_time = time.time()
    serial_time = end_time - start_time
    return results, serial_time

def parallel_execution(tasks, max_workers=4):
    """Execute tasks in parallel using ProcessPoolExecutor."""
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Use executor.map to maintain the order of results
        results = list(executor.map(cpu_bound_task, tasks))
    end_time = time.time()
    parallel_time = end_time - start_time
    return results, parallel_time

if __name__ == "__main__":
    # Define a list of tasks (inputs)
    tasks = [i for i in range(1000)]  # Adjust the range for more or fewer tasks

    # Serial Execution
    serial_results, serial_duration = serial_execution(tasks)
    print(f"Serial Execution Time: {serial_duration:.2f} seconds")

    # Parallel Execution
    parallel_results, parallel_duration = parallel_execution(tasks, max_workers=4)
    print(f"Parallel Execution Time: {parallel_duration:.2f} seconds")

    # Verify that both methods produce the same results
    assert serial_results == parallel_results, "Mismatch between serial and parallel results!"

    # Calculate and display the speedup
    speedup = serial_duration / parallel_duration
    print(f"Speedup: {speedup:.2f}x faster using parallel execution")
