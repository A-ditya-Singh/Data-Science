# Time and Space Complexity Examples

import time

# Constant Time Complexity: O(1)
def get_first_element(arr):
    return arr[0] if arr else None

# Linear Time Complexity: O(n)
def print_all_elements(arr):
    for item in arr:
        print(item)

# Quadratic Time Complexity: O(n^2)
def print_all_pairs(arr):
    for i in arr:
        for j in arr:
            print(i, j)

# Logarithmic Time Complexity: O(log n)
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

# Factorial Time Complexity: O(n!)
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

# Exponential Time Complexity: O(2^n)
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# Cubic Time Complexity: O(n^3)
def print_all_triplets(arr):
    for i in arr:
        for j in arr:
            for k in arr:
                print(i, j, k)

# Space Complexity Examples

# Constant Space Complexity: O(1)
def get_constant_value():
    return 10

# Linear Space Complexity: O(n)
def create_array(n):
    return [i for i in range(n)]

# Quadratic Space Complexity: O(n^2)
def create_matrix(n):
    return [[0] * n for _ in range(n)]

# Helper function to measure execution time
def measure_execution_time(func, *args):
    start = time.time()
    result = func(*args)
    end = time.time()
    print(f"Execution time for {func.__name__}: {end - start:.6f} seconds")
    return result

# Example usage
arr = [1, 2, 3, 4, 5]
measure_execution_time(get_first_element, arr)
measure_execution_time(print_all_elements, arr)
measure_execution_time(print_all_pairs, arr)
measure_execution_time(binary_search, sorted(arr), 3)
measure_execution_time(factorial, 5)
measure_execution_time(fibonacci, 5)
measure_execution_time(print_all_triplets, arr)

measure_execution_time(get_constant_value)
measure_execution_time(create_array, 5)
measure_execution_time(create_matrix, 5)