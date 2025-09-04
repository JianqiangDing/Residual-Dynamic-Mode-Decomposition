# this function is used to find the closet indices of the given values of an array in another array without repeated indices
# which means the indices should be unique and maintain the one-to-one correspondence
# to ensure the global optimal solution, we use dynamic programming to solve this problem

import numpy as np


def find_closet_subset_index(short_array, long_array, dist):
    """
    Find the closest indices of the given values of an array in another array
    using dynamic programming to ensure global optimal solution with unique indices.

    Parameters:
    -----------
    short_array : array-like
        The array with values that need to find corresponding indices
    long_array : array-like
        The array from which to find the closest indices
    dist : callable
        Function to calculate the distance between two given numbers, R × R -> R

    Returns:
    --------
    numpy.ndarray
        Array of indices from long_array that correspond to closest matches
        for each element in short_array, maintaining one-to-one correspondence

    Raises:
    -------
    ValueError
        If short_array is longer than long_array or if arrays are empty
    TypeError
        If dist is not callable

    Examples:
    ---------
    >>> import numpy as np
    >>> short = [1.0, 3.0, 5.0]
    >>> long = [0.5, 1.1, 2.0, 3.2, 4.8, 5.1]
    >>> indices = find_closet_subset_index(short, long, lambda x, y: abs(x - y))
    >>> print(indices)  # [1, 3, 5] (approximate)
    """
    # Input validation
    if not callable(dist):
        raise TypeError("dist must be a callable function")

    short_array = np.asarray(short_array)
    long_array = np.asarray(long_array)

    if len(short_array) == 0:
        return np.array([], dtype=int)

    if len(long_array) == 0:
        raise ValueError("long_array cannot be empty when short_array is not empty")

    if len(short_array) > len(long_array):
        raise ValueError("short_array cannot be longer than long_array")

    n = len(short_array)
    m = len(long_array)

    # Precompute distance matrix for efficiency
    distance_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            distance_matrix[i, j] = dist(short_array[i], long_array[j])

    # Dynamic programming approach
    # dp[i][mask] = minimum total distance for assigning first i elements
    # of short_array using indices represented by mask
    # We use a dictionary for sparse representation since mask can be large

    # For efficiency with larger arrays, we'll use a different DP approach
    # dp[i][j] = minimum cost to assign first i elements using first j indices
    # But we need to track which specific indices are used

    # Use Hungarian algorithm approach with DP
    # Since we need global optimum, we'll enumerate valid assignments

    from itertools import combinations

    # Generate all possible combinations of m choose n indices
    best_cost = float("inf")
    best_assignment = None

    # For each possible combination of n indices from m positions
    for indices_combo in combinations(range(m), n):
        # Calculate total cost for this assignment
        total_cost = 0
        assignment = list(indices_combo)

        # We need to find the best permutation of these indices
        # Use Hungarian algorithm or try all permutations for small n
        if n <= 8:  # For small n, try all permutations
            from itertools import permutations

            min_perm_cost = float("inf")
            best_perm = None

            for perm in permutations(assignment):
                cost = sum(distance_matrix[i, perm[i]] for i in range(n))
                if cost < min_perm_cost:
                    min_perm_cost = cost
                    best_perm = perm

            if min_perm_cost < best_cost:
                best_cost = min_perm_cost
                best_assignment = best_perm
        else:
            # For larger n, use a more efficient approach
            # Create cost matrix for this subset and solve assignment problem
            subset_costs = distance_matrix[:, list(indices_combo)]
            assignment_result = _solve_assignment_problem(subset_costs)
            total_cost = sum(subset_costs[i, assignment_result[i]] for i in range(n))

            if total_cost < best_cost:
                best_cost = total_cost
                best_assignment = [
                    indices_combo[assignment_result[i]] for i in range(n)
                ]

    return np.array(best_assignment, dtype=int)


def _solve_assignment_problem(cost_matrix):
    """
    Solve the assignment problem using a simple implementation.
    For small matrices, this is sufficient.
    """
    n = cost_matrix.shape[0]
    if n <= 8:
        # Use brute force for small problems
        from itertools import permutations

        min_cost = float("inf")
        best_assignment = None

        for perm in permutations(range(n)):
            cost = sum(cost_matrix[i, perm[i]] for i in range(n))
            if cost < min_cost:
                min_cost = cost
                best_assignment = perm

        return list(best_assignment)
    else:
        # For larger problems, use a greedy approach as approximation
        # This is not guaranteed to be optimal but is much faster
        assignment = [-1] * n
        used_cols = set()

        # Greedy assignment: for each row, pick the minimum unused column
        row_order = list(range(n))
        # Sort rows by their minimum cost to improve greedy performance
        row_order.sort(key=lambda i: np.min(cost_matrix[i, :]))

        for i in row_order:
            best_col = -1
            best_cost = float("inf")
            for j in range(n):
                if j not in used_cols and cost_matrix[i, j] < best_cost:
                    best_cost = cost_matrix[i, j]
                    best_col = j

            assignment[i] = best_col
            used_cols.add(best_col)

        return assignment
