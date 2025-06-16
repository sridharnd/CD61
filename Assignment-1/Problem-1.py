import numpy as np

# Step 1: Generate the 5x4 array with random integers between 1 and 50
np.random.seed(0)  # for reproducibility
array = np.random.randint(1, 51, size=(5, 4))
print("Original Array:\n", array)

# Step 2: Extract the anti-diagonal elements (top-right to bottom-left)
anti_diag = [array[i, -1 - i] for i in range(min(array.shape))]
print("\nAnti-diagonal elements:", anti_diag)

# Step 3: Compute and print the maximum value in each row
max_in_rows = np.max(array, axis=1)
print("\nMaximum value in each row:", max_in_rows)

# Step 4: Elements less than or equal to the mean
mean_val = np.mean(array)
filtered_elements = array[array <= mean_val]
print(f"\nMean of array: {mean_val:.2f}")
print("Elements <= mean:", filtered_elements)

# Step 5: Boundary traversal function
def numpy_boundary_traversal(matrix):
    rows, cols = matrix.shape
    boundary = []

    # Top row
    for col in range(cols):
        boundary.append(matrix[0, col])

    # Right column (excluding top)
    for row in range(1, rows):
        boundary.append(matrix[row, cols - 1])

    # Bottom row (excluding last element)
    if rows > 1:
        for col in range(cols - 2, -1, -1):
            boundary.append(matrix[rows - 1, col])

    # Left column (excluding top and bottom)
    if cols > 1:
        for row in range(rows - 2, 0, -1):
            boundary.append(matrix[row, 0])

    return boundary

# Apply boundary traversal
boundary = numpy_boundary_traversal(array)
print("\nBoundary traversal (clockwise from top-left):", boundary)
