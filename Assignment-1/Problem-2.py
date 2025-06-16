import numpy as np

# Step 1: Create a 1D array of 20 random floats between 0 and 10
np.random.seed(1)  # For reproducibility
array = np.random.uniform(0, 10, 20)
print("Original array:\n", array)

# Step 2: Round all elements to 2 decimal places
rounded_array = np.round(array, 2)
print("\nRounded array:\n", rounded_array)

# Step 3: Calculate and print min, max, median
min_val = np.min(rounded_array)
max_val = np.max(rounded_array)
median_val = np.median(rounded_array)
print(f"\nMinimum: {min_val}")
print(f"Maximum: {max_val}")
print(f"Median: {median_val}")

# Step 4: Replace elements < 5 with their squares
modified_array = np.where(rounded_array < 5, np.round(rounded_array**2, 2), rounded_array)
print("\nModified array (elements < 5 replaced with square):\n", modified_array)

# Step 5: Alternate sorting function
def numpy_alternate_sort(arr):
    sorted_arr = np.sort(arr)
    result = []
    i, j = 0, len(sorted_arr) - 1
    while i <= j:
        result.append(sorted_arr[i])
        if i != j:
            result.append(sorted_arr[j])
        i += 1
        j -= 1
    return np.array(result)

# Apply alternate sort
alt_sorted_array = numpy_alternate_sort(rounded_array)
print("\nAlternating sorted array:\n", alt_sorted_array)
