# Importing NumPy
import numpy as np

# Creating and Printing Arrays
arr = np.array([1, 2, 3, 4, 5])
print("Array:", arr)

arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print("2D Array:\n", arr2d)

# Array Properties
print("Shape:", arr2d.shape)
print("Size:", arr2d.size)

# Creating Arrays with Specific Values
zeros_arr = np.zeros((3, 3))
ones_arr = np.ones((2, 4))
full_arr = np.full((2, 2), 7)

print("Zeros Array:\n", zeros_arr)
print("Ones Array:\n", ones_arr)
print("Full Array:\n", full_arr)

# Creating Arrays with Ranges and Linspace
range_arr = np.arange(0, 10, 2)
linspace_arr = np.linspace(0, 1, 5)

print("Range Array:", range_arr)
print("Linspace Array:", linspace_arr)

# Identity Matrix and Random Numbers
identity_matrix = np.eye(3)
random_arr = np.random.rand(3, 3)
randint_arr = np.random.randint(1, 10, (2, 2))

print("Identity Matrix:\n", identity_matrix)
print("Random Array:\n", random_arr)
print("Random Integer Array:\n", randint_arr)

# Accessing and Slicing Arrays
print("Element at index 2:", arr[2])
print("Slicing array:", arr[1:4])

print("Element at (1,2):", arr2d[1, 2])

# Boolean Masking
bool_mask = arr > 2
filtered_arr = arr[arr > 2]

print("Boolean Mask:", bool_mask)
print("Filtered Array:", filtered_arr)

# Basic Arithmetic Operations
arr_sum = arr + 5
arr_prod = arr * 2
arr_square = arr ** 2

print("Sum Array:", arr_sum)
print("Product Array:", arr_prod)
print("Square Array:", arr_square)

# Matrix Operations
dot_product = np.dot(arr2d, arr2d.T)

print("Dot Product:\n", dot_product)

# Aggregation Functions
print("Sum:", arr.sum())
print("Mean:", arr.mean())
print("Max:", arr.max())
print("Min:", arr.min())

# Reshaping and Stacking Arrays
reshaped_arr = np.reshape(arr2d, (3, 2))
stacked_arr = np.vstack((arr, arr))
hstacked_arr = np.hstack((arr, arr))

print("Reshaped Array:\n", reshaped_arr)
print("Vertically Stacked Array:\n", stacked_arr)
print("Horizontally Stacked Array:\n", hstacked_arr)

# Splitting and Transposing Arrays
split_arr = np.array_split(arr, 2)
transposed_arr = arr2d.T

print("Split Arrays:", split_arr)
print("Transposed Array:\n", transposed_arr)

# Finding Unique Elements and Sorting
unique_elements = np.unique(arr2d)
sorted_arr = np.sort(arr)

print("Unique Elements:", unique_elements)
print("Sorted Array:", sorted_arr)

# Concatenating and Broadcasting
concat_arr = np.concatenate((arr, arr))
broadcast_arr = arr + np.array([10])

print("Concatenated Array:", concat_arr)
print("Broadcasted Array:", broadcast_arr)

# Finding Index of Max and Min Values
max_index = np.argmax(arr)
min_index = np.argmin(arr)

print("Index of Max Element:", max_index)
print("Index of Min Element:", min_index)

# Copying and Modifying Arrays
copied_arr = arr.copy()
modified_arr = np.where(arr > 3, 100, arr)

print("Copied Array:", copied_arr)
print("Modified Array:", modified_arr)

# Flattening and Saving/Loading Arrays
flattened_arr = arr2d.flatten()
np.save('saved_array.npy', arr)
loaded_arr = np.load('saved_array.npy')

print("Flattened Array:", flattened_arr)
print("Loaded Array:", loaded_arr)