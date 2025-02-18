
# List Operations in Python

# Squaring Numbers
numbers = [1, 2, 3, 4, 5]
squared_numbers = [x ** 2 for x in numbers]
print("Squared Numbers:", squared_numbers)

# Filtering Even Numbers
even_numbers = [x for x in numbers if x % 2 == 0]
print("Even Numbers:", even_numbers)

# Modifying Numbers
modified_numbers = [x if x % 2 == 0 else x * 2 for x in numbers]
print("Modified Numbers:", modified_numbers)

# Flattening a 2D List
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [num for row in matrix for num in row]
print("Flattened List:", flattened)

# Converting to Uppercase
names = ["alice", "bob", "charlie"]
uppercase_names = [name.upper() for name in names]
print("Uppercase Names:", uppercase_names)

# Creating a Dictionary
squares_dict = {x: x ** 2 for x in range(1, 6)}
print("Squares Dictionary:", squares_dict)

# Removing Duplicates and Squaring
numbers_with_duplicates = [1, 2, 2, 3, 4, 4, 5]
squared_set = {x ** 2 for x in numbers_with_duplicates}
print("Squared Set:", squared_set)

# Creating a Generator
gen = (x ** 2 for x in range(1, 6))
print("Generated Squared Values:", list(gen))
