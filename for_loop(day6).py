# Prints numbers from 1 to 5
for i in range(1, 6):
    print(i)
print()

# Loop with else statement
for i in range(3):
    print(i)
else:
    print("end")
print()

# Print table from 1 to 3
for i in range(1, 4):
    for j in range(1, 4):
        print(f"{i} x {j} = {i*j}")
print()

# skip number 3
for i in range(1, 6):
    if i == 3:
        continue
    print(i)
print()


# Stops loop at 3
for i in range(1, 6):
    if i == 3:
        break
    print(i)
