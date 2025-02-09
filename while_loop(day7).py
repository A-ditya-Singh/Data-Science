# Prints numbers from 1 to 5
x = 1
while x <= 5:
    print(x)
    x += 1
print()

#while else loop
x = 3
while x > 0:
    print(x)
    x -= 1
else:
    print("Loop ended")
print()

#infinite loop
x = 1
while True:
    print(x)
    if x == 3:
        break  # Stops the infinite loop
    x += 1
