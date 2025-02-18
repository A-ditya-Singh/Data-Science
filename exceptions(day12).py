def divide_numbers(a, b):
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        print("Error: Invalid data type. Please enter numbers only.")
        return None
    if b == 0:
        print("Error: Division by zero is not allowed.")
        return None
    result = a / b
    print("Division successful! Result:", result)
    return result

def check_positive_number(num):
    if not isinstance(num, (int, float)):
        print("Error: Invalid data type. Please enter a number.")
        return
    if num < 0:
        print("Error: Negative numbers are not allowed.")
    else:
        print("Valid number:", num)

# Testing the functions with different cases
print(divide_numbers(10, 2))  
print(divide_numbers(10, 0))  
print(divide_numbers(10, 'a'))  

check_positive_number(-5)  
check_positive_number(5)