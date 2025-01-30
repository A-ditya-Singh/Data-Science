# int and operation(arithmetic)

# a and b are two variables of int type

a_int = 2
b_int = 3

print("a_int and b_int are two variables of int type with values:", a_int, "and", b_int)

print("Type of a_int:", type(a_int), "Type of b_int:", type(b_int))
# addition
print("Addition of a_int and b_int (int) = ", a_int, "+", b_int, "=", a_int + b_int)
# subtraction
print("Subtraction of a_int and b_int (int) = ", a_int, "-", b_int, "=", a_int - b_int)
# multiplication
print("Multiplication of a_int and b_int (int) = ", a_int, "*", b_int, "=", a_int * b_int)
# division
print("Division of a_int and b_int (int) = ", a_int, "/", b_int, "=", a_int / b_int)
# modulus
print("Modulus of a_int and b_int (int) = ", a_int, "%", b_int, "=", a_int % b_int)
# floor division
print("Floor division of a_int and b_int (int) = ", a_int, "//", b_int, "=", a_int // b_int)
# exponentiation
print("Exponentiation of a_int and b_int (int) = ", a_int, "", b_int, "=", a_int ** b_int)
print("\n\n")

# float operations(arithmetic)

# c and d are two variables of float type

# we can make c and d user defined by float(input())

c_float = float(input("Enter a float value for c_float: "))
d_float = float(input("Enter a float value for d_float: "))

print("c_float and d_float are two variables of float type with values:", c_float, "and", d_float)

print("Type of c_float:", type(c_float), "Type of d_float:", type(d_float))
# addition
print("Addition of c_float and d_float (float) = ", c_float, "+", d_float, "=", c_float + d_float)
# substraction
print("Subtraction of c_float and d_float (float) = ", c_float, "-", d_float, "=", c_float - d_float)
# multiplication
print("Multiplication of c_float and d_float (float) = ", c_float, "*", d_float, "=", c_float * d_float)
# division
print("Division of c_float and d_float (float) = ", c_float, "/", d_float, "=", c_float / d_float)
# modulus
print("Modulus of c_float and d_float (float) = ", c_float, "%", d_float, "=", c_float % d_float)
# floor division
print("Floor division of c_float and d_float (float) = ", c_float, "//", d_float, "=", c_float // d_float)
# exponentiation
print("Exponentiation of c_float and d_float (float) = ", c_float, "", d_float, "=", c_float ** d_float)
print("\n\n")

# order of arithmetic operations
print("Order of arithmetic operations for 10 + 2 * 3 ** 2 - (5 // 2)")
result = 10 + 2 * 3 ** 2 - (5 // 2)
print("Result of the expression 10 + 2 * 3 ** 2 - (5 // 2) =", result)

''' 1.Parentheses ()
    2.Exponentiation **
    3.Unary plus and minus +x, -x
    4.Multiplication, Division, Floor Division, and Modulo *, /, //, % (evaluated from left to right)
    5.Addition and Subtraction +, - (evaluated from left to right)'''
print("1.Parentheses ()\n2.Exponentiation **\n3.Unary plus and minus +x, -x\n4.Multiplication, Division, Floor Division, and Modulo *, /, //, % (evaluated from left to right)\n5.Addition and Subtraction +, - (evaluated from left to right)")
print("\n\n")
# boolean operations
# e and f are two variables of boolean type

e_bool = True
f_bool = False

print("e_bool and f_bool are two variables of boolean type with values:", e_bool, "and", f_bool)

print("Type of e_bool:", type(e_bool), "Type of f_bool:", type(f_bool))
# and operation
print("AND operation between e_bool and f_bool = ", e_bool, "and", f_bool, "=", e_bool and f_bool)
# or operation
print("OR operation between e_bool and f_bool = ", e_bool, "or", f_bool, "=", e_bool or f_bool)
# not operation
print("NOT operation on e_bool =", "not", e_bool, "=", not e_bool)
print("\n\n")