# Define two integer variables
g_int = 2
h_int = 3

# Print the values and types of the variables
print("g_int and h_int are two variables of int type with values:", g_int, "and", h_int)
print("Type of g_int:", type(g_int), "Type of h_int:", type(h_int))

# Perform bitwise operations
print("Bitwise AND operation between g_int and h_int = ", g_int, "&", h_int, "=", g_int & h_int)
print("Bitwise OR operation between g_int and h_int = ", g_int, "|", h_int, "=", g_int | h_int)
print("Bitwise XOR operation between g_int and h_int = ", g_int, "^", h_int, "=", g_int ^ h_int)
print("Bitwise NOT operation on g_int =", "~", g_int, "=", ~g_int)
print("Bitwise left shift operation on g_int by 1 =", g_int, "<< 1 =", g_int << 1)
print("Bitwise right shift operation on g_int by 1 =", g_int, ">> 1 =", g_int >> 1)

# Define two integer variables for comparison
i_int = 2
j_int = 3

# Print the values of the variables
print("i_int and j_int are two variables of int type with values:", i_int, "and", j_int)

# Perform comparison operations
print("Greater than operation between i_int and j_int =", i_int, ">", j_int, "=", i_int > j_int)
print("Less than operation between i_int and j_int =", i_int, "<", j_int, "=", i_int < j_int)
print("Greater than or equal to operation between i_int and j_int =", i_int, ">=", j_int, "=", i_int >= j_int)
print("Less than or equal to operation between i_int and j_int =", i_int, "<=", j_int, "=", i_int <= j_int)
print("Equal to operation between i_int and j_int =", i_int, "==", j_int, "=", i_int == j_int)
print("Not equal to operation between i_int and j_int =", i_int, "!=", j_int, "=", i_int != j_int)

# Define two integer variables for identity operations
k_int = 3
l_int = 4

# Print the values of the variables
print("k_int and l_int are two variables of int type with values:", k_int, "and", l_int)

# Perform identity operations
print("Identity operation (is) between k_int and l_int =", k_int, "is", l_int, "=", k_int is l_int)
print("Identity operation (is not) between k_int and l_int =", k_int, "is not", l_int, "=", k_int is not l_int)

# Define a string variable
m_str = "hello"

# Print the value and type of the variable
print("m_str is a variable of string type with value:", m_str)
print("Type of m_str:", type(m_str))

# Perform membership operations
print("Membership operation (in) to check if 'h' is in m_str =", "'h' in", m_str, "=", 'h' in m_str)
print("Membership operation (not in) to check if 'h' is not in m_str =", "'h' not in", m_str, "=", 'h' not in m_str)

# Define two integer variables for assignment operations
n_int = 2
o_int = 3

# Print the initial value of n_int
print("n_int and o_int are two variables of int type with values:", n_int, "and", o_int)
print("Assignment operation: n_int =", n_int)

# Perform assignment operations
n_int += o_int
print("Addition assignment operation: n_int += o_int =", n_int)
n_int -= o_int
print("Subtraction assignment operation: n_int -= o_int =", n_int)
n_int *= o_int
print("Multiplication assignment operation: n_int *= o_int =", n_int)
n_int /= o_int
print("Division assignment operation: n_int /= o_int =", n_int)
n_int %= o_int
print("Modulus assignment operation: n_int %= o_int =", n_int)
n_int //= o_int
print("Floor division assignment operation: n_int //= o_int =", n_int)
n_int **= o_int
print("Exponentiation assignment operation: n_int **= o_int =", n_int)