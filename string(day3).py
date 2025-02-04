# Strings aur functions ka introduction

# String ka declaration
a = "Hello, World!"  
b = 'Python Programming'  
print(a)
print(b)

# Multiline string
c = '''Yeh ek
multiline string hai.'''
print(c)

# String indexing aur slicing
s = "Python"
print(s[0])  # Pehla char
print(s[-1])  # Last char
print(s[0:4])  # Pyth

# Strings immutable hote hain
# s[0] = 'J'  # Error dega

# String methods
string = "  Python Programming  "
print(string.lower())  # Lowercase
print(string.upper())  # Uppercase
print(string.strip())  # Spaces hata dega
print(string.replace("Python", "Java"))  # Replace karega
print(string.split())  # Split karega

# String ka length
print(len(string))

# String formatting
name = "Rahul"
age = 20
print(f"Mera naam {name} hai aur meri umar {age} saal hai.")  

# Functions ka introduction
def greet(name):  
    """Yeh function ek naam leta hai aur uska greeting return karta hai"""
    return "Namaste, " + name + "!"

print(greet("Amit"))  

def add(a, b=5):  
    return a + b

print(add(10))  
print(add(10, 20))  

def calculate(a, b):
    sum = a + b
    diff = a - b
    return sum, diff  

result = calculate(10, 5)
print(result)