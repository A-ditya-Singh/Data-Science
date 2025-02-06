# Tuples: Immutable sequence of values
student_info = ("Aarav", 21, "B.Sc", "XYZ University")
print("Student Info:", student_info)

# Accessing tuple elements
name, age, course, university = student_info
print(f"{name} is {age} years old and studying {course} at {university}.")

# Sets: Unordered collection of unique elements
tech_skills = {"Python", "C++", "Java", "Python"}  
tech_skills.add("Ruby")  
tech_skills.discard("C++")  

# Set operations
other_skills = {"HTML", "CSS", "Python"}
common_skills = tech_skills & other_skills  

print("Unique Tech Skills:", tech_skills)
print("Common Skills:", common_skills)

# Dictionaries: Key-value pairs
student_marks = {
    "Neha": 82,
    "Kabir": 95,
    "Riya": 78
}

# Access, update, and add dictionary values
print("Neha's Marks:", student_marks["Neha"])
student_marks["Riya"] = 88  
student_marks["Vikram"] = 91  

# Iterating through dictionary
for student, marks in student_marks.items():
    print(f"{student}: {marks}")

# Removing an entry
del student_marks["Kabir"]
print("Updated Marks:", student_marks)