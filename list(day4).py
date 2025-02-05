# Create a list and perform various operations
my_list = [10, 20, 30, 40, 50]  
my_list.append(60)  
my_list.extend([70, 80])  
my_list.insert(2, 25)  
my_list.remove(30)  
last_item = my_list.pop()  
second_item = my_list.pop(1)  
index_of_40 = my_list.index(40)  
count_20 = my_list.count(20)  
my_list.sort()  
my_list.reverse()  
copied_list = my_list.copy()  
my_list.clear()  

# Perform various operations on the copied list
squared_list = [x**2 for x in copied_list]  
combined_list = copied_list + [100, 110, 120]  
repeated_list = [1, 2, 3] * 3  
is_50_present = 50 in copied_list  
length = len(copied_list)  
min_value = min(copied_list)  
max_value = max(copied_list)  
sum_of_elements = sum(copied_list)  
string_list = ", ".join(map(str, copied_list))  
new_list_from_string = string_list.split(", ")  

# Print the results
print("Copied List:", copied_list)
print("Squared List:", squared_list)
print("Combined List:", combined_list)
print("Repeated List:", repeated_list)
print("Is 50 present:", is_50_present)
print("Length of Copied List:", length)
print("Min Value:", min_value)
print("Max Value:", max_value)
print("Sum of Elements:", sum_of_elements)
print("List as String:", string_list)
print("New List from String:", new_list_from_string)