
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  


ns= pd.Series([2,4,6,8,10],index =["A","B","c","D","E"],name = "EVEN")
print(ns)
print(ns[0])
print(ns["A"])

#creating DataFrame

arr2D = np.random.randint(1,10,(3,4))
firstDF = pd.DataFrame(arr2D,columns =["AJAY","AALU","PANDEY","MOSHI"],index=["MATHS","PHYSICS","CHEMISTRY"])
print(firstDF)

#list of lists

LOL = [["amar",10],["AKBAR",15],["anthony",14]]
df = pd.DataFrame(LOL,columns=["name","age"])
print(df)

#data in dictionary

emp = {"employe name":["raju","sabu","mangu"],"income":[25000,30000,45000]}
emp = pd.DataFrame(emp)
print(emp)


#dictionary from series

serdict ={"FIRST SER": ns,"SECOND SER": pd.Series([10,20,30,40,50])}
print(serdict)


# List of names 
actor_names = ['Ryan Reynolds', 'Benedict Cumberbatch', 'Robert Downey Jr.', 'Chris Hemsworth']
# List of their ages 
actor_ages = [48,62,54,64, np.nan ] 
# Get the list of tuples by zipping the two lists together 
list_of_tuples=list(zip(actor_names, actor_ages)) 
# Converting the list of tuples into pandas Dataframe 
actor_df = pd.DataFrame(list_of_tuples, columns = ['Name', 'Age']) 
# Print the dataframe 
print(actor_df)




height =100
width =100 
random_image =np.random.randint(0,256,(height ,width,3),dtype=np.uint8) 


plt.imshow(random_image)
plt.axis('off')
plt.show() 