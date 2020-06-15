#!/usr/bin/env python

# getting ready
import numpy as np
import pandas as pd
import os


df = pd.read_csv('covid.csv') 
df = df.iloc[:,0:33]
df['symptoms'] = df['symptoms'].str.replace(';',',')
x = df['symptoms'].str.split(',',expand=True).stack().str.strip().unique()

print(x)
print(len(x))

#make new columns for each symptom category
new_symptoms = ['fever','respiratory symptoms','headache/nausea','cough','weakness','vomitting/diarrhea']


print(df.columns.get_loc("symptoms"))

#see new symptoms
for i in new_symptoms:
    df[i] = 0
    
    
print(df.columns.get_loc("fever"))

count = 0
for i in new_symptoms:
    #print(count)
    #print(i)
    #print(df.iloc[count,13])
    #if df.iloc[count,13] == i:
        #df.iloc[count,i] = 1
    count += 1
        

#did not work
#for i in new_symptoms:
    #print(count)
    #print(df.iloc[count,'symptoms'])
    #count += 1        

#did not work
#count = 0
#for i in df.symptoms:
 #   for j in new_symptoms:
  #      if i == j:
   #         df.iloc[count,j] = 1 
    #        print(i)
     #       print(j)
      #      count +=1 
            #if j == "fever":
                #df.iloc[count,33] = 1




#put 1's and 0's for new symptom columns 
#i tried to have it loop through each symptom category and each sympton originally in the dataframe to put 1 or 0 in the corresponding cell
count = 0
for j in new_symptoms:
    print(j)
    for i in df.symptoms:
        if i == j:
            #print(i)
            #print(j)
            #print(df.columns.get_loc(j))
            if df.columns.get_loc(j) == 33:
                df.iloc[count,33] = 1
                x = (df.columns.get_loc(j))
                print(x)
        
            elif df.columns.get_loc(j) == 34:
                df.iloc[count,34] = 1
                x = (df.columns.get_loc(j))
                print(x)
                
            elif df.columns.get_loc(j) == 36:
                df.iloc[count,36] = 1
                x = (df.columns.get_loc(j))
                print(x)
                
            elif df.columns.get_loc(j) == 37:
                df.iloc[count,37] = 1
                x = (df.columns.get_loc(j))
                print(x)
                
            else:
                print('na')
     

