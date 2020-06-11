import numpy as np
import pandas as pd
import os

dm = pd.read_csv('/Users/MelanieAlfonzo/Desktop/medical.csv') 
dm = dm.iloc[:,0:33]


col_name = []
for i in dm:
    col_name.append(i)
    
print(col_name)

dm['Symptoms'] = dm['Symptoms'].replace('Missing["NotAvailable"]', 0 )

count=0
for i in dm['Symptoms']:
    if i != 0 :
        count+=1
print(count)

sym = dm['Symptoms']


tot_count = 0
for i in dm['Symptoms']:
    tot_count += 1
print(tot_count)

print(count/tot_count*100)

icount = 0
for i in dm['Age']:
    icount +=1
print(icount)    

s=[]
for i in dm['Symptoms']:
    if i != 0:
        s.append(i)
   
c_count = 0
for i in dm['ChronicDiseaseQ']: 
    if i == True:
        c_count += 1
print(c_count)

ci = []
ci_count = 0
for i in dm['ChronicDiseases']: 
    if i != 'Missing["NotAvailable"]':
        ci_count += 1
        ci.append(i)
print(ci_count)
print(ci)
