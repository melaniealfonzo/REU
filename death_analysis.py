#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 12:04:26 2020

@author: MelanieAlfonzo
"""

import numpy as np
import pandas as pd
import os


#this file has 14000 cases
df = pd.read_csv('/Users/MelanieAlfonzo/Desktop/covid.csv') 
df = df.iloc[:,0:33]

l = df['ID']
#print(len(l))
x = len(l)
col_name = []
for col in df:
    #print(col)
    col_name.append(col)
#print(col_name) 


#print(df['outcome'].unique())

count = 0
for i in df['outcome']:
    if i == 'died':
        count += 1
    elif i == 'death':
        count += 1
    else:
        count += 0
    
#print(count)
#print('there are', count, 'deaths in file covid.csv with', x, 'cases')  


 #this file has 2000000 cases

dl = pd.read_csv('/Users/MelanieAlfonzo/Desktop/latestdata.csv') 
dl = dl.iloc[:,0:33]

d_l = dl['ID']
#print(len(d_l))
y = len(d_l)
lcol_name = []
for col in dl:
    #print(col)
    lcol_name.append(col)
print(lcol_name) 
print('there are', len(lcol_name), 'coolumns in latestdata.csv')


#print(dl['outcome'].unique())

list_dead = ['died','death','dead','Death','Died','Dead','Deceased']
lcount = 0
ncount = 0
ocount = 0
for i in dl['outcome']:
    if i == 'died':
        lcount += 1
    elif i == 'death':
        lcount += 1
    elif i == 'dead':
        lcount += 1
    elif i == 'Death':
        lcount += 1
    elif i == 'Died':
        lcount += 1
    elif i == 'Dead':
        lcount += 1
    elif i == 'Deceased':
        lcount += 1
    elif pd.isnull(i) == True:
        ncount += 1
    else:
        ocount += 1
    
print('there are', count, 'deaths in file covid.csv with', x, 'cases') 
#print(lcount)
print('there are', lcount, 'deaths in file latestdata.csv with', y, 'cases')
print('there are ',ncount, 'empty rows in outcome column in latestdata.csv')
print(ocount,' people survived in various conditions in latestdata.csv file')


w = y - ncount
print('there are ',w, ' cases with data in latestdata.csv excluding empty rows')
q = lcount / w * 100
print('the death rate in this file (latestdata.csv) is',q, 'percent')


scount = 0
for s in dl['symptoms']:
    if pd.isnull(s) == True:
        scount += 0
    else:
        scount +=1
print('in latestdata.csv, ',scount, 'rows have symptoms')    


ds_count = 0
for s in dl['date_onset_symptoms']:
   if pd.isnull(s) == True:
        ds_count += 0
   else:
        ds_count +=1
print('in latestdata.csv, ',ds_count, 'rows have symptom onset datws') 


cdb_count = 0
for s in dl['chronic_disease_binary']:
   if pd.isnull(s) == True:
        cdb_count += 0
   elif s == 0:
        cdb_count += 0
   else:
        cdb_count +=1
print('in latestdata.csv, ',cdb_count, 'rows have chronic diseases according to binary')  


cd_count = 0
for s in dl['chronic_disease']:
   if pd.isnull(s) == True:
        cd_count += 0
   else:
        cd_count +=1
print('in latestdata.csv, ',cd_count, 'rows have chronic diseases') 

count_c = 0
for c in dl['chronic_disease_binary']:
    if c == 1:
        count_c += 1
print(count_c)
    
