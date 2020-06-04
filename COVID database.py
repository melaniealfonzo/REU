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

