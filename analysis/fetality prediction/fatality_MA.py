#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 12:12:39 2020

@author: MelanieAlfonzo
"""

import csv
import pandas as pd 
import datetime
import numpy as np
import seaborn
import matplotlib.pyplot as plt

#encode age range
AgeRange_Dict = {
      "0-9": 1,
      "9-19": 2,
      "20-29": 3,
      "30-39": 4,
      "40-49": 5,
      "50-59": 6,
      "60-69": 7,
      "70-79": 8,
      "80-89": 9,
      "90-99": 10
    }

def correlation_analysis(df):
    corr_df = df.corr(method='pearson')
    print("--------------- CORRELATIONS ---------------")
    print(corr_df.head(len(df)))
    
    print("--------------- CREATE A HEATMAP ---------------")
    # Create a mask to display only the lower triangle of the matrix
    mask = np.zeros_like(corr_df)
    mask[np.triu_indices_from(mask)] = True
    # Create the heatmap using seaborn library. 
    seaborn.heatmap(corr_df, cmap='RdYlGn_r', vmax=1.0, vmin=-1.0 , mask = mask, linewidths=2.5)
    # Show the plot 
    plt.yticks(rotation=0) 
    plt.xticks(rotation=90)
    plt.show()
    
def perpareTrainData(df):
    '''
    df['date_confirmation']= pd.to_datetime(df['date_confirmation']) 
    select_day = datetime.datetime(2020, 5, 29) 
    train_df = df.loc[(df.date_confirmation <= select_day)]
    test_df = df.loc[(df.date_confirmation > select_day)]
    '''
    
    df['date_confirmation']= pd.to_datetime(df['date_confirmation']) 
    select_day = datetime.datetime(2020, 5, 29) 
    non_test_df = df.loc[(df.date_confirmation <= select_day)]
    test_df = df.loc[(df.date_confirmation > select_day)]
    
    select_day2 = datetime.datetime(2020, 5, 25) 
    train_df = non_test_df.loc[(non_test_df.date_confirmation <= select_day2)]
    valid_df = non_test_df.loc[(non_test_df.date_confirmation > select_day2)]
    
    if data_process == 1:
        x_train_df =  train_df[['AgeRange_code','latitude', 'longitude', 'chronic_disease_binary','travel_history_binary','combine_symptoms', 'gender_binary']]
        x_test_df =  test_df[['AgeRange_code','latitude', 'longitude', 'chronic_disease_binary','travel_history_binary','combine_symptoms', 'gender_binary']]
    else:
        x_train_df =  train_df[['age','latitude', 'longitude', 'chronic_disease_binary','travel_history_binary','combine_symptoms', 'gender_binary']]
        x_test_df =  test_df[['age','latitude', 'longitude', 'chronic_disease_binary','travel_history_binary','combine_symptoms', 'gender_binary']]

    y_train_df =  train_df[['death']]
    y_test_df = test_df[['death']]
    
    death_rate_train = (y_train_df != 0).values.sum() / len(y_train_df)
    death_rate_test = (y_test_df != 0).values.sum() / len(y_test_df)
    
    x_train = x_train_df.values
    x_test = x_test_df.values
    y_train = y_train_df.values.flatten()
    y_test = y_test_df.values.flatten()
    
    return x_train, x_test, y_train, y_test
     
def is_int_number(data):
    try:
        int(data)
        return 1
    except ValueError:
        return 0  

def is_float_number(data):
    try:
        float(data)
        return 1
    except ValueError:
        return 0
    
def encode_age_range(data):
    if data in AgeRange_Dict:
        return AgeRange_Dict[data]
    else:
        return -1
    
def preprocessData_1(df):
    
    df['age_is_number'] = df.age.apply(is_int_number) 
    #keep age range and drop float ages 
    age_un_number_df = df.loc[df['age_is_number'] == 0]

    age_un_number_df['age_is_number'] = age_un_number_df.age.apply(is_float_number) 
    age_un_number_df = age_un_number_df.drop(age_un_number_df[age_un_number_df.age_is_number == 1].index)
    age_un_number_df['AgeRange'] = age_un_number_df['age']
    
    #convert age to age_range
    age_number_df = df.loc[df['age_is_number'] == 1]
    age_number_df = age_number_df.astype({"age": int})
    bins = [0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99]
    names = ['0-9', '9-19', '20-29', '30-39', '40-49', '50-59','60-69','70-79','80-89','90-99']
    age_number_df['AgeRange'] = pd.cut(age_number_df['age'], bins, labels=names)
    
    #mege two dataframe back
    frames = [age_number_df, age_un_number_df]
    patiant_df = pd.concat(frames)
    patiant_df['AgeRange_code'] =  patiant_df.AgeRange.apply(encode_age_range)
    patiant_df = patiant_df.drop(patiant_df[patiant_df.AgeRange_code == -1].index)
    
    return patiant_df

def preprocessData_2(df):
    
    df['age_is_number'] = df.age.apply(is_int_number) 
    #convert age to age_range
    age_number_df = df.loc[df['age_is_number'] == 1]
    return age_number_df
 
patient_file = "patient records.csv"
df = pd.read_csv(patient_file)

data_process = 2
if data_process == 1:
    patient_df = preprocessData(df)
    patient_df = patient_df.drop(['age_is_number', 'AgeRange'], axis=1)
else:
    patient_df = preprocessData_2(df)
    patient_df = patient_df.drop(['age_is_number'], axis=1)
    
#correlation_analysis(patient_df)
x_train, x_test, y_train, y_test = perpareTrainData(patient_df)


#logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
model = LogisticRegression(solver='liblinear', random_state=0) 
model.fit(x_train, y_train)
model = LogisticRegression(solver='liblinear', random_state=0).fit(x_train, y_train)
predictions = model.predict(x_test)
#get accuracy
lr_score = model.score(x_test, y_test)
print(lr_score)

#use confusion matrix to get all 3
from sklearn import metrics

cnf_matrix = metrics.confusion_matrix(y_test, predictions)
print('for logistic regression:')
print("Accuracy:",metrics.accuracy_score(y_test, predictions))
print("Precision:",metrics.precision_score(y_test, predictions))
print("Recall:",metrics.recall_score(y_test, predictions))


tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
specificity = tn / (tn+fp)
print('specificity', specificity)

sensitivity  = tp / (fn + tp)
print('sensitivity:', sensitivity)


#SVM 
from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)
svm_pred = clf.predict(x_test)

#use confusion matrix to get all 3
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, svm_pred)
cnf_matrix
print("for SVM")
print("Accuracy:",metrics.accuracy_score(y_test, svm_pred))
print("Precision:",metrics.precision_score(y_test, svm_pred))
print("Recall:",metrics.recall_score(y_test, svm_pred))

tn, fp, fn, tp = confusion_matrix(y_test, svm_pred).ravel()
specificity = tn / (tn+fp)
print('specificity:', specificity)

sensitivity  = tp / (fn + tp)
print('sensitivity:', sensitivity)


# random forest 
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(x_train, y_train);
rf_pred=rf.predict(x_test)
 

#use confusion matrix to get all 3
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, rf_pred.round())
print('for random forrest:')
print("Accuracy:",metrics.accuracy_score(y_test, rf_pred.round()))
print("Precision:",metrics.precision_score(y_test, rf_pred.round()))
print("Recall:",metrics.recall_score(y_test, rf_pred.round()))

tn, fp, fn, tp = confusion_matrix(y_test, rf_pred.round()).ravel()
specificity = tn / (tn+fp)
print('specificity:', specificity)

sensitivity  = tp / (fn + tp)
print('sensitivity:', sensitivity)


'''
#svm one class classification
from sklearn.svm import OneClassSVM
clf = OneClassSVM(gamma='auto').fit(x_train)
c = clf.predict(x_test)
s = clf.score_samples(x_test)
#clf.score_samples(x_train)


#use confusion matrix to get all 3
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, c.round())
print('for SVM one class')
print("Accuracy:",metrics.accuracy_score(y_test, c.round()))
print("Precision:",metrics.precision_score(y_test, c, average=None))
print("Recall:",metrics.recall_score(y_test, c, average=None))

tn, fp, fn, tp = confusion_matrix(y_test, c).ravel()
specificity = tn / (tn+fp)
print('specificity:', specificity)

sensitivity  = tp / (fn + tp)
print('sensitivity:', sensitivity)
'''


'''
for logistic regression:
Accuracy: 0.9890003626254079
Precision: 0.34375
Recall: 0.13580246913580246
specificity 0.9974365234375
sensitivity: 0.13580246913580246


for SVM
Accuracy: 0.9903299891212377
Precision: 0.5263157894736842
Recall: 0.12345679012345678
specificity: 0.9989013671875
sensitivity: 0.12345679012345678


for random forrest:
Accuracy: 0.986703735041702
Precision: 0.23636363636363636
Recall: 0.16049382716049382
specificity: 0.994873046875
sensitivity: 0.16049382716049382
'''