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
    df['date_confirmation']= pd.to_datetime(df['date_confirmation']) 
    select_day = datetime.datetime(2020, 5, 29) 
    train_df = df.loc[(df.date_confirmation <= select_day)]
    test_df = df.loc[(df.date_confirmation > select_day)]
    
    x_train_df =  train_df[['AgeRange_code','latitude', 'longitude', 'chronic_disease_binary','travel_history_binary','combine_symptoms', 'gender_binary']]
    x_test_df =  test_df[['AgeRange_code','latitude', 'longitude', 'chronic_disease_binary','travel_history_binary','combine_symptoms', 'gender_binary']]
    
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
    
def preprocessData(df):
    
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
    
patient_file = "patient records.csv"
df = pd.read_csv(patient_file)
patient_df = preprocessData(df)
patient_df = patient_df.drop(['age_is_number', 'AgeRange'], axis=1)
#correlation_analysis(patient_df)
x_train, x_test, y_train, y_test = perpareTrainData(patient_df)


model = 'RF'
if model == 'RF':
    
    #from sklearn.preprocessing import RobustScaler
    #transformer = RobustScaler().fit(x_train)
    #x_train = transformer.transform(x_train)  
    #x_test = transformer.transform(x_test)
    
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)
    clf.fit(x_train, y_train)
    
#metrics
pred_y_train = clf.predict(x_train)
tn, fp, fn, tp = confusion_matrix(y_train, pred_y_train,).ravel()


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

lr_a = metrics.accuracy_score(y_test, predictions)

tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
specificity = tn / (tn+fp)
print('specificity', specificity)

sensitivity  = tp / (fn + tp)
print('sensitivity:', sensitivity)

from sklearn.metrics import roc_auc_score
roc_lr=roc_auc_score(y_test, predictions)
print('AUC: ',roc_lr)

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

a_svm = metrics.accuracy_score(y_test, svm_pred)
tn, fp, fn, tp = confusion_matrix(y_test, svm_pred).ravel()
s_specificity = tn / (tn+fp)
print('specificity:', s_specificity)

s_sensitivity  = tp / (fn + tp)
print('sensitivity:', s_sensitivity)

from sklearn.metrics import roc_auc_score
roc_svm=roc_auc_score(y_test, svm_pred)
print('AUC: ',roc_svm)

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

rf_a = metrics.accuracy_score(y_test, rf_pred.round())
tn, fp, fn, tp = confusion_matrix(y_test, rf_pred.round()).ravel()
rf_specificity = tn / (tn+fp)
print('specificity:', rf_specificity)

rf_sensitivity  = tp / (fn + tp)
print('sensitivity:', rf_sensitivity)

from sklearn.metrics import roc_auc_score
roc_rf=roc_auc_score(y_test, rf_pred.round())
print('AUC: ',roc_rf)

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
#print("Precision:",metrics.precision_score(y_test, c, average='micro'))
#print("Recall:",metrics.recall_score(y_test, c, average='micro'))

#tn, fp, fn, tp = confusion_matrix(y_test, c).ravel()
#specificity = tn / (tn+fp)
#print('specificity:', specificity)

#sensitivity  = tp / (fn + tp)
#print('sensitivity:', sensitivity)

from sklearn.metrics import roc_auc_score
roc_one=roc_auc_score(y_test, c.round())
print('AUC: ',roc_one)


#make a bar chart
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt



#plt.bar(['LR accuracy',' LR specificity',' LR sensitivity','SVM accuracy',' SVM specificity',' SVM sensitivity','RF accuracy',' RF specificity',' RF sensitivity'],[lr_a, specificity, sensitivity,a_svm, s_specificity, s_sensitivity,rf_a, rf_specificity, rf_sensitivity],align='edge', width=0.3)

x_labels = ['LR accuracy',' LR specificity',' LR sensitivity','SVM accuracy',' SVM specificity',' SVM sensitivity','RF accuracy',' RF specificity',' RF sensitivity']
y_values = [lr_a, specificity, sensitivity,a_svm, s_specificity, s_sensitivity,rf_a, rf_specificity, rf_sensitivity]

plt.bar(x_labels,y_values, align='edge', width=0.3)
