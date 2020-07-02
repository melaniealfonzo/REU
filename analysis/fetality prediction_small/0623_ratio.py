import pandas as pd 
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
    
def perpareTrainData(df):
    
    df_d = df[df['death_binary'] == 1]
    df_a = df[df['death_binary'] == 0]
    
    len_d = len(df_d.index)
    len_a = len(df_a.index)
    
    test_pos_d =  int(len_d*0.7)
    test_pos_a =  int(len_a*0.7)
    
    
    df_a_nontest = df_a[:test_pos_a]
    df_a_test = df_a[test_pos_a:]
    
    df_d_nontest = df_d[:test_pos_d]
    df_d_test = df_d[test_pos_d:]
    
    test_df = pd.concat([df_a_test, df_d_test], axis=0)
    
    
    valid_pos_d =  int(test_pos_d*0.7)
    valid_pos_a =  int(test_pos_a*0.7)
    
    df_a_train = df_a_nontest[:valid_pos_a]
    df_a_valid = df_a_nontest[valid_pos_a:]
    
    df_d_train = df_d_nontest[:valid_pos_d]
    df_d_valid = df_d_nontest[valid_pos_d:]
    
    train_df = pd.concat([df_a_train, df_d_train], axis=0)
    valid_df = pd.concat([df_a_valid, df_d_valid], axis=0)
    
    
    print(train_df.shape)
    print(valid_df.shape)
    print(test_df.shape)
    
    
    if data_process == 1:
        x_train_df =  train_df[['AgeRange_code','gender_binary','respiratory','weakness/pain','fever','gastrointestinal','other','nausea','cardiac','high fever','kidney','asymptomatic','diabetes','neuro','NA','hypertension','cancer','ortho','respiratory_CD','cardiacs_cd','kidney_CD','blood','prostate','thyroid']]
        x_test_df =  test_df[['AgeRange_code','gender_binary','respiratory','weakness/pain','fever','gastrointestinal','other','nausea','cardiac','high fever','kidney','asymptomatic','diabetes','neuro','NA','hypertension','cancer','ortho','respiratory_CD','cardiacs_cd','kidney_CD','blood','prostate','thyroid']]
    else:
        x_train_df =  train_df[['age','gender_binary','respiratory','weakness/pain','fever','gastrointestinal','other','nausea','cardiac','high fever','kidney','asymptomatic','diabetes','neuro','NA','hypertension','cancer','ortho','respiratory_CD','cardiacs_cd','kidney_CD','blood','prostate','thyroid']]
        x_test_df =  test_df[['age','gender_binary','respiratory','weakness/pain','fever','gastrointestinal','other','nausea','cardiac','high fever','kidney','asymptomatic','diabetes','neuro','NA','hypertension','cancer','ortho','respiratory_CD','cardiacs_cd','kidney_CD','blood','prostate','thyroid']]

    y_train_df =  train_df[['death_binary']]
    y_test_df = test_df[['death_binary']]
    
    death_rate_train = (y_train_df != 0).values.sum() / len(y_train_df)
    death_rate_test = (y_test_df != 0).values.sum() / len(y_test_df)
    
    print(death_rate_train)
    #print(death_rate_train)
    print(death_rate_test)
    
    x_train = x_train_df.values
    x_test = x_test_df.values
    y_train = y_train_df.values.flatten()
    y_test = y_test_df.values.flatten()
    
    return x_train, x_test, y_train, y_test
     

patient_file = "small_records_0623.csv"
df = pd.read_csv(patient_file)
data_process = 1
if data_process == 1:
    patient_df = preprocessData_1(df)
    patient_df = patient_df.drop(['age_is_number', 'AgeRange'], axis=1)
else:
    patient_df = preprocessData_2(df)
    patient_df = patient_df.drop(['age_is_number'], axis=1)
    
#correlation_analysis(patient_df)
x_train, x_test, y_train, y_test = perpareTrainData(patient_df)


#logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_auc_score

model = LogisticRegression(solver='liblinear', random_state=0).fit(x_train, y_train)
predictions = model.predict(x_test)
tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
specificity = tn / (tn+fp)
sensitivity  = tp / (fn + tp)
roc_lr=roc_auc_score(y_test, predictions)
lr_a = metrics.accuracy_score(y_test, predictions.round())
print('')
print('for logistic regression:')
print("Accuracy:",metrics.accuracy_score(y_test, predictions))
print("Precision:",metrics.precision_score(y_test, predictions))
print("Recall:",metrics.recall_score(y_test, predictions))
print("true negative:",tn)
print("false postive:",fp)
print("false negative:",fn)
print("true postive:",tp)
print('specificity', specificity)
print('sensitivity:', sensitivity)
print('AUC: ',roc_lr)
print('********************************')
print('')

#SVM 
from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(x_train, y_train)
svm_pred = clf.predict(x_test)
tn, fp, fn, tp = confusion_matrix(y_test, svm_pred).ravel()
s_specificity = tn / (tn+fp)
s_sensitivity  = tp / (fn + tp)
roc_svm=roc_auc_score(y_test, svm_pred)
a_svm = metrics.accuracy_score(y_test, svm_pred.round())
print("for SVM")
print("Accuracy:",metrics.accuracy_score(y_test, svm_pred))
print("Precision:",metrics.precision_score(y_test, svm_pred))
print("Recall:",metrics.recall_score(y_test, svm_pred))
print("true negative:",tn)
print("false postive:",fp)
print("false negative:",fn)
print("true postive:",tp)
print('specificity:', s_specificity)
print('sensitivity:', s_sensitivity)
print('AUC: ',roc_svm)
print('********************************')
print('')

# random forest 
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(x_train, y_train);
rf_pred=rf.predict(x_test)
tn, fp, fn, tp = confusion_matrix(y_test, rf_pred.round()).ravel()
rf_specificity = tn / (tn+fp)
rf_sensitivity  = tp / (fn + tp)
roc_rf=roc_auc_score(y_test, rf_pred.round())
rf_a = metrics.accuracy_score(y_test, rf_pred.round())

print('for random forrest:')
print("Accuracy:",metrics.accuracy_score(y_test, rf_pred.round()))
print("Precision:",metrics.precision_score(y_test, rf_pred.round()))
print("Recall:",metrics.recall_score(y_test, rf_pred.round()))
print("true negative:",tn)
print("false postive:",fp)
print("false negative:",fn)
print("true postive:",tp)
print('specificity:', rf_specificity)
print('sensitivity:', rf_sensitivity)
print('AUC: ',roc_rf)
print('********************************')
print('')

rf_a = metrics.accuracy_score(y_test, rf_pred.round())
'''
for onesvm:
Accuracy: 0.8169014084507042
Precision: 0.3185840707964602
Recall: 0.972972972972973
specificity: 0.8020565552699229
sensitivity: 0.972972972972973
AUC:  0.8875147641214479
'''


'''
for autoencoder
Accuracy: 0.9976525821596244
Precision: 1.0
Recall: 0.972972972972973
specificity: 1.0
sensitivity: 0.972972972972973
AUC:  0.9864864864864865
'''

#make a bar chart
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

barWidth = 0.15

#set height
bars1 = [metrics.accuracy_score(y_test, predictions),metrics.accuracy_score(y_test, svm_pred),rf_a, 0.8169014084507042, 0.9976525821596244 ]
bars2 = [specificity, s_specificity, rf_specificity, 0.8020565552699229, 1]
bars3 = [sensitivity, s_sensitivity, rf_sensitivity,0.972972972972973,0.972972972972973  ]
bars4 = [roc_lr, roc_svm, roc_rf, 0.8875147641214479, 0.9864864864864865]

#set position of bar on X axis 
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]
r6 = [x + barWidth for x in r5]

# Make the plot
plt.bar(r1, bars1, color='#87CEFA', width=barWidth, edgecolor='white', label='var1')
plt.bar(r2, bars2, color='#FFC1C1', width=barWidth, edgecolor='white', label='var2')
plt.bar(r3, bars3, color='#FFE1FF', width=barWidth, edgecolor='white', label='var3')
plt.bar(r4, bars4, color='#27408B', width=barWidth, edgecolor='white', label='var4') 
plt.xticks([r + barWidth for r in range(len(bars1))], ['Logistic Regression','SVM','Random Forrest', 'SVM One Class', 'Autoencoder'], rotation = 'vertical')
plt.show

metric = ['Accuracy', 'Specificity', 'Sensitivity', 'AUC']
plt.legend(metric, loc = 7)

