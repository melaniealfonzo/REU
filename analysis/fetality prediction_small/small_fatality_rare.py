
import pandas as pd 
import datetime
import numpy as np
import seaborn
import matplotlib.pyplot as plt


import seaborn as sns
from pylab import rcParams
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, classification_report, auc, roc_curve
from sklearn.metrics import precision_recall_fscore_support, f1_score
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
from keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau

set_random_seed(2)
SEED = 123 #used to help randomly select the data points
DATA_SPLIT_PCT = 0.2
rcParams['figure.figsize'] = 8, 6
LABELS = ["Normal","Break"]

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
        train_df =  train_df[['AgeRange_code','gender_binary','respiratory','weakness/pain','fever','gastrointestinal','other','nausea','cardiac','high fever','kidney','asymptomatic','diabetes','neuro','NA','hypertension','cancer','ortho','respiratory_CD','cardiacs_cd','kidney_CD','blood','prostate','thyroid', 'death_binary']]
        test_df =  test_df[['AgeRange_code','gender_binary','respiratory','weakness/pain','fever','gastrointestinal','other','nausea','cardiac','high fever','kidney','asymptomatic','diabetes','neuro','NA','hypertension','cancer','ortho','respiratory_CD','cardiacs_cd','kidney_CD','blood','prostate','thyroid', 'death_binary']]
        valid_df =  valid_df[['AgeRange_code','gender_binary','respiratory','weakness/pain','fever','gastrointestinal','other','nausea','cardiac','high fever','kidney','asymptomatic','diabetes','neuro','NA','hypertension','cancer','ortho','respiratory_CD','cardiacs_cd','kidney_CD','blood','prostate','thyroid', 'death_binary']]
        
    else:
        train_df =  train_df[['age','gender_binary','respiratory','weakness/pain','fever','gastrointestinal','other','nausea','cardiac','high fever','kidney','asymptomatic','diabetes','neuro','NA','hypertension','cancer','ortho','respiratory_CD','cardiacs_cd','kidney_CD','blood','prostate','thyroid', 'death_binary']]
        test_df =  test_df[['age','gender_binary','respiratory','weakness/pain','fever','gastrointestinal','other','nausea','cardiac','high fever','kidney','asymptomatic','diabetes','neuro','NA','hypertension','cancer','ortho','respiratory_CD','cardiacs_cd','kidney_CD','blood','prostate','thyroid', 'death_binary']]
        valid_df =  valid_df[['age','gender_binary','respiratory','weakness/pain','fever','gastrointestinal','other','nausea','cardiac','high fever','kidney','asymptomatic','diabetes','neuro','NA','hypertension','cancer','ortho','respiratory_CD','cardiacs_cd','kidney_CD','blood','prostate','thyroid', 'death_binary']]
  

    y_train_df =  train_df[['death_binary']]
    y_test_df = test_df[['death_binary']]
    y_valid_df = valid_df[['death_binary']]
    
    death_rate_train = (y_train_df != 0).values.sum() / len(y_train_df)
    death_rate_test = (y_test_df != 0).values.sum() / len(y_test_df)
    death_rate_valid = (y_valid_df != 0).values.sum() / len(y_valid_df)
    
    print(death_rate_train)
    print(death_rate_valid)
    print(death_rate_test)

    train_df = train_df.rename(columns={"death_binary" : "death"})
    valid_df = valid_df.rename(columns={"death_binary" : "death"})
    test_df = test_df.rename(columns={"death_binary" : "death"})
    
    return train_df, valid_df, test_df

    
patient_file = "small_records_0623.csv"
df = pd.read_csv(patient_file)

data_process = 1
if data_process == 1:
    patient_df = preprocessData_1(df)
    #patient_df = patient_df.drop(['age_is_number', 'AgeRange'], axis=1)
else:
    patient_df = preprocessData_2(df)
    #patient_df = patient_df.drop(['age_is_number'], axis=1)
#correlation_analysis(patient_df)
#drop 

df_train, df_valid, df_test = perpareTrainData(patient_df)

df_train_0 = df_train.loc[df_train['death'] == 0]
df_train_1 = df_train.loc[df_train['death'] == 1]
df_train_0_x = df_train_0.drop(['death'], axis=1)
df_train_1_x = df_train_1.drop(['death'], axis=1)

df_valid_0 = df_valid.loc[df_valid['death'] == 0]
df_valid_1 = df_valid.loc[df_valid['death'] == 1]
df_valid_0_x = df_valid_0.drop(['death'], axis=1)
df_valid_1_x = df_valid_1.drop(['death'], axis=1)

df_test_0 = df_test.loc[df_test['death'] == 0]
df_test_1 = df_test.loc[df_test['death'] == 1]
df_test_0_x = df_test_0.drop(['death'], axis=1)
df_test_1_x = df_test_1.drop(['death'], axis=1)


scaler = StandardScaler().fit(df_train_0_x)
df_train_0_x_rescaled = scaler.transform(df_train_0_x)
df_valid_0_x_rescaled = scaler.transform(df_valid_0_x)
df_valid_x_rescaled = scaler.transform(df_valid.drop(['death'], axis = 1))
df_test_0_x_rescaled = scaler.transform(df_test_0_x)
df_test_x_rescaled = scaler.transform(df_test.drop(['death'], axis = 1))


'''
#train
nb_epoch = 1000
batch_size = 32
input_dim = df_train_0_x_rescaled.shape[1] #num of predictor variables, 
encoding_dim = 8
hidden_dim = int(encoding_dim / 2)
learning_rate = 1e-3

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="relu", activity_regularizer=regularizers.l1(learning_rate))(input_layer)
encoder = Dense(hidden_dim, activation="relu")(encoder)
decoder = Dense(hidden_dim, activation="relu")(encoder)
decoder = Dense(encoding_dim, activation="relu")(decoder)
decoder = Dense(input_dim, activation="linear")(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.summary()


autoencoder.compile(metrics=['mse'],
                    loss='mean_squared_error',
                    optimizer='adam')


#earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('autoencoder_classifier.h5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, epsilon=1e-4, mode='min')


history = autoencoder.fit(df_train_0_x_rescaled, df_train_0_x_rescaled,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(df_valid_0_x_rescaled, df_valid_0_x_rescaled),
                    verbose=1,
                    callbacks=[ mcp_save, reduce_lr_loss]
                    ).history


autoencoder.load_weights("autoencoder_classifier.h5")
#classification
valid_x_predictions = autoencoder.predict(df_valid_x_rescaled)
mse = np.mean(np.power(df_valid_x_rescaled - valid_x_predictions, 2), axis=1)
error_df = pd.DataFrame({'Reconstruction_error': mse,
                        'True_class': df_valid['death']})
precision_rt, recall_rt, threshold_rt = precision_recall_curve(error_df.True_class, error_df.Reconstruction_error)
f1_rt = 2*(precision_rt*recall_rt)/(precision_rt+recall_rt)
plt.plot(threshold_rt, precision_rt[1:], label="Precision",linewidth=5)
plt.plot(threshold_rt, recall_rt[1:], label="Recall",linewidth=5)
plt.plot(threshold_rt, f1_rt[1:], label="F1",linewidth=5)
plt.title('Precision and recall for different threshold values')
plt.xlabel('Threshold')
plt.ylabel('Precision/Recall')
plt.legend()
plt.show()

#test
test_x_predictions = autoencoder.predict(df_test_x_rescaled)
mse = np.mean(np.power(df_test_x_rescaled - test_x_predictions, 2), axis=1)
error_df_test = pd.DataFrame({'Reconstruction_error': mse,
                        'True_class': df_test['death']})
error_df_test = error_df_test.reset_index()
threshold_fixed = 4
groups = error_df_test.groupby('True_class')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.index, group.Reconstruction_error, marker='o', ms=3.5, linestyle='',
            label= "Death" if name == 1 else "Alive")
ax.hlines(threshold_fixed, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show();

pred_y = [1 if e > threshold_fixed else 0 for e in error_df_test.Reconstruction_error.values]
conf_matrix = confusion_matrix(error_df_test.True_class, pred_y)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()


from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(error_df_test.True_class, pred_y)
print('for encoder:')
print("Accuracy:",metrics.accuracy_score(error_df_test.True_class, pred_y))
print("Precision:",metrics.precision_score(error_df_test.True_class, pred_y))
print("Recall:",metrics.recall_score(error_df_test.True_class, pred_y))

tn, fp, fn, tp = confusion_matrix(error_df_test.True_class, pred_y).ravel()
specificity = tn / (tn+fp)
print('specificity:', specificity)

sensitivity  = tp / (fn + tp)
print('sensitivity:', sensitivity)

from sklearn.metrics import roc_auc_score
roc_rf=roc_auc_score(error_df_test.True_class, pred_y)
print('AUC: ',roc_rf)

false_pos_rate, true_pos_rate, thresholds = roc_curve(error_df_test.True_class, error_df_test.Reconstruction_error)
roc_auc = auc(false_pos_rate, true_pos_rate,)
plt.plot(false_pos_rate, true_pos_rate, linewidth=5, label='AUC = %0.3f'% roc_auc)
plt.plot([0,1],[0,1], linewidth=5)
plt.xlim([-0.01, 1])
plt.ylim([0, 1.01])
plt.legend(loc='lower right')
plt.title('Receiver operating characteristic curve (ROC)')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
'''

#onesvm
from sklearn.svm import OneClassSVM
from sklearn import metrics
from sklearn.metrics import roc_auc_score
clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf.fit(df_train_0_x_rescaled)
y_pred_train = clf.predict(df_train_0_x_rescaled)
y_pred_test = clf.predict(df_test_x_rescaled)

pred_y = [1 if e == -1 else 0 for e in y_pred_test]
conf_matrix = confusion_matrix(df_test['death'], pred_y)
cnf_matrix = metrics.confusion_matrix(df_test['death'], pred_y)
tn, fp, fn, tp = confusion_matrix(df_test['death'], pred_y).ravel()
specificity = tn / (tn+fp)
sensitivity  = tp / (fn + tp)
roc_rf=roc_auc_score(df_test['death'], pred_y)

print('for onesvm:')
print("Accuracy:",metrics.accuracy_score(df_test['death'], pred_y))
print("Precision:",metrics.precision_score(df_test['death'], pred_y))
print("Recall:",metrics.recall_score(df_test['death'], pred_y))
print('specificity:', specificity)
print('sensitivity:', sensitivity)
print('AUC: ',roc_rf)
