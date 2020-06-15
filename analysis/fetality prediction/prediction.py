import csv
import pandas as pd 
import datetime
import numpy as np
import seaborn
import matplotlib.pyplot as plt

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
    
def perpareData(df):
    df['date_confirmation']= pd.to_datetime(df['date_confirmation']) 
    select_day = datetime.datetime(2020, 5, 29) 
    train_df = df.loc[(df.date_confirmation <= select_day)]
    test_df = df.loc[(df.date_confirmation > select_day)]
    
    x_train_df =  train_df[['age','latitude', 'longitude', 'chronic_disease_binary','travel_history_binary','combine_symptoms', 'gender_binary']]
    x_test_df =  test_df[['age','latitude', 'longitude', 'chronic_disease_binary','travel_history_binary','combine_symptoms', 'gender_binary']]
    y_train_df =  train_df[['death']]
    y_test_df = test_df[['death']]
    
    death_rate_train = (y_train_df != 0).values.sum() / len(y_train_df)
    death_rate_test = (y_test_df != 0).values.sum() / len(y_test_df)
    
    print(death_rate_train)
    print(death_rate_test)
    
    x_train = x_train_df.values
    x_test = x_test_df.values.flatten()
    y_train = y_train_df.values
    y_test = y_test_df.values.flatten()
    
    return x_train, x_test, y_train, y_test
    
    

patient_file = "patient records.csv"
df = pd.read_csv(patient_file)

#correlation_analysis(df)
df['date_confirmation']= pd.to_datetime(df['date_confirmation']) 
select_day = datetime.datetime(2020, 5, 29) 
train_df = df.loc[(df.date_confirmation <= select_day)]
test_df = df.loc[(df.date_confirmation > select_day)]

x_train_df =  train_df[['age','latitude', 'longitude', 'chronic_disease_binary','travel_history_binary','combine_symptoms', 'gender_binary']]
x_test_df =  test_df[['age','latitude', 'longitude', 'chronic_disease_binary','travel_history_binary','combine_symptoms', 'gender_binary']]
y_train_df =  train_df[['death']]
y_test_df = test_df[['death']]

death_rate_train = (y_train_df != 0).values.sum() / len(y_train_df)
death_rate_test = (y_test_df != 0).values.sum() / len(y_test_df)

x_train = x_train_df.values
x_test = x_test_df.values.flatten()
y_train = y_train_df.values
y_test = y_test_df.values.flatten()


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

