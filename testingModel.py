#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 22:04:47 2022

@author: jprince
"""

### Phase 0/1 Code.
import scipy.cluster.hierarchy as sch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


dfTest = pd.read_csv("Phishing_Legitimate_test_student.csv", na_values=['',' ','n/a'])

################################# Data Cleaning of Test csv ############################################

## cleaning & standardizing testing data ##
testRowsWithNa = dfTest[ dfTest.isnull().any(axis=1) ]
rowsToDrop = dfTest[ dfTest.isnull().sum(axis=1) > 1 ].index
dfTest.drop(rowsToDrop, inplace=True)
imputer = KNNImputer(n_neighbors=10)
dfTest2 = pd.DataFrame(imputer.fit_transform(dfTest),columns = dfTest.columns)
rows_to_drop=dfTest2[dfTest2['UrlLength']>500].index
dfTest2.drop(rows_to_drop,inplace=True)
rows_to_drop=dfTest2[dfTest2['NumNumericChars']>100].index
dfTest2.drop(rows_to_drop,inplace=True)
rows_to_drop=dfTest2[dfTest2['NumDash']>20].index
dfTest2.drop(rows_to_drop,inplace=True)
clf= LocalOutlierFactor(n_neighbors=20)
X=dfTest2.to_numpy()
outlier_label=clf.fit_predict(X)
rows_to_drop= dfTest2.iloc[clf.negative_outlier_factor_ < -1.30].index
dfTest2.drop(rows_to_drop,inplace=True)

# Dataframe of selected features, must be loaded in same order as in training set
X=dfTest2[['NumDash','PathLength','UrlLength','PathLevel','DomainInPaths','RandomString','InsecureForms']].to_numpy()

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Import model
import pickle


load_model=pickle.load(open('test','rb'))

# Test same model on different dataset 

y=load_model.predict(X) # Build prediction model


### export to csv ###
ids = dfTest2['id'].astype(int)
# DF created to output the csv file in correct format
final_predictions = pd.DataFrame(y,ids).rename_axis('Id')
final_predictions.columns = ['Prediction']
# must change file path 
final_predictions.to_csv('test_export.csv')


######### TEST IMPORTED MODEL ON ENTIRE TRAINING SET ########### 
### Test model on training dataset ### This can be removed prior to submittal
### Phase 0/1 Code.
import scipy.cluster.hierarchy as sch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


dfTest = pd.read_csv("Phishing_Legitimate_train_missing_data.csv", na_values=['',' ','n/a'])

##### Data Cleaning of Test csv ##

## cleaning & standardizing training data ##
testRowsWithNa = dfTest[ dfTest.isnull().any(axis=1) ]
rowsToDrop = dfTest[ dfTest.isnull().sum(axis=1) > 1 ].index
dfTest.drop(rowsToDrop, inplace=True)
imputer = KNNImputer(n_neighbors=10)
dfTest2 = pd.DataFrame(imputer.fit_transform(dfTest),columns = dfTest.columns)
rows_to_drop=dfTest2[dfTest2['UrlLength']>500].index
dfTest2.drop(rows_to_drop,inplace=True)
rows_to_drop=dfTest2[dfTest2['NumNumericChars']>100].index
dfTest2.drop(rows_to_drop,inplace=True)
rows_to_drop=dfTest2[dfTest2['NumDash']>20].index
dfTest2.drop(rows_to_drop,inplace=True)
clf= LocalOutlierFactor(n_neighbors=20)
X=dfTest2.to_numpy()
outlier_label=clf.fit_predict(X)
rows_to_drop= dfTest2.iloc[clf.negative_outlier_factor_ < -1.30].index
dfTest2.drop(rows_to_drop,inplace=True)

# Dataframe of selected features, must be loaded in same order as in training set
X=dfTest2[['NumDash','PathLength','UrlLength','PathLevel','DomainInPaths','RandomString','InsecureForms']].to_numpy()

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# Import model
import pickle
load_model=pickle.load(open('test','rb'))

# Test same model on different dataset 

y=load_model.predict(X) # Build prediction model
Y=dfTest2['CLASS_LABEL'].to_numpy()

# ### export to csv ###
# ids = dfTest2['id'].astype(int)
# # DF created to output the csv file in correct format
# final_predictions = pd.DataFrame(y,ids).rename_axis('Id')
# final_predictions.columns = ['Prediction']
# # must change file path 
# final_predictions.to_csv('test_export.csv')

# Test model
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


accuracy_score(Y, y) # LoL bad
print('Confusion Matrix', confusion_matrix(Y, y)) # Big Problem with FN














