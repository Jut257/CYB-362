#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 19:52:38 2022

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
dfTrain = pd.read_csv("Phishing_Legitimate_train_missing_data.csv", na_values=['',' ','n/a'])
################################# Phase 0 and 1 ####################################################
trainRowsWithNa = dfTrain[ dfTrain.isnull().any(axis=1) ]
rowsToDrop = dfTrain[ dfTrain.isnull().sum(axis=1) > 1 ].index
dfTrain.drop(rowsToDrop, inplace=True)
imputer = KNNImputer(n_neighbors=10)
dfTrain2 = pd.DataFrame(imputer.fit_transform(dfTrain),columns = dfTrain.columns)
rows_to_drop=dfTrain2[dfTrain2['UrlLength']>500].index
dfTrain2.drop(rows_to_drop,inplace=True)
rows_to_drop=dfTrain2[dfTrain2['NumNumericChars']>100].index
dfTrain2.drop(rows_to_drop,inplace=True)
rows_to_drop=dfTrain2[dfTrain2['NumDash']>20].index
dfTrain2.drop(rows_to_drop,inplace=True)
dfTrain2Numerical = dfTrain2[['NumNumericChars','NumDots','SubdomainLevel','PathLevel','UrlLength','NumDash','NumDashInHostname',
          'NumUnderscore','NumPercent','NumQueryComponents','NumAmpersand','NumHash','HostnameLength','PathLength','QueryLength','NumSensitiveWords']]
clf= LocalOutlierFactor(n_neighbors=20)
X=dfTrain2Numerical.to_numpy()
outlier_label=clf.fit_predict(X)
rows_to_drop= dfTrain2.iloc[clf.negative_outlier_factor_ < -1.30].index
dfTrain2.drop(rows_to_drop,inplace=True)
dfTrain2Numerical = dfTrain2[['NumNumericChars','NumDots','SubdomainLevel','PathLevel','UrlLength','NumDash','NumDashInHostname','NumUnderscore','NumPercent','NumQueryComponents','NumAmpersand','NumHash','HostnameLength','PathLength','QueryLength','NumSensitiveWords']]
X = dfTrain2Numerical.to_numpy()
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
dfTrain2Numerical.is_copy = False
dfTrain2Numerical[['NumNumericChars_Standardized','NumDots_Standardized','SubdomainLevel_Standardized','PathLevel_Standardized','UrlLength_Standardized','NumDash_Standardized','NumDashInHostname_Standardized','NumUnderscore_Standardized','NumPercent_Standardized','NumQueryComponents_Standardized','NumAmpersand_Standardized','NumHash_Standardized','HostnameLength_Standardized','PathLength_Standardized','QueryLength_Standardized','NumSensitiveWords_Standardized']]=X

## Phase 3/4 Code.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Create dataframe of target 'CLASS_LABEL'
dfClass_labels = pd.DataFrame(dfTrain2['CLASS_LABEL'])

# Create dataframe of standardized numerical features
dfTrain2Numstd= dfTrain2Numerical[['UrlLength_Standardized','NumNumericChars_Standardized','NumDots_Standardized','SubdomainLevel_Standardized','PathLevel_Standardized','NumDash_Standardized','NumDashInHostname_Standardized','NumUnderscore_Standardized','NumPercent_Standardized','NumQueryComponents_Standardized','NumAmpersand_Standardized','NumHash_Standardized','HostnameLength_Standardized','PathLength_Standardized','QueryLength_Standardized','NumSensitiveWords_Standardized']]

# Created dataframe of all binary data, now binary and standardized numerical features can be seperated IF NEEDED.
dfTrain2Bin= dfTrain2[['AtSymbol','TildeSymbol','NoHttps','RandomString','IpAddress','DomainInSubdomains','DomainInPaths','DoubleSlashInPath','EmbeddedBrandName','PctExtResourceUrls','ExtFavicon','InsecureForms','RelativeFormAction','ExtFormAction','RightClickDisabled','PopUpWindow','IframeOrFrame','MissingTitle','ImagesOnlyInForm']]
# dfTrain2Bin contains binary features and dfTrain2Numstd contains standardized numerical data. 

# Create main dataframe that contains standardized numerical data and binary data
dfTrain3=dfTrain2Bin.join(dfTrain2Numstd)
# Create list of all features
featurecols=dfTrain3.columns.to_list()

# Create seperate lists for numerical & binary features lists
NUMfeaturecols=dfTrain2Numstd.columns.to_list()
BINfeaturecols=dfTrain2Bin.columns.to_list()

####### Dataframe list: ######
# 'dfTrain3' = Main dataframe of training data, std numerical and binary data 
# 'featurecols' = list of all feature cols in 'dfTrain3'
# 'NUMfeaturecols' = list of all numerical features 
# 'BINfeaturecols' = list of all binary features
# 'dfTrain2Bin' = all binary features 
# 'dfTrain2Numstd' = all numerical features standardized

# Create dataframe of target 'CLASS_LABEL'
dfClass_labels = pd.DataFrame(dfTrain2['CLASS_LABEL'])
# Add 'dfClass_labels' to main dfTrain3 
dfTrain3 = dfTrain3.join(dfClass_labels)
chosenNumeric = dfTrain3[['CLASS_LABEL','NumDash_Standardized','PathLength_Standardized','UrlLength_Standardized','PathLevel_Standardized']]

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
columns_to_select=['AtSymbol','TildeSymbol','NoHttps','RandomString','IpAddress','DomainInSubdomains','DomainInPaths','DoubleSlashInPath','EmbeddedBrandName','PctExtResourceUrls','ExtFavicon','InsecureForms','RelativeFormAction','ExtFormAction','RightClickDisabled','PopUpWindow','IframeOrFrame','MissingTitle','ImagesOnlyInForm']

rfe_selector = RFE(estimator=LogisticRegression(),n_features_to_select = 3, step = 1)
rfe_selector.fit(dfTrain3[columns_to_select], dfTrain3['CLASS_LABEL'])
dfTrain3[columns_to_select].columns[ rfe_selector.get_support() ]
columns_to_plot=['NoHttps','IpAddress', 'InsecureForms','CLASS_LABEL']

### End Data Cleaning Section

## Final Features Selected from Phase 2 & 3
### Final Feature Selection:
#1. 'NumDash_Standardized'
#2. 'PathLength_Standardized'
#3. 'UrlLength_Standardized'
#4. 'PathLevel_Standardized'
#5. 'DomainsInPaths'
#6. 'RandomString'
#7. 'InsecureForms'

## Start Phase V ##
from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

clf = tree.DecisionTreeClassifier(max_depth=(6))

# make dataframe features only selected features
dfFeatures=dfTrain3[['NumDash_Standardized','PathLength_Standardized','UrlLength_Standardized','PathLevel_Standardized','DomainInPaths','RandomString','InsecureForms']]
Y=dfTrain3['CLASS_LABEL']
X=dfFeatures

# Split train/test data

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.50,shuffle=False,random_state=1)

###### Random forest classifier 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
from sklearn.ensemble import RandomForestClassifier
rfclf=RandomForestClassifier(n_estimators=100, max_depth=8,random_state=1, max_samples=400)
rfclf.fit(X_train, Y_train)

Y_test_predicted=rfclf.predict(X_test)
 
auc=roc_auc_score(Y_test, Y_test_predicted)
print(auc)
fig, ax = plt.subplots()
plot_roc_curve(rfclf, X_test, Y_test,name='Forest', lw=1, ax=ax)
 
plt.show()

####### Support Vector Classification #####
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

## Build a SVM model
Y=dfTrain3['CLASS_LABEL'].to_numpy()
X=dfFeatures.to_numpy()

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC

########## Select best parameters and display the results ######### 

# This section of code will always be the same, only thing that will change will be the type
base_model=SVC()
kernel_values=['rbf']
c_values=np.linspace(77.8,10,endpoint=True)
kf=KFold(n_splits=5,shuffle=True)
tuned_parameters=[{'C':c_values,'kernel':kernel_values}]
scores=['precision','recall','f1','roc_auc']

from sklearn.model_selection import GridSearchCV

clf=GridSearchCV(estimator=base_model,
                 param_grid=tuned_parameters,
                 scoring=scores,
                 refit='f1',
                 cv=10,
                 verbose=3)

clf.fit(X,Y) # Look at variabe in 'best_params_' for best 
#'cv_results_' calculations for each fold
print(clf.best_estimator_) #Shows the best estimator & kernel
print(clf.best_params_) # Shows the best parameters, kerel and c-value
print(clf.cv_results_['params'])

# Take the results from (clf.cv_results_['params']) and create a dataframe to store results
df_results=pd.DataFrame(clf.cv_results_['params'])
df_results['f1']=clf.cv_results_['mean_test_f1']
df_results['auc']=clf.cv_results_['mean_test_roc_auc']
df_results['precision']=clf.cv_results_['mean_test_precision']

# Visualize df_results to find best kernel algorithm and c-value
# performance of f1 score
sns.lineplot(data=df_results,x='C',y='f1',hue='kernel')

# Look at precision instead of 'f1'
#sns.lineplot(data=df_results,x='C',y='precision',hue='kernel')

### Build the final model #####
####### Building the final model using SVM ########
final_model=SVC().set_params(**clf.best_params_) # **clf.best_params_ will keep the best params automatically
final_model.fit(X,Y) # Fit the final model with the entire traning set 

##### Best c = 77.8, kernel = 'rbf' #######

# Performance metrics of the final model based on the C value found prior
from sklearn import svm
clf = svm.SVC(C=77.8,kernel='rbf')
clf.fit(X, Y)
Y_pre=clf.predict(X)
 
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
 
print('Accuracy:', accuracy_score(Y, Y_pre))
print('Precision:',precision_score(Y, Y_pre))
print('Recall:', recall_score(Y, Y_pre))
print('Confusion Matrix', confusion_matrix(Y, Y_pre))
plot_confusion_matrix(clf, X, Y)

# Performance metrics of the final model based on the C value found prior
from sklearn import svm

final_model.fit(X, Y)
Y_pre=clf.predict(X)




