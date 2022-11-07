# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 21:42:55 2022

@author: jlusk
"""



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
################################# Phase 2 and 3 ###################################################################
#dfTrainStandardized = dfTrain2Numerical
dfTrain2Numerical= dfTrain2Numerical[['UrlLength_Standardized','NumNumericChars_Standardized','NumDots_Standardized','SubdomainLevel_Standardized','PathLevel_Standardized','NumDash_Standardized','NumDashInHostname_Standardized','NumUnderscore_Standardized','NumPercent_Standardized','NumQueryComponents_Standardized','NumAmpersand_Standardized','NumHash_Standardized','HostnameLength_Standardized','PathLength_Standardized','QueryLength_Standardized','NumSensitiveWords_Standardized']]

# Next, must combine dfTrain2Numerical with binary data still in dfTrain2
dfTrain3= dfTrain2Numerical.join(dfTrain2[['AtSymbol','TildeSymbol','NoHttps','RandomString','IpAddress','DomainInSubdomains','DomainInPaths','HttpsInHostname','DoubleSlashInPath','EmbeddedBrandName','PctExtResourceUrls','ExtFavicon','InsecureForms','RelativeFormAction','ExtFormAction','RightClickDisabled','PopUpWindow','IframeOrFrame','MissingTitle','ImagesOnlyInForm']])

# Should drop 'HttpsInHostname' as each case =0. Will mess up correlation otherwise
dfTrain3=dfTrain3.drop('HttpsInHostname',axis=1)

# Perform standard feature correlation on entirety of dfTrain3 by feature
dfTrain3corr=dfTrain3.corr()
dfClass_labels = pd.DataFrame(dfTrain2['CLASS_LABEL'])
# Perform manhattan measure
from sklearn.metrics.pairwise import manhattan_distances
distallpairs = manhattan_distances(dfTrain3.iloc[0:35],dfTrain3[0:35])
print("Manhattan distance among all cases:\n", distallpairs)

### investigate the relationship between the target and the features##
## area under the curve, closer to one the more accurate ##
Y = dfClass_labels
X = dfTrain3
from sklearn.model_selection import train_test_split
from sklearn import tree
train, test = train_test_split(dfTrain3, test_size=0.2, random_state=0)
clf = tree.DecisionTreeClassifier(min_samples_split=0.3)
clf = clf.fit(X,Y)
Y_predicted = clf.predict(X)
print(Y_predicted)

from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve
auc = roc_auc_score(Y,Y_predicted)
print(auc)

### kfold cross validation ###
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, random_state=None, shuffle=False)
X=np.random.rand(10,2)
Y=np.random.rand(10,2)
for train_index, test_index in kf.split(X):
    print('train:', train_index, 'test:', test_index)
    X_train=X[train_index]
    X_test=X[test_index]
    Y_train=Y[train_index]
    Y_test=Y[test_index]
    print('train_x:', X_train, 'test_x:', X_test)
    print('train_y:', Y_train, 'test_y:', Y_test)

### Grid Search ###
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

max_depth_params=[2,3,4,5,6,7,8,10,12,14,15,16,20]
base_model = DecisionTreeClassifier()
searched_parameters=[{'max_depth':[2,3,4,5,6,7,8,10,12,14],'min_sample_':[0.05,0.01,0.02]}]
s=['precision','recall','f1']
clf=GridSearchCV(estimator=base_model, param_grid=searched_parameters,scoring=s,refit='f1',cv=10,verbose=3)
Y=dfClass_labels
X=dfTrain3
clf.fit(X,Y)

print(clf.cv_results_['mean_test_score'])
print(clf.cv_results_['mean_test_precision'])
print(clf.cv_results_['mean_test_recall'])

finalmodel = DecisionTreeClassifier(max_depth=10,min_samples_split=0.01)
finalmodel.fit(X,Y)


