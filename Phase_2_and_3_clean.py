# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 21:56:56 2022

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
dfTrain2Numerical = dfTrain2[['NumNumericChars','NumDots','SubdomainLevel','PathLevel','NumDash','NumDashInHostname',
          'NumUnderscore','NumPercent','NumQueryComponents','NumAmpersand','NumHash','HostnameLength','PathLength','QueryLength','NumSensitiveWords']]
clf= LocalOutlierFactor(n_neighbors=20)
X=dfTrain2Numerical.to_numpy()
outlier_label=clf.fit_predict(X)
rows_to_drop= dfTrain2.iloc[clf.negative_outlier_factor_ < -1.30].index
dfTrain2.drop(rows_to_drop,inplace=True)
dfTrain2Numerical = dfTrain2[['NumNumericChars','NumDots','SubdomainLevel','PathLevel','NumDash','NumDashInHostname','NumUnderscore','NumPercent','NumQueryComponents','NumAmpersand','NumHash','HostnameLength','PathLength','QueryLength','NumSensitiveWords']]
X = dfTrain2Numerical.to_numpy()
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
dfTrain2Numerical.is_copy = False
dfTrain2Numerical[['NumNumericChars_Standardized','NumDots_Standardized','SubdomainLevel_Standardized','PathLevel_Standardized','NumDash_Standardized','NumDashInHostname_Standardized','NumUnderscore_Standardized','NumPercent_Standardized','NumQueryComponents_Standardized','NumAmpersand_Standardized','NumHash_Standardized','HostnameLength_Standardized','PathLength_Standardized','QueryLength_Standardized','NumSensitiveWords_Standardized']]=X
################################# Phase 2 and 3 ###################################################################
### Identify the features that are highly correlated ###
## Manhattan distance ##
from sklearn.metrics.pairwise import manhattan_distances
distallpairs = manhattan_distances(dfTrain2Numerical)
print(distallpairs)

from sklearn.metrics import silhouette_samples
labels=[]
silh = silhouette_samples(dfTrain2Numerical,labels)


#####################
### cluster model ###
from sklearn.cluster import KMeans

cluster_model = KMeans(n_clusters=30, init="k-means++",max_iter=300, n_init=15, random_state=0)
cluster_model.fit(dfTrain2Numerical)
centers = cluster_model.cluster_centers_
pred_y = cluster_model.fit_predict(dfTrain2Numerical)

## elbow method ##
inertia_elbow = []
for k in range (1,30):
    cluster_model_elbow = KMeans(n_clusters=k, init="k-means++",max_iter=300, n_init=10, random_state=0)
    cluster_model_elbow.fit(dfTrain2Numerical)
    inertia_elbow.append(cluster_model_elbow.inertia_)
    
plt.figure()
print(inertia_elbow)
plt.plot(range(1,30),inertia_elbow)
plt.xlabel('K')
plt.ylabel('inertia') 
plt.show()   

## silhouette method ##
from sklearn.metrics import silhouette_score

sil=[]
for k in range (2,30):
    cluster_model_sil = KMeans(n_clusters = k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y = cluster_model_sil.fit_predict(dfTrain2Numerical)
    sil.append(silhouette_score(dfTrain2Numerical, pred_y, metric = 'euclidean'))

plt.figure()
print(sil)
plt.plot(range(2,30),sil)
plt.title('silhouette method')
plt.xlabel('k')
plt.ylabel('silhouette')
plt.show()

# 














