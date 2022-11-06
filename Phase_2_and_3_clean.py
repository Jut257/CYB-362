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
dfTrainStandardized = dfTrain2Numerical

### Identify the features that are highly correlated ###
## Manhattan distance ##
# euclidean isnt good for multiple features
from sklearn.metrics.pairwise import manhattan_distances
distallpairs = manhattan_distances(dfTrainStandardized.iloc[0:30],dfTrainStandardized[0:30])
print(distallpairs)
# all cases against all cases
#dist_test = manhattan_distances(dfTrainStandardized)

######### needs work/ might not be viable ##########
from sklearn.metrics import silhouette_samples
labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
silh = silhouette_samples(dfTrainStandardized, labels)
################################

#####################
### cluster model ###
from sklearn.cluster import KMeans
###############################################################
cluster_model = KMeans(n_clusters=30, init="k-means++",max_iter=300, n_init=15, random_state=0)
cluster_model.fit(dfTrainStandardized)
centers = cluster_model.cluster_centers_
print(centers)
inertia = cluster_model.inertia_
print(inertia)
plt.figure()
pred_y = cluster_model.fit_predict(dfTrainStandardized)
# testing different vars
plt.scatter(dfTrainStandardized, dfTrainStandardized.iloc[1:30], c = pred_y.astype(float), s=10)
plt.scatter(centers[:,0], centers[:,1], c='blue', s=50)
plt.show()

print(list(dfTrainStandardized[0:30].columns))
###############################################################
## elbow method ##
inertia_elbow = []
for k in range (1,30):
    cluster_model_elbow = KMeans(n_clusters=k, init="k-means++",max_iter=300, n_init=10, random_state=0)
    cluster_model_elbow.fit(dfTrainStandardized)
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
    pred_y = cluster_model_sil.fit_predict(dfTrainStandardized)
    sil.append(silhouette_score(dfTrainStandardized, pred_y, metric = 'euclidean'))

plt.figure()
print(sil)
plt.plot(range(2,30),sil)
plt.title('silhouette method')
plt.xlabel('k')
plt.ylabel('silhouette')
plt.show()

# comparative analysis to determine best value of K #
# run "pip install yellowbrick" from console 
from yellowbrick.cluster import SilhouetteVisualizer
fig, ax = plt.subplots(4, 2, figsize=(15,8))
for i in [2,3,4,5,6,7,8,9,10,11,12,13,14,15]:
    cluster_model_YB = KMeans(n_clusters = i, init='k-means++', max_iter=100, n_init=10, random_state=0)
    q, mod = divmod(i,2)
    visualizer = SilhouetteVisualizer(cluster_model_YB, colors='yellowbrick', ax=ax[q-1][mod])
    visualizer.fit(dfTrainStandardized)
  
### Hierarchical Clustering ###
## Dendrogram ##
# doesnt display graph in jupyter if you print(dend)
# case names appear as black bar(too many, too close)
# find out names of cases to narrow down the 7
import scipy.cluster.hierarchy as sch
plt.figure()
dend = sch.dendrogram(sch.linkage(dfTrainStandardized, method = 'ward'))
plt.axhline(y=750, linestyle='--', c='orange')
plt.title('dendrogram')
plt.xlabel('cases')
plt.ylabel('distance')
plt.show()

## Agglomerative Clustering ##
from sklearn.cluster import AgglomerativeClustering
cluster_model_Agglo = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster_model_Agglo.fit(dfTrainStandardized)
print(cluster_model_Agglo.n_clusters)
print(cluster_model_Agglo.labels_)
#let_me_see = cluster_model_Agglo.labels_

### Training split into training and validation sets ###
from sklearn.model_selection import KFold
kf = KFold(n_splits=5, random_state=None, shuffle=False)
kf_x = np.random.rand(1000,2)
kf_y = np.random.rand(1000,1)
for train_index, test_index in kf.split(kf_x):
    print('TRAIN: ', train_index, 'TEST: ', test_index)
    X_train = kf_x[train_index]
    Y_train = kf_x[train_index]
    X_test = kf_y[test_index]
    Y_test = kf_y[test_index]
    print('train_x: ', X_train, 'test_x: ', X_test)
    print('train_y: ', Y_train, 'train_y: ', Y_train)

from sklearn import tree
from sklearn import f1_score








