# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 14:23:11 2022

@author: jlusk
"""

##### Import CSV data and set NA values

import scipy.cluster.hierarchy as sch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor

dfTest = pd.read_csv("Phishing_Legitimate_test_student.csv", na_values=['',' ','n/a'])
dfTrain = pd.read_csv("Phishing_Legitimate_train_missing_data.csv", na_values=['',' ','n/a'])

print(dfTrain.dtypes)

print(dfTrain.isnull().sum())
print(dfTrain.isnull().any(axis=0))# Show sum of columns with N/A
print(dfTrain.isnull().any(axis=1))# Show sum of rows with N/A
trainRowsWithNa = dfTrain[ dfTrain.isnull().any(axis=1) ]
print(trainRowsWithNa)

print(dfTest.isnull().sum())
print(dfTest.isnull().any(axis=0))
print(dfTest.isnull().any(axis=1))

rowsToDrop = dfTrain[ dfTrain.isnull().sum(axis=1) > 1 ].index
print(rowsToDrop.shape)
print(rowsToDrop) # Index of rows that were dropped because >1 N/A value

# Drop rows with > 1 N/A's
dfTrain.drop(rowsToDrop, inplace=True)
print(dfTrain.shape)

imputer = KNNImputer(n_neighbors=10) # Choose imputer based on 10 nearest neighbors
# Create new dataframe
dfTrain2 = pd.DataFrame(imputer.fit_transform(dfTrain),columns = dfTrain.columns)
print(dfTrain2)
print(dfTrain2.shape)

# Example of one heavy outlier for 'UrlLength'
plt.figure()
dfTrain2[['UrlLength']].plot.box()
plt.show()

# Drop big outlier row
# Should drop row with 'UrlLengh' > 500 because 1 data point > 1750 will skew data heavily
rows_to_drop=dfTrain2[dfTrain2['UrlLength']>500].index
print(rows_to_drop)
# Drop rows where 'UrlLength' > 500
dfTrain2.drop(rows_to_drop,inplace=True)
print(dfTrain2.shape)

# Plog again after outlier removal
plt.figure()
dfTrain2[['UrlLength']].plot.box()
plt.show()

# Example of one heavy outlier for 'NumNumericChars'
plt.figure()
dfTrain2[['NumNumericChars']].plot.box()
plt.show()

# Drop big outlier row
# Should drop row with 'NumNumericChars' > 100 because 2 data points will skew data
rows_to_drop=dfTrain2[dfTrain2['NumNumericChars']>100].index
print(rows_to_drop)
# Drop rows where 'UrlLength' > 100
dfTrain2.drop(rows_to_drop,inplace=True)
print(dfTrain2.shape)

# Plog again after outlier removal
plt.figure()
dfTrain2[['NumNumericChars']].plot.box()
plt.show()

# Example of one heavy outlier for 'NumDash'
plt.figure()
dfTrain2[['NumDash']].plot.box()
plt.show()

# Drop big outlier row
# Should drop row with 'NumDash' > 20 because a few data points will skew data
rows_to_drop=dfTrain2[dfTrain2['NumDash']>20].index
print(rows_to_drop)
# Drop rows where 'NumDash' > 20
dfTrain2.drop(rows_to_drop,inplace=True)
print(dfTrain2.shape)

# Plot again after outlier removal
plt.figure()
dfTrain2[['NumDash']].plot.box()
plt.show()

dfTrain2Numerical = dfTrain2[['NumNumericChars','NumDots','SubdomainLevel','PathLevel','NumDash','NumDashInHostname',
          'NumUnderscore','NumPercent','NumQueryComponents','NumAmpersand','NumHash','HostnameLength','PathLength','QueryLength','NumSensitiveWords']]
plt.figure()
dfTrain2Numerical.plot.box()
plt.show()

from sklearn.neighbors import LocalOutlierFactor
clf= LocalOutlierFactor(n_neighbors=20)
X=dfTrain2Numerical.to_numpy()
print(dfTrain2.shape)
# Find labels of outliers
outlier_label=clf.fit_predict(X)
print(clf.negative_outlier_factor_)
print(clf.offset_)
print(outlier_label)

plt.boxplot(clf.negative_outlier_factor_)

# Identify index of rows to drop
# Use threshold of -1.30, as -1.5 may leave too many outliers in

rows_to_drop= dfTrain2.iloc[clf.negative_outlier_factor_ < -1.30].index

# Drop rows with negative_outlier_factor < -1.30
print(rows_to_drop) # Index of rows that will be dropped from dfTrain2
dfTrain2.drop(rows_to_drop,inplace=True)
print(dfTrain2.shape)

dfTrain2Numerical = dfTrain2[['NumNumericChars','NumDots','SubdomainLevel','PathLevel','NumDash','NumDashInHostname','NumUnderscore','NumPercent','NumQueryComponents','NumAmpersand','NumHash','HostnameLength','PathLength','QueryLength','NumSensitiveWords']]
plt.figure()
dfTrain2Numerical.plot.box()
plt.show()

from sklearn.preprocessing import StandardScaler

dfTrain2Numerical = dfTrain2[['NumNumericChars','NumDots','SubdomainLevel','PathLevel','NumDash','NumDashInHostname','NumUnderscore','NumPercent','NumQueryComponents','NumAmpersand','NumHash','HostnameLength','PathLength','QueryLength','NumSensitiveWords']]
X = dfTrain2Numerical.to_numpy()
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

dfTrain2Numerical.is_copy = False

# Error is happening here when adding the new rows to the dfTrain2Numerical dataframe. Tried top solutions from stack overflow and nothing is working. Does not effect data.
dfTrain2Numerical[['NumNumericChars_Standardized','NumDots_Standardized','SubdomainLevel_Standardized','PathLevel_Standardized','NumDash_Standardized','NumDashInHostname_Standardized','NumUnderscore_Standardized','NumPercent_Standardized','NumQueryComponents_Standardized','NumAmpersand_Standardized','NumHash_Standardized','HostnameLength_Standardized','PathLength_Standardized','QueryLength_Standardized','NumSensitiveWords_Standardized']]=X

sns.boxplot(data=dfTrain2Numerical[['NumNumericChars','NumDots','SubdomainLevel','PathLevel','NumDash','NumDashInHostname','NumUnderscore','NumPercent','NumQueryComponents','NumAmpersand','NumHash','HostnameLength','PathLength','QueryLength','NumSensitiveWords']])













