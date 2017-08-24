#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 22:18:32 2017

@author: tgupta2
"""

# Importing the libraries
import pandas as pd
import numpy as np

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
                
#importing labelencoder and onehotencoder for the categorical variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder = LabelEncoder()
X[:,3] = labelEncoder.fit_transform(X[:,3])

onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

#removing one dummy variable to avoid dummy trap
X = X[:, 1:]

#splitting the data in test and train
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#using linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#BackWard Elimination
import statsmodels.formula.api as sm
#need to add column of 1s to the dataset 
#this will add the column to the end
#X = np.append(arr = X, values = np.ones((50,1)).astype(int), axis=1)
X = np.append(arr = np.ones((50,1)).astype(int), values = X , axis=1)

X_opt = X[:,[0,1,2,3,4,5]]
#fit the full model to the predictor
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
regressor_ols.summary()

#since x2 has highest p value, we will remove it
X_opt = X[:,[0,1,3,4,5]]
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
regressor_ols.summary()


X_opt = X[:,[0,3,4,5]]
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
regressor_ols.summary()

X_opt = X[:,[0,3,5]]
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
regressor_ols.summary()

X_opt = X[:,[0,3]]
regressor_ols = sm.OLS(endog=y, exog=X_opt).fit()
regressor_ols.summary()

