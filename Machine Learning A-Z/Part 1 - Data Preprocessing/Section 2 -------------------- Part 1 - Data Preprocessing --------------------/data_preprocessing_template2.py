#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 15:30:46 2017

@author: tgupta2
"""
#data preprocessing
#importing the libraries

import numpy as np #contains mathematical tools
import matplotlib.pyplot as plt#used to plot charts
import pandas as pd #import and manage dataset


#importing the dataset
dataset = pd.read_csv('Data.csv')

#index starts at 0 
#need to create matrix of 3 independent variables
#: means all the lines, : columns --> :-1, means all the columns except the last
X = dataset.iloc[:,:-1].values
                
# : means all the rows, 3 means the third
Y = dataset.iloc[:,3].values

from sklearn.preprocessing import Imputer
#create an object
#strategy is the way you need to replace the missing valyes
#axis = 0--> means of columns
#axis =1--> nean of row
imputer = Imputer(missing_values= 'NaN', strategy='mean', axis = 0)
#we need to mention the columns where we want to fit the imputer, X taking all the lines, and 2 and 3 columns
imputer = imputer.fit(X[:,1:3])                
X[:,1:3] = imputer.transform(X[:,1:3])


#this class encodes labels
from sklearn.preprocessing import LabelEncoder
labelEncoder_x = LabelEncoder()
#this takes the first column, this will return the first encoded of first column
X[:,0]= labelEncoder_x.fit_transform(X[:,0])

from sklearn.preprocessing import OneHotEncoder
#column number of the categorial data
onehotencoder = OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()


labelEncoder_y = LabelEncoder()
Y= labelEncoder_y.fit_transform(Y)


#splitting data into training and testing set
from sklearn.cross_validation import train_test_split
#test size is size of the testing %
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, train_size = 0.8, random_state = 42)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test= sc_X.transform(X_test)

