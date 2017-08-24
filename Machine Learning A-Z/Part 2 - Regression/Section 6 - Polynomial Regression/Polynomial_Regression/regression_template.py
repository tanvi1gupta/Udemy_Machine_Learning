#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 12:09:28 2017

@author: tgupta2
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#fetching data from csv files
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values
                
#split data into train and test
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

"""from sklearn.linear_model import LinearRegression
linear_reg= LinearRegression()
linear_reg.fit(X, y)
y_pred = linear_reg.predict(X)"""

 #fitting the polynomial regression model 



# Predicting a new result with Polynomial Regression
y_pred = regressor.predict(6.5)


# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Poly Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results for higher resolution and smoother curve
X_grid = np.arrange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Poly Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
