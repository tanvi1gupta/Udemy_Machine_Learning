#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 14:21:53 2017

@author: tgupta2

people registered for the card [Gender, age, income] 
output is spending score [higher the spending score, more the person spends]
this is clustering problem, because we need to find the people with higher spending score
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the data set using pandas
dataset = pd.read_csv('Mall_Customers.csv')
#since there are only 2 parameters that drive the clustering, keeping only those 2 variables
X = dataset.iloc[:, [3, 4]].values
                
from sklearn.cluster import KMeans
#initialize the array to find the number of clusters (value of k)
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init= 'k-means++', max_iter=300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

#plotting the wcss values    
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#number of clusters should be 5
kmeans = KMeans(n_clusters= 5, init= 'k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()