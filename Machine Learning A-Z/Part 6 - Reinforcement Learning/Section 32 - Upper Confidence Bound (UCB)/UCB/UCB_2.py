#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 17:23:17 2017

@author: tgupta2

reinforcement learning

the data is matrix of 0s and 1s
we have different versions of the same advertisement (10 versions)
which advertisement to put on the social media
each time user connects to his account, we will randomly put an ad
if the user clicks Y --> 1 reward
else --> 0 reward

algo will look at the results at k-1 time stamp

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing UCB

import math
N = 10000
d = 10
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0

#at each round n, we consider 2 numbers d and i
# N(i,n) == number of items, ad i was selected upto this round
# R(i,n) == sum of rewards of ad i till round n 
   
   
#average reward of ad i upto n average == R(i,n)/N(i,n)
#confidence interval == average - delta; average + delta
#delta = sqrt(3/2 * log(n)/N(i,n))
#select ad i that has the max UCB = average + delta

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if numbers_of_selections[i]>0:
            average_reward = sums_of_rewards[i]/numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1)/numbers_of_selections[i])
            upper_bound = average_reward + delta_i 
        else:
            upper_bound = 1e400 #(10 to power 400)
        
        if upper_bound > max_upper_bound :
            max_upper_bound = upper_bound
            ad = i
    #append the selected ad in the ads_selected array
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad]+1
    reward = dataset.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad]+reward
    total_reward = total_reward + reward
    
# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()