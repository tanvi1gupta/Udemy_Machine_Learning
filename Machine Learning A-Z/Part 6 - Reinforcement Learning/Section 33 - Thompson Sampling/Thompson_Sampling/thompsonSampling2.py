#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 17:28:01 2017

@author: tgupta2

this is thompson sampling problem
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing Thompson Sampling
#at each round, we consider 2 values for each ad i:
# N_0[i]: the number of times i got reward 0 upto round n 
# N_1[i]: the number of times i got reward 1 upto round n 
#for each ad i , random draw the number from beta( N_0[i]+1 and N_1[i]+1) = theta
#select the ad with the highest theta

import random
N = 10000
d = 10
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0

for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        #this will give a random variate
        random_theta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_theta > max_random :
            max_random = random_theta
            ad = i
    #append the selected ad in the ads_selected array
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1: 
        numbers_of_rewards_1[ad]= numbers_of_rewards_1[ad]+1
    else:
        numbers_of_rewards_0[ad]= numbers_of_rewards_0[ad]+1
    total_reward = total_reward + reward
    
# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()