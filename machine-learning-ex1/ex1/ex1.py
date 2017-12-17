# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 21:46:32 2017

@author: Alejandro
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('ex1data1.txt',header = None,
                      names = ['Population', 'Profit'])
x = dataset.iloc[:,0:1].values
y = dataset.iloc[:,-1:].values

# Plot values
#plt.scatter(x,y)


X = np.append(arr=np.ones((len(x),1)).astype(int),values=x,axis=1)
theta = np.zeros((2,1))

iterations = 1500
alpha = 0.01

def computeCost(X,y,theta):
    m = len(y)
    J=0
    J = J+np.sum(np.power((X.dot(theta)-y),2))
    J = J/(2*m)
    return J

def gradientDescent(X,y,theta,alpha,num_iters):
    m = len(y)
    J_history = np.zeros((num_iters,1))
    for iter in range(0,num_iters):
        #print('Theta before', theta)
        theta = theta-alpha/m*np.transpose(X).dot((X.dot(theta)-y))
        J_history[iter] = computeCost(X,y,theta)
        #print('Theta after',theta)
        
    return theta

theta = np.zeros((2,1))
iterations = 1500
theta= gradientDescent(X,y,theta,alpha,iterations)
print('Theta found by gradient descent: {} {}'.format(theta[0],theta[1]))
        
       

# Plot the results
x_plot = np.linspace(dataset.Population.min(), dataset.Population.max(), 100)  
f = theta[0, 0] + (theta[1, 0] * x_plot)

fig, ax = plt.subplots(figsize=(12,8))  
ax.plot(x_plot, f, 'r', label='Prediction')  
ax.scatter(dataset.Population, dataset.Profit, label='Traning Data')  
ax.legend(loc=2)  
ax.set_xlabel('Population')  
ax.set_ylabel('Profit')  
ax.set_title('Predicted Profit vs. Population Size')  