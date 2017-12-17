# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 12:12:50 2017

@author: Alejandro
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb

dataset2 = pd.read_csv('ex1data2.txt',header = None, 
                       names=['Size','Bedrooms','Price'])
X = dataset2.iloc[:,:-1].values
y = dataset2.iloc[:,-1:].values


# Normalize data
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Add bias therm
X = np.append(arr=np.ones((len(X),1)).astype(int),values=X,axis=1)

def computeCost(X,y,theta):
    m = len(y)
    J = np.sum(np.power((X.dot(theta)-y),2))
    J = J/(2*m)
    return J

def gradientDescent(X,y,theta,alpha,num_iters):
    m = len(y)
    cost = np.zeros((num_iters,1))
    #pdb.set_trace()
    X = np.matrix(X)
    y = np.matrix(y)
    theta = np.matrix(theta)
    for iter in range(0,num_iters):
        theta = theta-alpha/m*X.T*((X*theta-y))
        cost[iter,0] = computeCost(X,y,theta)
        #print(theta)
        #print(cost)
        
    return theta, cost

# Initialize Theta

iters = 50
alpha = np.array((0.01,.03,.1,.3,1))
theta = np.matrix(np.zeros((3,len(alpha))))
cost2 = np.matrix(np.zeros((iters,len(alpha))))

X = np.matrix(X)
y = np.matrix(y)
theta = np.matrix(np.zeros((3,1)))

#alpha = 1
for i in range(0,len(alpha)):
    theta[:,i], cost2[:,i] = gradientDescent(X, y, theta[:,i], alpha[i], iters)
    
theta, cost2 = gradientDescent(X, y, theta, alpha[0], iters)

# get the cost (error) of the model
computeCost(X, y, g2)

# Plot the results

x_plot = np.linspace(1,50,50)
plt.plot(x_plot,np.array(cost2[:,5]))
plt.title('Cost Function over Num Iters')



