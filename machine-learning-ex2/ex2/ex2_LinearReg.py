# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 19:26:56 2017

@author: Alejandro
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('ex2data1.txt',
                      header = None,names = ['Exam 1','Exam 2','Admitted'])
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1:].values

# Plot results

pos = dataset[dataset['Admitted'].isin([1])]
neg = dataset[dataset['Admitted'].isin([0])]
    
plt.scatter(pos['Exam 1'], pos['Exam 2'],c='b',marker = 'o',label='Admitted')
plt.scatter(neg['Exam 1'], neg['Exam 2'],c='r',marker = 'x',label='Not Admitted')
plt.legend()
plt.xlabel('Exam 1 Score')
plt.ylabel('Exam 2 Score')


def sigmoid(z):
    return 1/(1+np.exp(-z))

# Plot Sigmoid function
x_nums = np.arange(-10,10,step=1)
#plt.plot(x_nums,sigmoid(x_nums))

# Definition of the cost function
def cost(theta,X,y):
    m = X.shape[0]
    n = X.shape[1]
    theta = np.matrix(theta).reshape(n,1)
    X = np.matrix(X)
    y = np.matrix(y)
    z = np.matrix(X)*np.matrix(theta)
    m = len(y)
    first_term = -np.multiply(y,np.log(sigmoid(z)))    
    second_term = -np.multiply((1-y),np.log(1-sigmoid(z)))    
    J = 1/m*np.sum(first_term + second_term)
    return J

def gradient (theta,X,y):
    m = X.shape[0]
    n = X.shape[1]
    theta = np.matrix(theta).reshape(n,1)
    X=np.matrix(X)
    y=np.matrix(y)
    z = np.matrix(X)*np.matrix(theta)
    grad = X.T*(sigmoid(z)-y)/m
    return grad
    
X = np.append(arr=np.ones((len(X),1)).astype(int),values=X,axis=1)
m = np.size(X,0)
n = np.size(X,1)
theta = np.zeros((n))

import scipy.optimize as opt
opt_result = opt.fmin_tnc(func = cost, x0 =theta,fprime = gradient,args =(X,y))

def predict(theta,X):
    n = X.shape[1]
    X = np.matrix(X)
    theta = np.matrix(theta).reshape(n,1)
    probability = sigmoid(X*theta)
    return[1 if x>=0.5 else 0 for x in probability]

theta_min = opt_result[0]
y_pred = predict(theta_min,X)
correct = [1 if ((a==1 and b ==1) or (a==0 and b==0)) else 0 for (a,b) in zip(y_pred,y)]
accuracy = (sum(map(int,correct)))/len(correct)
print('acuracy = {0}%'.format(accuracy))





""" Linear Regression with regularization"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset2 = pd.read_csv('ex2data2.txt',
                      header = None,names = ['Test 1','Test 2','State'])
X = dataset2.iloc[:,:-1].values
y = dataset2.iloc[:,-1:].values
          
positive= dataset2[dataset2['State'].isin([1])]
negative= dataset2[dataset2['State'].isin([0])]
# Plot results

plt.scatter(positive['Test 1'],positive['Test 2'],s=50,c='b', marker = 'o',
            label = 'Accepted')
plt.scatter(negative['Test 1'],negative['Test 2'],s=50,c='r', marker = 'x',
            label = 'Accepted')
plt.legend()
plt.xlabel('Test 1 Score')
plt.ylabel('Test 2 Score')
plt.show()

#Sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))
# Definition of the cost function
def costReg(theta,X,y,param):
    m = X.shape[0]
    n = X.shape[1]
    theta = np.matrix(theta).reshape(n,1)
    X = np.matrix(X)
    y = np.matrix(y)
    z = np.matrix(X)*np.matrix(theta)
    m = len(y)
    first_term = -np.multiply(y,np.log(sigmoid(z)))    
    second_term = -np.multiply((1-y),np.log(1-sigmoid(z)))    
    J = 1/m*np.sum(first_term + second_term)
    J = J + param/(2*m)*np.sum(np.power(theta[1:],2))
    return J

def gradientReg (theta,X,y,param):
    m = X.shape[0]
    n = X.shape[1]
    theta = np.matrix(theta).reshape(n,1)
    X=np.matrix(X)
    y=np.matrix(y)
    z = np.matrix(X)*np.matrix(theta)
    grad = X.T*(sigmoid(z)-y)/m
    grad[1:] = grad[1:] + param/m*theta[1:]
    return grad

# Add Bias Therm
#X = np.append(arr=np.ones((len(X),1)).astype(int),values=X,axis=1)

# Polynomial Features to the dataset
from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree = 5)
X_poly = polyReg.fit_transform(X)

m = np.size(X,0)
n = np.size(X,1)
initial_theta = np.zeros(np.size(X_poly,1))
learningRate = 1

import scipy.optimize as opt
opt_result = opt.fmin_tnc(func = costReg, x0 =initial_theta,fprime = gradientReg,args =(X_poly,y,learningRate))

def predict(theta,X):
    n = X.shape[1]
    X = np.matrix(X)
    theta = np.matrix(theta).reshape(n,1)
    probability = sigmoid(X*theta)
    return[1 if x>=0.5 else 0 for x in probability]

theta_min = opt_result[0]
y_pred = predict(theta_min,X_poly)
correct = [1 if ((a==1 and b ==1) or (a==0 and b==0)) else 0 for (a,b) in zip(y_pred,y)]
accuracy = (sum(map(int,correct)))/len(correct)
print('acuracy = {0}%'.format(accuracy))


