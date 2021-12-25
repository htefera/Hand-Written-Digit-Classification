
# coding: utf-8

# Logistic Regression
# This notebook shows how Binary Logistic Regression  is preformed using python. 
# First import all the necessary libraries
# The aim of the logistic regression is to find the "best" line to predict the output variable of Y from X based on 
# the following logistic function  Y = σ(X ∗W + b)
# 

# In[153]:


get_ipython().magic('matplotlib inline')

import matplotlib.pyplot as plt
import numpy as np
from ReadData import get_first_data, get_MNIST_data

np.random.seed(12345)


# In[170]:


# The logistic function.
# Should return a vector/matrix with the same dimensions as x.
from math import sin, exp
def sigmoid(x):
  return (1 / (1 + np.exp(-x)))


# In[195]:


# Normalize each column ("attribute") of a given matrix.
# The normalization is done so that each column of the output matrix
# have 0 mean and unit standard diviation.
def featureNormalization(X):
    mean = np.mean(X,0)
    std = np.std(X,0)
    X = (X-mean)/std
    return X


# In[209]:


# The cost function, the price we are paying for beeing wrong.

def cost(X, W, b, Y):
    Yhat = sigmoid(np.dot(X,W) + b) 
    cost = (-1.0/np.shape(X)[0]) * np.sum(Y * np.log(Yhat) + (1-Y) * (np.log(1-Yhat)))
    if np.isnan(cost):
        return np.inf
    return cost
   


# In[210]:


# The derivative of the cost function at the
# Should return a tuple of (d cost)/(d W) and (d cost)/(d b).
# This function should work both when W and b are vectors or matrices.
def d_cost(X, W, b, Y):
    Yhat = sigmoid(np.dot(X,W) + b) 
    dW = (np.dot(-X.T,(Y -Yhat)))/(1.0*np.shape(X)[0])
    db = (np.sum(-(Y-Yhat)))/(1.0*np.shape(X)[0])
    return (dW, db)


# In[211]:


# Preform gradient descent given the input data X and
# the expected output Y.
# alpha is the learning rate.
# Returns the value of W and b and a list of the cost for each iteration.
def gradient_descent(X, Y, alpha, iterations=5000):
    cost_over_time = []

    W = np.zeros((X.shape[1], Y.shape[1]))
    b = np.zeros((1, Y.shape[1]))

    for i in range(iterations):

        cost_over_time.append(cost(X, W, b, Y))

        dW, db = d_cost(X, W, b, Y)
        W = W - alpha * dW
        b = b - alpha * db
        

    return (W, b, cost_over_time)


# In[212]:


# Classify each output depending on if it is smaller or larger than 0.5

def classify(X, W, b):
 X == sigmoid(np.dot(X,W)+b)
 return [1 if x > 0.5 else 0 for x in X]
 


# In[213]:


def classify_zeros():

    X, Y = get_first_data()
    X = featureNormalization(X)

    # Split dataset into a training and testing set
    testing = np.random.uniform(low=0, high=1, size=X.shape[0]) > 0.1
    training = np.logical_not(testing)

    X_test = X[testing]
    Y_test = Y[testing]
    X = X[training]
    Y = Y[training]

    alpha = 0.1

    W, b, cost_over_time = gradient_descent(X, Y, alpha)

    # Plot the achived result
    f, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(cost_over_time)
    ax1.set_title("Cost over time")

    X_l = np.asarray([np.linspace(np.min(X), np.max(X), 1000)]).T
    Yhat = sigmoid(np.dot(X_l,W) + b)
    ax2.plot(X, Y, 'bo' , X_l, Yhat, 'r-')
    ax2.set_title("The fitted logistic function")

    plt.show()
   #percentage of correctly classified examples 
    
    y = classify(X_test ,W,b) 
    result= (np.sum(y == Y_test)/(1.0*np.shape(Y_test)[0]))
    print("Model Accuracy=",round(result,1),'%')
    
    
    
   


# In[214]:



def learn_digits():

    # Train model
    X, Y = get_MNIST_data()
    W, b, cost_over_time = gradient_descent(X, Y, 0.1, 2500)

    # Create matrix to plot
    mp = [np.reshape(W[:,x], (28,28)) for x in range(10)]


    m1 = np.concatenate(mp[ :5], 1)
    m2 = np.concatenate(mp[5:], 1)
    m =  np.concatenate((m1, m2), 0)


    # Plot the learnt weights in a grid
    plt.matshow(m)
    plt.plot([0, m.shape[1]], [m.shape[0]/2, m.shape[0]/2], 'k-')

    for i in range(1,5):
        plt.plot([28*i, 28*i], [0, m.shape[0]-1], 'k-')

    plt.axis([0, m.shape[1]-1, m.shape[0]-1,0])

    plt.show()


# In[218]:


classify_zeros()


# In[13]:


learn_digits()

