#!/usr/bin/env python
# coding: utf-8

# In[1422]:


import pandas as pd
import numpy as np
from pandas import DataFrame, Series
import matplotlib.pyplot as plt


# # Loading datasets

# In[1423]:


dfRW = pd.read_csv('/home/sayan/Downloads/winequality-red.csv', delimiter = ';')
dfRW = pd.read_csv('/home/sayan/Downloads/winequality-white.csv', delimiter = ';')


# # Making choice

# In[1424]:


print("For analysis on Red wine press 'r' and for White wine press 'w'")
i = True
while i is True: 
    Q = input()
    if Q == 'r':
        C = dfRW
        i = False
        print('Analysing for Red wine')
    elif Q == 'w':
        C = dfWW
        i = False
        print('Analysing for White wine')
    else:
        print("Try again")


# In[1425]:


x = C.iloc[:, 0:11]


# In[1426]:


y1 = C.iloc[: , 11:12]


# In[1427]:


#splitting parmeter
l = int((x.shape[0]-(x.shape[0])%5)/5)
l


# In[1428]:


X1 = x.iloc[0:-l, :]


# In[1429]:


#MeanNormalising of trainset
X1 = (X1 - X1.mean())/X1.std()
X1.shape


# In[1430]:


ones = np.ones([X1.shape[0],1])
ones
ones.size


# In[1431]:


# Adding bias term
X = np.concatenate([ones,X1],1)
X.shape


# In[1432]:


y = y1.iloc[0:-l, :]
y.shape


# In[1433]:


theta = np.random.randn(12,1)
#theta = np.zeros(12).T
theta.shape


# # Cost function

# In[1434]:


def cost_func(theta,X,y):
    
    
    m = len(y)
    h = X.dot(theta)
    cost = (1/(2*m))*np.sum(np.square(h - y))
    return cost


# # Gradient descent

# In[1435]:


def grad_desc(X,y,theta,alpha,iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations,12))
    
    for t in range(iterations):
        h = np.dot(X,theta)
        #theta = theta - (1/m)*alpha*( X.T.dot((h - y)))
        theta = theta - (1/m)*alpha* np.dot(X.T,(h-y))
        theta_history[t,:] = theta.T
        cost_history[t] = cost_func(theta,X,y)
        
    return theta, cost_history, theta_history, iterations


# In[1436]:


theta, cost_history, theta_history, iters = grad_desc(X,y,theta,alpha = 0.57, iterations = 1000)


# In[1437]:


theta


# In[1438]:


cost_history


# In[1439]:


theta_history


# # Plotting the Graph

# In[1440]:


#plot
fig, ax = plt.subplots()
ax.plot(np.arange(iters), cost_history, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')


# # Cost

# In[1441]:


print('This is the optimized cost')


# In[1442]:


cost_history[-1]


# # Training test set

# In[1443]:


#testset

x_t = X1.iloc[0:(len(X1)-l), :]
x_t.shape


# In[1444]:


#MeanNormalising of testset
x_t = (x_t - x_t.mean())/x_t.std()
x_t.shape


# In[1445]:


# Adding bias term
x_test = np.concatenate([np.ones([x_t.shape[0],1]),x_t],1)
x_test.shape


# In[1446]:


y_test = y.iloc[0:(len(y)-l), :]
y_test.shape


# In[1447]:


y_pred = x_test.dot(theta)
y_pred


# In[1448]:


#Only for verfication purpose and not for evaluation
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
r2


# In[ ]:




