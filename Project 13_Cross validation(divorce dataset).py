#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Cross Validation Classification Accuracy
# Load libraries
import pandas as pd
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")


# In[4]:


data = pd.read_csv("divorce-csv.csv")


# In[5]:


data.shape


# In[6]:


data.head


# In[7]:


data.describe


# In[8]:


print(data.groupby('Class').size())


# In[9]:


data.isnull().sum()


# In[10]:


x=data.iloc[:,0:-1]
x.shape


# In[11]:


y=data.iloc[:,-1]
y.shape


# In[17]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[21]:


lr=LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)


# In[22]:


lr.fit(x_train,y_train)


# In[23]:


y_pred = lr.predict(x_test)


# In[24]:


from sklearn.metrics import mean_squared_error


# In[25]:


print(mean_squared_error(y_test,y_pred))


# In[26]:


from sklearn.metrics import r2_score
print(r2_score(y_test,y_pred))


# In[29]:


import numpy as np
rmse_no_crossval=print(np.sqrt(mean_squared_error(y_test, y_pred)))


# CROSS VALIDATION

# In[32]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[36]:


kf = KFold(n_splits=100, random_state=1)


# In[37]:


scores= cross_val_score(lr, x_train, y_train, cv=kf, scoring='neg_mean_squared_error')
rmse_crossval = np.sqrt(-scores)
print(rmse_crossval.mean())


# In[ ]:




