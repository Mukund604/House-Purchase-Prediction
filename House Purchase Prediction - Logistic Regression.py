#!/usr/bin/env python
# coding: utf-8

# # House predicition using Logisitc Regression
# 

# In[1]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# In[2]:


df = pd.read_csv('Social_Network_Ads.csv')


# In[3]:


df.head()


# In[4]:


df = df.iloc[:, 1:]


# In[5]:


df.head()


# In[6]:


genderDummies = pd.get_dummies(df.Gender).astype(int)
df = pd.concat([df,genderDummies], axis='columns')
df


# In[7]:


df = df.drop(['Gender', 'Female'], axis='columns')


# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score, classification_report


# In[9]:


y = df['Purchased']
X = df.drop(['Purchased'], axis='columns')
y


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns = X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns = X_test.columns)
X_test_scaled


# In[11]:


clf = LogisticRegression()
clf.fit(X_train_scaled,y_train)
y_pred = clf.predict(X_test_scaled)


# In[12]:


accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)


# In[13]:


print(report)


# In[14]:


np.round(X_train.describe(), 1)


# In[15]:


np.round(X_train_scaled.describe(), 1)


# # Here is why scaling the data is important.

# In[16]:


lr = LogisticRegression()
lr_scaled = LogisticRegression()

lr.fit(X_train, y_train)
lr_scaled.fit(X_train_scaled, y_train)


# In[17]:


y_pred = lr.predict(X_test)
y_pred_scaled = lr_scaled.predict(X_test_scaled)


# In[18]:


print("Accuracy before Scaling the data : ", accuracy_score(y_test,y_pred))
print("Accuracy after Scaling the data: ", accuracy_score(y_test, y_pred_scaled))

