#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[4]:


df = pd.read_csv('Churn.csv')


# In[5]:


X = pd.get_dummies(df.drop(['Churn', 'Customer ID'], axis=1))
y = df['Churn'].apply(lambda x: 1 if x=='Yes' else 0)


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)


# In[8]:


y_train.head()


# In[7]:


pip install tensorflow


# In[9]:


from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score


# In[10]:


from keras.models import Sequential
model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=len(X_train.columns)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')


# In[11]:


model.fit(X_train, y_train, epochs=200, batch_size=32)


# In[12]:


y_hat = model.predict(X_test)
y_hat = [0 if val < 0.5 else 1 for val in y_hat]


# In[13]:


accuracy_score(y_test, y_hat)


# In[14]:


model.save('tfmodel')

