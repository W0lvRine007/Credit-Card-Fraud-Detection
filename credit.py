#!/usr/bin/env python
# coding: utf-8

# In[1]:


import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


credit_card_data = pd.read_csv(
    'D:\Development\Machine_Learning\Fraud_Credit_Card_Detection\creditcard.csv')


# In[3]:


credit_card_data.head()


# In[4]:


credit_card_data.tail()


# In[5]:


credit_card_data.info()


# In[6]:


credit_card_data.isnull().sum()


# In[7]:


credit_card_data['Class'].value_counts()

# this data set is highly unbalanced0-> Normal Transaction1-> Fraudulent Transaction
# In[8]:


legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[9]:


print(legit.shape)
print(fraud.shape)


# In[10]:


legit.Amount.describe()


# In[11]:


fraud.Amount.describe()


# In[12]:


credit_card_data.groupby('Class').mean()

# Under-SamplingNumber of Fraudulent Translation-->492
# In[13]:


legit_sample = legit.sample(n=492)


# In[14]:


new_dataset = pd.concat([legit_sample, fraud], axis=0)


# In[15]:


new_dataset.head()


# In[16]:


new_dataset.tail()


# In[17]:


new_dataset['Class'].value_counts()


# In[18]:


new_dataset.groupby('Class').mean()

# splittin the data into Targets & Features
# In[19]:


X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']


# In[20]:


print(X)


# In[21]:


print(Y)

# Split data into Training data & Testing Data
# In[22]:


X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[23]:


print(X.shape, X_train.shape, X_test.shape)

# Model TrainingLogistic Regression
# In[24]:


model = LogisticRegression()


# In[25]:


# training the Logistic Regression Model with Training Data
m_learner = model.fit(X_train, Y_train)

# Model AccuracyAccuracy Score
# In[26]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[27]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[28]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[29]:


print('Accuracy score on Test Data : ', test_data_accuracy)


# In[300]:


# In[300]:


# In[300]:
joblib.dump(m_learner, "credit.pkl")
