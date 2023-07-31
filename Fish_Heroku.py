#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[2]:


df = pd.read_csv("Fish.csv")


# In[3]:


print(df.head())  # Display the first few rows of the dataset
print(df.info())  # Check for missing values and data types


# In[4]:


# Preprocess the data
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])


# In[5]:


# Split the data into features (X) and target (y)
X = df.drop('Species', axis=1)
y = df['Species']


# In[6]:


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)


# In[8]:


# Make predictions on the test set
y_pred = clf.predict(X_test)


# In[9]:


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)


# In[10]:


import joblib


# In[11]:


# Save the trained model
joblib.dump(clf, 'fish_species_model.pkl')


# In[ ]:




