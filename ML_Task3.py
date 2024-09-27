#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("Dataset .csv")

df.head()


# **Data Preprocessing and Splitting**

# In[3]:


# removing features that will inhibit model training
df.drop('Restaurant ID', axis=1, inplace=True)
df.drop('Country Code', axis=1, inplace=True)
df.drop('City', axis=1, inplace=True)
df.drop('Address', axis=1, inplace=True)
df.drop('Locality', axis=1, inplace=True)
df.drop('Locality Verbose', axis=1, inplace=True)
df.drop('Longitude', axis=1, inplace=True)
df.drop('Latitude', axis=1, inplace=True)
df.drop('Currency', axis=1, inplace=True)
df.drop('Has Table booking', axis=1, inplace=True)
df.drop('Has Online delivery', axis=1, inplace=True)
df.drop('Is delivering now', axis=1, inplace=True)
df.drop('Switch to order menu', axis=1, inplace=True)
df.drop('Price range', axis=1, inplace=True)
df.drop('Aggregate rating', axis=1, inplace=True)
df.drop('Rating color', axis=1, inplace=True)
df.drop('Rating text', axis=1, inplace=True)
df.drop('Votes', axis=1, inplace=True)


# In[4]:


#handle missing values
df.isna().sum()


# In[5]:


df.dropna(inplace=True)
df.shape


# In[6]:


df.describe(include="all")


# In[7]:


from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df['Restaurant Name'] = label_encoder.fit_transform(df['Restaurant Name'])
df['Cuisines'] = label_encoder.fit_transform(df['Cuisines'])
df


# In[8]:


x = df.drop('Cuisines',axis=1)
y = df['Cuisines']


# In[9]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x=scaler.fit_transform(x)


# In[10]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=15)


# **Logistic Regression Model**

# In[11]:


from sklearn.linear_model import LogisticRegression

classifier_logreg = LogisticRegression(multi_class="multinomial",solver ="newton-cg")
classifier_logreg.fit(x_train, y_train)
logreg_pred = classifier_logreg.predict(x_test)


# In[12]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
accuracy = accuracy_score(y_test, logreg_pred)
print(f"Accuracy: {accuracy:.2f}")

# Precision, recall, F1-score
precision = precision_score(y_test, logreg_pred, average='micro')
recall = recall_score(y_test, logreg_pred, average='micro')
f1 = f1_score(y_test, logreg_pred, average='micro')
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")


# **Random Forest Model**

# In[13]:


from sklearn.ensemble import RandomForestClassifier
model_rfc = RandomForestClassifier(n_estimators=100, random_state=42)

model_rfc.fit(x_train, y_train)

rfc_pred = model_rfc.predict(x_test)


# In[14]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

accuracy = accuracy_score(y_test, rfc_pred)
print(f"Accuracy: {accuracy:.2f}")

# Precision, recall, F1-score
precision = precision_score(y_test, rfc_pred, average='micro')
recall = recall_score(y_test, rfc_pred, average='micro')
f1 = f1_score(y_test, rfc_pred, average='micro')
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")


# **Conclusion:**
# 
# On comparison, we can conclude that
# Random Forest performs better on our model than logistic regression.
# 
# Despite repeatedly trying my best on preprocessing and model selection, model performance could not be elevated beyond the current accuracy score.
# 
# This might be because of some underlying biases either in the model training or the dataset itself.

# In[ ]:




