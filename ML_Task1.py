#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report


# In[2]:


df = pd.read_csv("Dataset .csv")

df.head()


# 
# Info about the data

# In[3]:


df.columns


# In[4]:


df.info()


# In[5]:


df.dtypes


# In[6]:


df.describe()


# In[8]:


df.shape


# In[9]:


df.isnull().sum()


# In[10]:


df = df.dropna()


# In[11]:


df.shape


# In[13]:


df['Aggregate rating'].describe()


# In[14]:


df['Aggregate rating'].value_counts()


# In[15]:


import plotly.express as px
fig = px.pie(df,names ="Aggregate rating",hole = 0.3,template ="plotly_dark")
fig.show()


# In[16]:


fig = px.scatter(df,x ="Average Cost for two",y="Price range",color= "Aggregate rating",template="plotly_dark")
fig.show()


# In[17]:


fig = px.scatter(df,x ="Has Online delivery",y="Price range",color= "Aggregate rating",template="plotly_dark")
fig.show()


# In[18]:


fig = px.scatter(df,x ="City",y="Cuisines",color= "Aggregate rating",template="plotly_dark")
fig.show()


# In[19]:


from sklearn.preprocessing import LabelEncoder


#encoded_df = df.copy()

label_encoder = LabelEncoder()

columns_to_encode = ['Restaurant Name', 'City', 'Address', 'Locality', 'Locality Verbose', 'Cuisines', 'Currency', 'Rating color', 'Rating text']

# Encode categorical columns
for column in columns_to_encode:
    df[column] = label_encoder.fit_transform(df[column])

# Map 'Yes' and 'No' to numerical values for binary categorical columns
binary_columns = ['Has Table booking', 'Has Online delivery', 'Is delivering now', 'Switch to order menu']
binary_mapping = {'No': 0, 'Yes': 1}

# Encode binary categorical columns
for column in binary_columns:
    df[column] = df[column].map(binary_mapping)

# Print first few rows of encoded DataFrame
print(df.head())


# In[20]:


# drop features that inhibit model building
df = df.drop('Restaurant ID', axis=1)
df = df.drop('Restaurant Name', axis=1)
df = df.drop('Country Code', axis=1)
df = df.drop('City', axis=1)
df = df.drop('Address', axis=1)
df = df.drop('Locality', axis=1)
df = df.drop('Locality Verbose', axis=1)
df = df.drop('Longitude', axis=1)
df = df.drop('Latitude', axis=1)
df = df.drop('Cuisines', axis=1)
df = df.drop('Currency', axis=1)


# In[21]:


print(df.describe())


# In[22]:


df


# In[23]:


sns.distplot(df['Aggregate rating'])


# In[24]:


sns.scatterplot(x=df["Aggregate rating"],y=df["Votes"],hue=df["Price range"])


# In[25]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True)
plt.title("Correlation between the attributes")
plt.show()


# In[26]:


x = df.drop('Aggregate rating', axis=1)
y = df['Aggregate rating']


# **Data splitting**

# In[27]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=250)
x_train.head()
y_train.head()


# In[28]:


print("x_train: ", x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)


# **Linear Regression Model**

# In[29]:


#training by linear regression algorithm
linreg = LinearRegression()
linreg.fit(x_train,y_train)
linreg_pred=linreg.predict(x_test)


# In[30]:


#evaluating performance metrics of linear regression
linreg_mae = mean_absolute_error(y_test, linreg_pred)
linreg_mse = mean_squared_error(y_test, linreg_pred)
linreg_r2 = r2_score(y_test, linreg_pred)
print(f"Mean Absolute Error of the linear regression model is: {linreg_mae:.2f}")
print(f"Mean Squared Error of the linear regression model is: {linreg_mse:.2f}")
print(f"R2 score of the linear regression model is: {linreg_r2:.2f}")


# **Decision Tree**

# In[31]:


# training by decision tree regressor algorithm
dtree = DecisionTreeRegressor()
dtree.fit(x_train, y_train)
dtree_pred = dtree.predict(x_test)


# In[32]:


#evaluating performance metrics of decision tree
dtree_mae = mean_absolute_error(y_test, dtree_pred)
dtree_mse = mean_squared_error(y_test, dtree_pred)
dtree_r2 = r2_score(y_test, dtree_pred)
print(f"Mean Absolute Error of the decision tree model is: {dtree_mae:.2f}")
print(f"Mean Squared Error of the decision tree model is: {dtree_mse:.2f}")
print(f"R2 score of the decision tree model is: {dtree_r2:.2f}")


# **Model have 98% accuracy**
# 
# MSE of 0.05 indicates that model's predictions are very accurate & low errors.
# 
# R2 value of 0.98 suggests that model is highly effective at explaining & predicting the target variable.
# 
# Decision Tree Regressor model is performing exceptionally well on the test data.
# 
# **Analysing the factors affecting restaurant ratings**
# 
# Distribution of the target variable ("Aggregate rating") is well balanced.
# 
# Expensive restaurants (higher price range) tend to have higher ratings.

# In[ ]:




