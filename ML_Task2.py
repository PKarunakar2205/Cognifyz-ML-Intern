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


# In[3]:


df.columns


# In[4]:


df.info()


# In[5]:


df.dtypes


# In[6]:


df.describe()


# **visualize null values present in dataset**

# In[8]:


#create a refined dataframe
dfR = df[['Restaurant ID','Restaurant Name','Cuisines','Price range','Aggregate rating','Votes']]
dfR


# In[10]:


#handle missing values
dfR.isna().sum()


# In[11]:


dfR = dfR.dropna()
dfR.isna().sum()


# In[12]:


dfR.duplicated().sum()


# In[13]:


dfR['Restaurant Name'].duplicated().sum()


# In[14]:


dfR['Restaurant Name'].value_counts()


# In[16]:


#sorting the restaurants by name and rating
dfR = dfR.sort_values(by=['Restaurant Name','Aggregate rating'],ascending=False)
dfR.head()


# In[17]:


dfR[dfR["Restaurant Name"]=="Cafe Coffee Day"].head()


# In[18]:


#removing duplicate entries of same restaurant name
dfR = dfR.drop_duplicates('Restaurant Name',keep='first')
dfR


# In[19]:


dfR['Restaurant Name'].value_counts()


# In[20]:


dfR = dfR[dfR['Aggregate rating']>3.9]
dfR


# In[21]:


#splitting cuisines into list
dfR['Cuisines'] = dfR['Cuisines'].str.split(', ')
dfR


# In[22]:


dfR = dfR.explode('Cuisines')
dfR


# In[23]:


dfR['Cuisines'].value_counts()


# In[24]:


restoXcuisines = pd.crosstab(dfR['Restaurant Name'], dfR['Cuisines'])
restoXcuisines


# In[25]:


dfR['Restaurant Name'].sample(20, random_state=194)


# In[26]:


from sklearn.metrics import jaccard_score
print(jaccard_score(restoXcuisines.loc["Olive Bistro"].values,
                    restoXcuisines.loc["Rose Cafe"].values))


# In[27]:


from scipy.spatial.distance import pdist, squareform

jaccardDist = pdist(restoXcuisines.values, metric='jaccard')
jaccardMatrix = squareform(jaccardDist)
jaccardSim = 1 - jaccardMatrix
dfJaccard = pd.DataFrame(
    jaccardSim,
    index=restoXcuisines.index,
    columns=restoXcuisines.index)

dfJaccard


# In[28]:


dfR['Restaurant Name'].sample(20)


# **Final Recommendation System**
# 
# (use algos like RF for user input recommendation.. here it is just static rigid output)

# In[29]:


resto = 'Ooma'

sim = dfJaccard.loc[resto].sort_values(ascending=False)
sim = pd.DataFrame({'Restaurant Name': sim.index, 'simScore': sim.values})
sim = sim[(sim['Restaurant Name']!= resto) & (sim['simScore']>=0.7)].head(5)
RestoRec = pd.merge(sim,dfR[['Restaurant Name','Aggregate rating']],how='inner',on='Restaurant Name')
FinalRestoRec = RestoRec.sort_values('Aggregate rating',ascending=False).drop_duplicates('Restaurant Name',keep='first')


# In[30]:


FinalRestoRec


# In[ ]:




