#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


di=pd.read_csv('diabetes.csv')


# In[4]:


di.head()


# In[5]:


di.info()


# In[6]:


di.describe()


# In[7]:


di.isnull().sum()


# In[8]:


di['Insulin'].value_counts()


# In[9]:


di['SkinThickness'].value_counts().head()


# In[10]:


sns.jointplot(x='Insulin',y='SkinThickness',data=di,kind='reg')


# In[11]:


di.corr()


# In[12]:


p=0
q=0
for i in range(len(di['Insulin'])):
    if di['Insulin'][i]==0 and di['SkinThickness'][i]!=0:
        p+=1
print(p)


# In[13]:


di=pd.read_csv('diabetes.csv')


# In[14]:


p=int(di['Insulin'][di['Insulin']<400].mean())


# In[15]:


di['SkinThickness'].replace(0,int(di['SkinThickness'].mean()),inplace=True)


# In[16]:


di['Insulin'].replace(0,p,inplace=True)


# In[17]:


di.head()


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


xtr, xte, ytr, yte = train_test_split(di.drop('Outcome',axis=1),di['Outcome'],test_size=0.50)


# In[20]:


ytr


# In[21]:


from sklearn.linear_model import LogisticRegression


# In[22]:


logmodel=LogisticRegression()
logmodel.fit(xtr,ytr)


# In[23]:


predictions = logmodel.predict(xte)


# In[24]:


from sklearn.metrics import confusion_matrix


# In[25]:


accuracy = confusion_matrix(yte,predictions)
accuracy


# In[26]:


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(yte,predictions)
accuracy


# In[27]:


from sklearn.ensemble import RandomForestClassifier


# In[28]:


clf=RandomForestClassifier(n_estimators=100)


# In[29]:


clf.fit(xtr,ytr)


# In[30]:


ypr=clf.predict(xte)


# In[31]:


from sklearn import metrics
print('Accuracy  :',metrics.accuracy_score(yte,ypr))


# In[36]:


sns.pairplot(hue='Outcome',kind='scatter',data=di)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




