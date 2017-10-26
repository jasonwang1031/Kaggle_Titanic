
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


# In[7]:


train = pd.read_csv("C:/Users/ywang17/Desktop/kaggle/train.csv")


# In[10]:


train.describe(include="all")


# In[23]:


cleaned_train = train.drop(['Cabin'], axis=1)
cleaned_train = cleaned_train.drop(['Ticket'], axis=1)
cleaned_train = cleaned_train.dropna()
cleaned_train.describe(include="all")


# In[33]:


dsex = pd.get_dummies(cleaned_train['Sex'])
cleaned_train = pd.concat([cleaned_train,dsex],axis =1)
cleaned_train.head()


# In[35]:


feature_cols = ['Pclass','male','Age','Parch','Fare']


# In[36]:


X.shape


# In[37]:


y = cleaned_train.Survived


# In[38]:


y.shape


# In[39]:


X = cleaned_train.loc[:,feature_cols]


# In[40]:


from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X,y)


# In[32]:


cleaned_train.head()


# In[92]:


test = pd.read_csv("C:/Users/ywang17/Desktop/kaggle/test.csv")
tsex = pd.get_dummies(test['Sex'])
test = pd.concat([test,tsex],axis =1)


# In[94]:


test.fillna(test.mean(),inplace = True)
test.describe()


# In[95]:


X_test = test.loc[:,feature_cols]
X_test.shape


# In[96]:


new_pred_class = logreg.predict(X_test)


# In[98]:


pd.DataFrame({'PassengerId':test.PassengerId, 'Survived':new_pred_class}).set_index('PassengerId').to_csv("C:/Users/ywang17/Desktop/kaggle/sub.csv")

