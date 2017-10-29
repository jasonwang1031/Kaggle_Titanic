# Second model: Logistic Regression with base treatment on some predictors
# coding: utf-8

# In[101]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[80]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# import csv files


# In[81]:


train.drop(['Name', 'Ticket','Cabin','Embarked'], axis=1, inplace=True) 
test.drop(['Name', 'Ticket','Cabin','Embarked'], axis=1, inplace=True) 
# drop the usless variables


# In[82]:


tsex = pd.get_dummies(test['Sex'])
test = pd.concat([test,tsex],axis =1)
test.drop(['Sex'], axis = 1, inplace =True)


# In[83]:


train_sex = pd.get_dummies(train['Sex'])
train = pd.concat([train,train_sex],axis =1)
train.drop(['Sex'], axis =1, inplace = True)
# create dummy variables for sex


# In[84]:


TrainAgeMean = train['Age'].mean()
TestAgeMean = test['Age'].mean()
train['Age'].fillna(TrainAgeMean, inplace=True)
test['Age'].fillna(TestAgeMean, inplace=True)
# missing values in age


# In[85]:


def determine_child(age):
    if age <= 16: return 1
    else: return 0


# In[86]:


train['Child'] = train['Age'].apply(determine_child)
test['Child']= test['Age'].apply(determine_child)


# In[87]:


train_pclass = pd.get_dummies(train['Pclass'])
train = pd.concat([train, train_pclass], axis=1)
train.rename(columns={1:'Class1',2:'Class2',3:'Class3'}, inplace=True)
train.drop(['Pclass'],axis=1, inplace= True)
# create the dummy variable for Pclass


# In[88]:


test_pclass = pd.get_dummies(test['Pclass'])
test = pd.concat([test, test_pclass], axis=1)
test.rename(columns={1:'Class1',2:'Class2',3:'Class3'}, inplace=True)
test.drop(['Pclass'], axis=1, inplace= True)


# In[91]:


test['Fare'].fillna(test['Fare'].median(), inplace = True)
train['Fare'] = train['Fare'].astype(int)
test['Fare']=test['Fare'].astype(int)


# In[92]:


train


# In[144]:


#train the data with logistic regression
feature_cols=['SibSp', 'Parch','Fare','female','Child','Class1','Class2','Class3']
x = train.loc[:,feature_cols]
x.shape
y = train.Survived
y.shape
logreg = LogisticRegression()
logreg.fit(x,y)


# In[145]:


#classify the test dataset
x_test = test.loc[:,feature_cols]
x_test.shape
new_pred_class = logreg.predict(x_test)
pd.DataFrame({'PassengerId':test.PassengerId, 'Survived':new_pred_class}).set_index('PassengerId').to_csv("submission.csv")

