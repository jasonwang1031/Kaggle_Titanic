#Model Three: get familiar with random forest
# coding: utf-8

# In[79]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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


#Using Random Forest to train the data
feature_cols=['SibSp', 'Parch','Fare','female','Child','Class1','Class2','Class3']
x = train.loc[:,feature_cols]
y = train["Survived"]

xPre = test.loc[:,feature_cols]
# Divided the train dataset to train and test
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.7, random_state=1)

randomForest = RandomForestClassifier(n_estimators=300)
randomForest.fit(xTrain, yTrain)

i = float(100 * randomForest.score(xTrain, yTrain))

print("The accuracy of RandomForest is",i,"%")

T_Pred = randomForest.predict(xTest)
score =accuracy_score(T_Pred, y_Train)

print("The accuracy of RandomForest is",score,'%')
yPre = randomForest.predict(xPre)
pd.DataFrame({'PassengerId':test.PassengerId, 'Survived':yPre}).set_index('PassengerId').to_csv("submission.csv")


