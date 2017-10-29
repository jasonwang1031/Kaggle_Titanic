
# coding: utf-8

# In[319]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score


# In[320]:


# Import data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine_df = [train_df, test_df]


# In[321]:


#Name_length
for df in combine_df:
    df['Name_Len'] = df['Name'].apply(lambda x: len(x))


# In[322]:


#Title
title_dict = {"Mr":1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, 
              "Mme": 8, "Don": 7, "Dona":2, "Lady": 2, "Countess": 10, "Jonkheer": 10, "Sir": 7, "Capt": 7, 
              "Ms": 2}
def Find_Title(title):
    for t in title_dict:
        if t == title:
            return int(title_dict[t])


# In[323]:



train_df['Title'] = train_df['Name'].apply(lambda x: x.split(', ')[1]).apply(lambda x: x.split('.')[0])
train_df['Title'] = train_df['Title'].apply(Find_Title)

test_df['Title'] = test_df['Name'].apply(lambda x: x.split(', ')[1]).apply(lambda x: x.split('.')[0])
test_df['Title'] = test_df['Title'].apply(Find_Title)


# In[324]:


dummy = pd.get_dummies(train_df['Title'], prefix='Title')

dummy.rename(columns={'Title_1.0':'Title_1', 'Title_2.0':'Title_2','Title_3.0':'Title_3','Title_4.0':'Title_4',
                      'Title_5.0':'Title_5','Title_6.0':'Title_6','Title_7.0':'Title_7','Title_8.0':'Title_8',
                      'Title_10.0':'Title_10'}, inplace = True)
                   
train_df = pd.concat([dummy, train_df], axis=1)
train_df = train_df.drop(['Title'],axis=1)

dummy = pd.get_dummies(test_df['Title'], prefix='Title')
test_df = pd.concat([dummy, test_df], axis=1)
test_df = test_df.drop(['Title'],axis=1)

test_df['Title_8']= 0
test_df['Title_10']= 0


# In[325]:


train_df


# In[326]:


#Dead_female_family & Survive_male_family
train_df['Surname'] = train_df['Name'].apply(lambda x:x.split(',')[0])
test_df['Surname'] = test_df['Name'].apply(lambda x:x.split(',')[0])
dead_female_surname = list(set(train_df[(train_df.Sex=='female') & (train_df.Age>=16)
                              & (train_df.Survived==0) & ((train_df.Parch>0) | (train_df.SibSp > 0))]['Surname'].values))
survive_male_surname = list(set(train_df[(train_df.Sex=='male') & (train_df.Age>=16)
                              & (train_df.Survived==1) & ((train_df.Parch>0) | (train_df.SibSp > 0))]['Surname'].values))

train_df['Dead_female_family'] = np.where(train_df['Surname'].isin(dead_female_surname),0,1)
train_df['Survive_male_family'] = np.where(train_df['Surname'].isin(survive_male_surname),0,1)
train_df = train_df.drop(['Name','Surname'],axis=1)

test_df['Dead_female_family'] = np.where(test_df['Surname'].isin(dead_female_surname),0,1)
test_df['Survive_male_family'] = np.where(test_df['Surname'].isin(survive_male_surname),0,1)
test_df = test_df.drop(['Name','Surname'],axis=1)


# In[327]:


#Child
def determine_child(age):
    if age <= 12: return 1
    else: return 0
train_df['Age'].fillna(df['Age'].dropna().median(),inplace=True)
test_df['Age'].fillna(df['Age'].dropna().median(),inplace=True)
train_df['Child'] = train_df['Age'].apply(determine_child)
test_df['Child'] = test_df['Age'].apply(determine_child)


# In[328]:


#Embarked
#df = df.drop('Embarked',axis=1)
for df in combine_df:
    df.Embarked = df.Embarked.fillna('S')
    
dummy_train = pd.get_dummies(train_df['Embarked'])
train_df = pd.concat([dummy_train, train_df], axis=1)
dummy_test = pd.get_dummies(test_df['Embarked'])
test_df = pd.concat([test_df, dummy_test], axis=1)



# In[329]:


#FamilySize
train_df['FamilySize'] = np.where(train_df['SibSp']+train_df['Parch']==0, 'Alone', 
                                  np.where(train_df['SibSp']+train_df['Parch']<=3, 'Small', 'Big'))
test_df['FamilySize'] = np.where(test_df['SibSp']+test_df['Parch']==0, 'Alone', 
                                  np.where(test_df['SibSp']+test_df['Parch']<=3, 'Small', 'Big'))

dummy_train = pd.get_dummies(train_df['FamilySize'])
train_df = pd.concat([dummy_train, train_df], axis=1)
dummy_test = pd.get_dummies(test_df['FamilySize'])
test_df = pd.concat([test_df, dummy_test], axis=1)


# In[330]:



#PClass: create dummy variables
d_pclass = pd.get_dummies(train_df['Pclass'], prefix='class')
train_df = pd.concat([train_df,d_pclass],axis=1)

d_pclass = pd.get_dummies(test_df['Pclass'], prefix='class')
test_df = pd.concat([test_df,d_pclass],axis=1)

    
#Sex: create dummy variables
train_df['Sex'] = train_df['Sex'].map({'female': 1, 'male': 0})
test_df['Sex'] = test_df['Sex'].map({'female': 1, 'male': 0})

#Fare, Age: filling the missing value by median

train_df['Fare'].fillna(train_df['Fare'].dropna().median(),inplace=True)
test_df['Fare'].fillna(test_df['Fare'].dropna().median(),inplace=True)



# In[331]:


#Drop unnecessary variables
train_df = train_df.drop(['Ticket', 'Cabin', 'FamilySize','Pclass','Embarked','SibSp','Parch'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin', 'FamilySize','Pclass','Embarked','SibSp','Parch'], axis=1)


# In[332]:


data_train = train_df.drop(['PassengerId','Alone','Q','class_2','Big','Small','C','S','class_1','class_3',
                            'Title_2','Title_3','Title_4','Title_5','Title_6','Title_7','Title_8','Title_10','Child'], axis=1)
data_test  = test_df.drop(['PassengerId','Alone','Q','class_2','Big','Small','C','S','class_1','class_3',
                            'Title_2','Title_3','Title_4','Title_5','Title_6','Title_7','Title_8','Title_10','Child'], axis=1).copy()


X_train = data_train.drop(['Survived'],axis=1)
Y_train = data_train['Survived']
X_test = data_test


# In[333]:


#play with different models

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred = logreg.predict(X_train)
acc_log = accuracy_score(Y_pred, Y_train)

# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred = svc.predict(X_train)
acc_svc = accuracy_score(Y_pred, Y_train)


#  k-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_train)
acc_knn = accuracy_score(Y_pred, Y_train)

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred = gaussian.predict(X_train)
acc_gaussian = accuracy_score(Y_pred, Y_train)

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_train)
acc_linear_svc = accuracy_score(Y_pred, Y_train)

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred = sgd.predict(X_train)
acc_sgd = accuracy_score(Y_pred, Y_train)

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred = decision_tree.predict(X_train)
acc_decision_tree = accuracy_score(Y_pred, Y_train)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_train)
acc_random_forest = accuracy_score(Y_pred, Y_train)


# In[334]:


# showing accuracy
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes',
              'Stochastic Gradient Decent', 'Linear SVC',
              'Decision Tree'],
    'Score': [acc_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian,
              acc_sgd, acc_linear_svc, acc_decision_tree]})
print(models.sort_values(by='Score', ascending=False))


# In[335]:


# Using Random Forest
# You can change the variables used for the model
random_forest = RandomForestClassifier(n_estimators=3000, max_depth=5, criterion='gini')
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)


features = pd.DataFrame()
features['Feature'] = X_train.columns
features['importance'] = random_forest.feature_importances_
print(features)


# In[337]:


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })
submission.to_csv('submission15.csv', index=False)

