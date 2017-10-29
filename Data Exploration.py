
# Trying different graphs with matplotlib 
# From graphs, we can see: 
                        #1. Female Survival Rate are much higher than man
                        #2. Fare price and Pclass are important
                        #3. Children survival rate are really high

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


train = pd.read_csv('train.csv')


# In[27]:


fig1 = plt.figure(figsize=(10,4))
fig1.add_subplot(121)
train.Survived[train['Sex']=='male'].value_counts().plot(kind='pie')
fig1.add_subplot(122)
train.Survived[train['Sex']=='female'].value_counts().plot(kind='pie')


# In[32]:


fig2 = plt.figure(figsize=(8,6))
g1 = fig2.add_subplot(121)
male = train[train['Sex'] == 'male']
female = train[train['Sex']=='female']
s_male = male['Survived'] == 1
s_female = female['Survived'] == 1

g1.set_title('Survival Male/Female Number')
g1.set_xticks([0,1])
g1.set_xticklabels(['male','female'])
g1.set_xlabel('Sex')
g1.set_ylabel('Number')
g1.grid
g1.bar(0, s_male.sum(), align="center", color='yellow', alpha=0.5)
g1.bar(1, s_female.sum(), align="center", color='pink', alpha=0.5)



# In[38]:


fig3 = plt.figure(figsize=(8,6))
g2 = fig3.add_subplot(121)
male = train[train['Sex'] == 'male']
female = train[train['Sex']=='female']
s_male = male['Survived'] == 1
s_female = female['Survived'] == 1

g2.set_title('Male/Female Survival Rate')
g2.set_xticks([0,1])
g2.set_xticklabels(['male','female'])
g2.set_xlabel('Sex')
g2.set_ylabel('Survival Rate')
g2.grid
g2.bar(0, float(s_male.sum())/len(male.index),align="center", color='yellow', alpha=0.5)
g2.bar(1, float(s_female.sum())/len(female.index),align="center", color='pink', alpha=0.5)


# In[11]:


fig4 = plt.figure(figsize=(10,6))
age = train[train['Age'].notnull()]
age['Age']=age['Age'].astype(int)
Survived = train[train['Survived']== 1 ]
survived_age = []
for i in range (1,18):
    survived_age.append(len(Survived[(Survived['Age']>= (i-1)*5)&(Survived['Age']<i*5)]))
ax1 = fig4.add_subplot(1,2,1)
ax1.set_title('Age Distribution of Survivors')
ax1.set_xticks(range(5,85,5))
ax1.set_xlabel("Age")
ax1.set_ylabel("Survival Count")
ax1.set_ylim(0,45)
ax1.grid()
ax1.bar(range(5,90,5),survived_age, width = 3, align='center', color = 'g', alpha=0.5)


# In[16]:


train['Fare'] = train['Fare'].astype(int)
survived = train[train['Survived']==1]
unsurvived = train[train['Survived'] == 0]
survivedFare = survived.groupby('Fare').size()
unsurvivedFare = unsurvived.groupby('Fare').size()
fig1 = plt.figure(figsize = (8,9))
ax1 = fig1.add_subplot(211)
ax1.set_title("Survivor's Fare Distribution")
ax1.set_xlabel('Fare')
ax1.set_ylabel('Count')
ax1.set_xlim(0,180)
ax1.bar(survivedFare.index, survivedFare, color = 'g')


# In[20]:


fig = plt.figure(figsize=(12,6))
p1 = train[train['Pclass']==1]
p2 = train[train['Pclass']==2]
p3 = train[train['Pclass']==3]
s_p1 = len(p1[p1['Survived']== 1])
s_p2 = len(p2[p2['Survived']== 1])
s_p3 = len(p3[p3['Survived']== 1])
ax1 = fig.add_subplot(121)
ax1.set_title('Survival Number vs Pclass')
ax1.set_xlabel('class')
ax1.set_ylabel('Number')
ax1.set_xticks([0,1,2])
ax1.set_xticklabels(['Pclass1','Pclass2','Pclass3'])
ax1.set_ylim(0,140)
ax1.bar(0, s_p1, align='center', color='grey', alpha=0.5)
ax1.bar(1, s_p2, align='center', color='yellow', alpha=0.5)
ax1.bar(2, s_p3, align='center', color='green', alpha=0.5)


# In[1]:


y_dead = train[train.Survived==0].groupby('Pclass')['Survived'].count()
y_alive = train[train.Survived==1].groupby('Pclass')['Survived'].count()
fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(121)
ax.set_xticks([1,2,3])
ax.bar([1,2,3], y_dead, color='r', alpha=0.6, label='dead')
ax.bar([1,2,3], y_alive, color='g', bottom=y_dead, alpha=0.6, label='alive')
ax.legend(fontsize=16, loc='best')
ax.set_xticklabels(['Pclass%d'%(i) for i in range(1,4)], size=15)
ax.set_title('Pclass Surveved count', size=20)


# In[26]:


fig = plt.figure(figsize=(10,6))
s = train[train['SibSp']==1]
us = train[train['SibSp']==0]
s_s = s[s['Survived']==1]
s_us = us[us['Survived']==1]
rate_s = float(len(s_s)/len(s))
rate_us = float(len(s_us)/len(us))
ax1 = fig.add_subplot(121)
ax1.set_title('Survival Rate / Sibsp')
ax1.set_xlabel('Sibsp')
ax1.set_ylabel('Survival Rate')
ax1.set_xticks([0,1])
ax1.set_xticklabels(['1', '0'])
ax1.set_ylim(0,1)
ax1.bar(0, rate_s, align='center', color='red', alpha = 0.5)
ax1.bar(1, rate_us, align='center', color='blue', alpha = 0.5)


# In[28]:


fig = plt.figure(figsize=(10,6))
p = train[train['Parch']>0]
up = train[train['Parch']==0]
s_p = p[p['Survived']==1]
s_up = up[up['Survived']==1]
rate_p = float(len(s_p)/len(p))
rate_up = float(len(s_up)/len(up))
ax1 = fig.add_subplot(131)
ax1.set_title('Survival Rate / Parch')
ax1.set_xlabel('Parch')
ax1.set_ylabel('Survival Rate')
ax1.set_xticks([0,1])
ax1.set_xticklabels(['1', '0'])
ax1.set_ylim(0,1)
ax1.bar(0, rate_p, align='center', color='pink', alpha = 0.5)
ax1.bar(1, rate_up, align='center', color='green', alpha = 0.5)


# In[8]:


y_dead = train[train.Survived==0].groupby('Pclass')['Survived'].count()
y_alive = train[train.Survived==1].groupby('Pclass')['Survived'].count()
fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(111)
pclass = [1,2,3]
ax.bar(pclass, y_dead, color='r', alpha=0.7, label='dead')
ax.bar(pclass, y_alive, color='g', bottom=y_dead, alpha=0.7, label='alive')
ax.legend(fontsize=16, loc='best')
ax.set_xticks(pclass)
ax.set_xticklabels(['Pclass%d'%(i) for i in range(1,4)], size=15)
ax.set_title('Pclass Surveved count', size=20)


# In[9]:


print(train.Sex.value_counts())


# In[10]:


print(train.groupby('Sex')['Survived'].mean())


# In[31]:


label = []
for pclass in range (1,4):
    for sex in ['Female', 'Male']:
        label.append('Sex:%s|Pclass:%d'%(sex, pclass))
        
pos = range(6)
fig = plt.figure(figsize=(16,4))
ax = fig.add_subplot(111)
ax.bar(pos, 
        train[train['Survived']==0].groupby(['Pclass','Sex'])['Survived'].count().values, 
        color='yellow', 
        alpha=1, 
        align='center',
        tick_label=label, 
        label='dead')
ax.bar(pos, 
        train[train['Survived']==1].groupby(['Pclass','Sex'])['Survived'].count().values, 
        bottom=train[train['Survived']==0].groupby(['Pclass','Sex'])['Survived'].count().values,
        color='pink',
        alpha=1,
        align='center',
        tick_label=label, 
        label='alive')
ax.tick_params(labelsize=13)
ax.set_title('Sex & Pclass % Survived', size=18)
ax.legend(fontsize=12)


# In[33]:


train.isnull().sum()


# more aggregated graph 

# In[34]:


train.hist(figsize=(15,15))


# In[35]:


train.plot(kind='density', subplots='True', sharex=False, figsize=(10,10))


# In[36]:


from pandas.plotting import scatter_matrix
scatter_matrix(train, figsize=(10,10))


