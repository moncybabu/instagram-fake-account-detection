#!/usr/bin/env python
# coding: utf-8

# In[40]:


# load the dataset
import pandas as pd
df = pd.read_csv('C:/Users/bamo2001/OneDrive/moncy/train.csv')
df


# In[41]:


#rename the columns in data set
df.rename(columns={'profile pic': 'Profile pic', 'nums/length username': 'Length_username', 'fullname words': 'Fullname_words','name==username':'Name_username','description length':'Bioinfo_iength','external URL':'External_url','private':'Private','#posts':'Noofpost','#followers':'Nooffollowers','#follows':'Nooffollowing','fake':'Fake','nums/length fullname':'Fullname_length'}, inplace=True)


# In[42]:


df.tail()


# In[43]:


#To see the column names
columns_list = df.columns
print(columns_list)


# In[44]:


#To see no or rows and columns in a dataset
df.shape


# In[45]:


# Getting dataframe info
df.info()


# In[46]:


# Get the statistical summary of the dataframe
df.describe()


# In[47]:


# Get the number of unique values in the "profile pic" feature
df['Profile pic'].value_counts()


# In[48]:


# Get the number of unique values in "fake" (Target column)
df['Fake'].value_counts()


# In[49]:


#DATA PREPROCESSING
#1.OUTLIER DETECTION
#IQR METHOD
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df.plot(kind="box",subplots=True,layout=(7,2),figsize=(15,20))


# In[50]:


from sklearn.preprocessing import MinMaxScaler
Scaling=MinMaxScaler()
arr=Scaling.fit_transform(df)


# In[51]:


df1= pd.DataFrame(arr,columns=df.columns)


# In[54]:


plt.figure(figsize = (4,8))
sns.boxplot(y = df.Noofpost)


# In[84]:


#Outlier detection for no of post
q1 = df['Noofpost'].quantile(0.10)
q3 = df['Noofpost'].quantile(0.90)
iqr = q3 - q1
lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr
outliers1=df[(df.Noofpost<lower_limit)|(df.Noofpost>upper_limit)]


# In[86]:


#Outlier detection for no of follewers
q1 = df['Nooffollowers'].quantile(0.10)
q3 = df['Nooffollowers'].quantile(0.90)
iqr = q3 - q1
lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr
outliers2=df[(df.Nooffollowers<lower_limit)|(df.Nooffollowers>upper_limit)]


# In[87]:


#Outlier detection for no of following
q1 = df['Nooffollowing'].quantile(0.10)
q3 = df['Nooffollowing'].quantile(0.90)
iqr = q3 - q1
lower_limit = q1 - 1.5 * iqr
upper_limit = q3 + 1.5 * iqr
outliers3=df[(df.Nooffollowing<lower_limit)|(df.Nooffollowing>upper_limit)]


# In[111]:


#replace outliers with median for noofpost
median_value = df['Noofpost'].median()
df['Noofpost']= df['Noofpost'].apply(lambda x:median_value if x > upper_limit or x < lower_limit else x)


# In[113]:


#replace outliers with median for noof followers
median_value = df['Nooffollowers'].median()
df['Noofpost']= df['Nooffollowers'].apply(lambda x:median_value if x > upper_limit or x < lower_limit else x)


# In[114]:


#replace outliers with median for noof following
median_value = df['Nooffollowing'].median()
df['Noofpost']= df['Nooffollowing'].apply(lambda x:median_value if x > upper_limit or x < lower_limit else x)


# In[115]:


#finding missing values
missing_values = df.isnull()


# In[118]:


missing_count = missing_values.sum()
missing_count


# In[119]:


#finding duplicates
duplicate_rows = df.duplicated()
duplicate_count = duplicate_rows.sum()
duplicate_count


# In[123]:


#dropping duplicates
df_no_duplicates = df.drop_duplicates()
df_no_duplicates
df.shape


# In[124]:


df.head()


# # DATA VISUALIZATION
# 

# In[134]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


# In[135]:


#The account having no profile picture showing more fake accounts
pd.crosstab(df.Profile_pic,df.Fake).plot(kind="bar")


# In[136]:


#The account which is private showing more fake accounts
pd.crosstab(df.	Private,df.Fake).plot(kind="bar")


# In[137]:


#the account showing 2 name words showing more fake accounts
pd.crosstab(df.Fullname_words,df.Fake).plot(kind="bar")


# In[140]:


#The account having less no of post showing more fake accounts.
sns.histplot(x='Noofpost', data=df, kde=True, hue='Fake')
plt.show()


# In[142]:


#The account having less no of followers showing more fake accounts
sns.histplot(x='Nooffollowers', data=df, kde=True, hue='Fake')
plt.show()


# In[143]:


#The account having less no of following showing more fake accounts.
sns.histplot(x='Nooffollowing', data=df, kde=True, hue='Fake')
plt.show()


# # Model building
# 

# In[151]:


#logistic regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X = df.drop('Fake', axis=1)  # Features
y = df['Fake'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
log = LogisticRegression()
log.fit(X_train, y_train)
y_pred = log.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(log.score(X_test, y_test)))





# In[152]:


#Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
#The result is telling us that we have 59+46 correct predictions and 7+4 incorrect predictions.


# In[153]:


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[160]:


# DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

kfold = KFold(n_splits=10, shuffle=False)

Decision= DecisionTreeClassifier(criterion = 'entropy',max_depth=3,random_state=7)
Decision.fit(X_train, y_train)
y_pred1 = Decision.predict(X_test)
print('Accuracy of decision tree classifier on test set: {:.2f}'.format(Decision.score(X_test, y_test)))


# In[161]:


#Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred1)
print(confusion_matrix)


# In[162]:


#CLASSIFICATION REPORT
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[163]:


# SVC
from sklearn.svm import SVC

kfold = KFold(n_splits=10, shuffle=False)

SVC = SVC(kernel="rbf",random_state=7)
SVC.fit(X_train, y_train)
y_pred2 = SVC.predict(X_test)
print('Accuracy of SUPPORT VECTOR MACHINE classifier on test set: {:.2f}'.format(SVC.score(X_test, y_test)))


# In[168]:


#RANDOM FOREST CLASSIFIER
from sklearn.ensemble import RandomForestClassifier
num_trees = 40
max_features = 5
kfold = KFold(n_splits=10, shuffle=False)
Random = RandomForestClassifier(n_estimators=num_trees, max_features=max_features, criterion="entropy",random_state=7)
Random.fit(X_train, y_train)
y_pred3 = Random.predict(X_test)
print('Accuracy of Random forest classifier on test set: {:.2f}'.format(Random.score(X_test, y_test)))


# In[170]:


# Naive bayes classification
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.naive_bayes import GaussianNB as GB

kfold = KFold(n_splits=10, shuffle=False)
MultinomialNB = MB()
MultinomialNB.fit(X_train, y_train)
y_pred3 = MultinomialNB.predict(X_test)
print('Accuracy of naive bayes classifier on test set: {:.2f}'.format(MultinomialNB.score(X_test, y_test)))


# In[ ]:




