#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# In[2]:


print(sns.get_dataset_names())


# In[3]:


df= sns.load_dataset("tips")


# In[4]:


print(df)


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.shape


# In[8]:


df.isnull().sum()


# In[9]:


df.duplicated()


# In[10]:


df.drop_duplicates()


# In[11]:


# deduplicated_df.to_csv('cleaned_dataset.csv', index=False)


# In[12]:


df.info()


# In[13]:


df.describe()


# In[14]:


#  row1 =data.loc[3]
# row2 = data.iloc[3]
# ow1 = data.iloc[[4, 5, 6, 7]]
 
# # retrieving rows by loc method
# row2 = data.iloc[4:8]
df.quantile()


# In[15]:


df.corr()


# In[16]:


missing_value = df.isnull().sum()/df.shape[0]*100
missing_value


# In[17]:


df.isnull().sum()


# In[18]:


# drop null values
df.dropna(inplace=True)


# In[19]:


df.columns


# In[20]:


df.mean()


# In[21]:


df.median()


# In[22]:


df.mode()


# In[23]:


#rename 
df.rename(columns={'sex':'gender'})


# In[24]:


df["total_bill"].value_counts()


# In[25]:


sns.boxplot([df.total_bill])
plt.show()


# In[26]:


sns.barplot(x = 'sex',y= 'total_bill' ,data =df)
plt.show()


# In[27]:


sns.countplot(data= df, x= "sex" , hue= "tip")
plt.show()


# In[28]:


sns.scatterplot(x= "total_bill", y= "sex" , data = df)
plt.show()


# In[29]:


# sales_state = df.groupby(['Marital_Status', 'Gender'], as_index=False)['Amount'].sum().sort_values(by='Amount', ascending=False)

# sns.set(rc={'figure.figsize':(6,5)})
# sns.barplot(data = sales_state, x = 'Marital_Status',y= 'Amount', hue='Gender')


# In[30]:


sns.pairplot(df, hue ='sex')
plt.show()


# In[31]:


sns.lineplot(x="total_bill", y="size", 
             hue="sex", style="sex", 
             data=df) 
  
plt.show()


# # #splitting the data

# In[32]:


# x = df.iloc[:,:4]
# y = df.iloc[:,4]


# In[33]:


# x = df.iloc[:,:1]
# y = df.iloc[:,2]
x = df[['total_bill']]
y = df[['tip']]


# In[34]:


x.head()


# In[35]:


y.head()


# In[36]:


# x = df[['Age']]
# y = df[['Orders']]


# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.20 , random_state = 0)


# In[39]:


x_train


# In[40]:


y_train


# In[41]:


from sklearn.linear_model import LinearRegression


# In[42]:


lr = LinearRegression()


# In[43]:


lr.fit(x_train,y_train)


# In[44]:


y_pred = lr.predict(x_test)
print(y_pred)


# In[45]:


from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_squared_error 


# In[46]:


mean_squared_error(y_test, y_pred)


# In[47]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[48]:


# print(accuracy_score(y_test, y_pred))


# In[49]:


# print(classification_report(y_test, y_pred))


# In[50]:


# print(confusion_matrix(y_test, y_pred))


# In[52]:


# df = pd.read_csv(r"C:\Users\HP\Downloads\Resume Screening1.csv")
# print(df)
# df.to_csv('d\\hjug.csv')


# In[53]:


# counts = df['Category'].value_counts()
# labels = df['Category'].unique()
# plt.figure(figsize=(15,10))
# plt.pie(counts, labels=labels, autopct='%1.1f%%', shadow=True, colors=plt.cm.plasma(np.linspace(0,1,3)))
# plt.show()
df = pd.read_excel("D:\clg pic\salkanpur\Excel Test for MIS.xlsx")
print(df)


# In[ ]:




