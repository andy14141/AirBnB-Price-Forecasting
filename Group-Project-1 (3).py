#!/usr/bin/env python
# coding: utf-8

# In[2]:


## Andy Kim -- 28396159
## Dimitrije Gacic -- 84137744
## Nila Akter -- 31043169


# In[3]:


#Imported Pandas
import pandas as pd


# In[4]:


#Read the given airbnb csv file and check for duplicated values
df = pd.read_csv('airbnb data.csv', sep='\t')

df.duplicated()
has_duplicate = df.duplicated()
duplicates = df[has_duplicate]

print(duplicates)


# In[5]:


#Remove duplicate and missing values
df = pd.read_csv('airbnb data.csv').drop_duplicates().dropna()


# In[6]:


df


# In[7]:


#Independent data
x = df[['accommodates', 'bathrooms', 'num_beds','property_type', 'neighbourhood']]
#Dependent data
y = df['price']


# In[8]:


#Set dummy values for categorical columns to avoid the dummy variable trap (multicollinearity)
ptype = pd.get_dummies(x['property_type'], drop_first=True)


# In[9]:


hood = pd.get_dummies(x['neighbourhood'], drop_first=True)


# In[10]:


#Remove the categorical values from the orginal dataset
x = x.drop('property_type', axis=1)


# In[11]:


x = x.drop('neighbourhood', axis=1)


# In[12]:


#Append new assigned values into the original dataset
x = pd.concat([x,ptype,],axis=1)


# In[13]:


x = pd.concat([x,hood,],axis=1)


# In[14]:


#Split the data into training data and testing data
from sklearn.model_selection import train_test_split


# In[15]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)


# In[16]:


x_train


# In[17]:


#Our Multilinear cost function
from sklearn.linear_model import LinearRegression


# In[18]:


regressor = LinearRegression()
regressor.fit(x_train, y_train)


# In[19]:


y_pred = regressor.predict(x_test)


# In[20]:


#Compare original airbnb cost with our cost prediction model
from sklearn.metrics import r2_score


# In[21]:


score = r2_score(y_test, y_pred)


# In[22]:


score


# ### Project Report

# We began our data analysis by importing pandas for data manipulation. Using the Pandas library, we successfully imported the ‘airbnb data.csv’ file into Jupyter lab. This was done by using the .read_csv function. Our first task was to remove the duplicates and drop any rows that were missing values. We simply implemented the ‘drop_duplicates()’ (return the data frame with any duplicate values removed) and ‘dropna()’ (return a new series with missing values removed.) functions in the same syntax.
# 
# Once we ran our dataframe (df), we were able to pull 408 rows from the original 410. We were also able to visually notice one row to be missing some values from the original ‘airbnb data.csv’ file. Also, using df.duplicated(), we were able to detect one row that answered ‘True’. Hence, we knew that our code was working correctly. During this step, something our group noticed is that we had to complete multiple attempts to come to the correct conclusion. When doing so, our ‘wrong’ versions didn’t give us any errors but we could tell it wasn’t the right code because we could print and see our output. 
# 
# To create a function that would predict our model, we set the following x and y variables:
# 
# x = df[['accommodates', 'bathrooms', 'num_beds','property_type', 'neighbourhood']]
# y = df['price']
# 
# During this process, we realized that some of the columns were not quantifiable (i.e. property type and neighborhood). We needed to transform the categorical data into quantitative data by using the pandas.get_dummies() function. The function allowed us to convert the categorical variables into dummy/indicator variables. 
# 
# ptype = pd.get_dummies(x['property_type'], drop_first=True)
# hood = pd.get_dummies(x['neighbourhood'], drop_first=True)
# 
# We also added a ‘drop_first=True’ statement to avoid a dummy variable trap (scenario in which the independent variable is multicollinear - a value can be easily predicted from other values). We executed the ‘get_dummies()’ function on both the 'property_type' and 'neighbourhood' columns. 
# 
# Once we set a dummy value for the two columns, we had to append those back to our previous dataframe. We used the pandas.concat() function to complete this step. To train and test our final data, we imported ‘train_test_split’ from the sklearn.model_selection library to split the data into training sets and test sets. By importing the LinearRegression from the same library, we were able to fit the multiple linear regression to the training set. Using LinearRegression, we were able to return the y_pred (prediction model) to be compared to the given data of ‘airbnb data.csv’.
# 
# We used the r^2 statistical measure to evaluate the accuracy of our model. This measure represents how close the data is to the fitted linear regression line we created earlier. The value of r^2 can range between 0% and 100%, with a score closer to 100% signifying the regression predictions fit the data well. Overall we achieved a score value of 0.9476008274475219 (94.76%), which is a strong indication that our prediction is accurate. Hence we were able to conclude that the features in Airbnb are a good indicator for predicting the price.
# 
# By (Name -- Student Number):
# 
# Andy Kim -- 28396159
# Dimitrije Gacic -- 84137744
# Nila Akter -- 31043169

# In[ ]:




