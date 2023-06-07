#!/usr/bin/env python
# coding: utf-8

# # Data Analysis with Python

# 1. Problem statement
# What  are the main characteristics which have the most impact on the  car price ?

# In[1]:


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileEDA.csv'
df = pd.read_csv(path)
df.head()


# 2.Analyzing Individual Feature Patterns using Visualization

# In[6]:


print(df.dtypes)


# In[7]:


df.corr()


# In[8]:


# Correlation between  bore, stroke ,compression-ratio  and horsepower.
df[['bore','stroke','compression-ratio','horsepower']].corr()


# In[9]:


#  Engine size as potential predictor  variable of price
sns.regplot(x="engine-size",y="price", data=df )
plt.ylim(0,)


# output:9 As the engine-size goes ups,the  price goes up:this indicates a positive  direct correlation between these two variables.Engine size seems like a pretty good predictor of price since the  regression  line is almost a perfect diagonal line.

# In[10]:


# Calculating  the  correlation  between engine and price
df[["engine-size" , "price"]].corr()


# In[14]:


# highway mpg is a potential predictor variable of price
sns.regplot(x="highway-mpg" , y="price" , data=df)
plt.ylim(0,)


# Output 14: As the highway-mpg goes up ,the price goes down:this indicates  an inverse/negative  relationship between these two variables.Highway mpg could potentially  be a predictor of price.

# In[15]:


# Correlation highway-mpg and price
df[["highway-mpg" , "price"]].corr()


# In[19]:


# Scatter plot for  peak-rpm  and price

sns.regplot(x="peak-rpm", y="price", data=df)

plt.ylim(0,)


# Output 19: Peak rpm does not seem like  a good predictor of the price at all since the regression line is close to horizontal.Also,the data points are very scattered and far  from the fitted line ,showing lots of variability.There its  it is not a reliable variable.

# In[20]:


# Calculating the Correlation peak-rpm and price
df[['peak-rpm','price']].corr()


# In[21]:


# Calculation the correlation between  stroke  and price
df[["stroke","price"]].corr()


# In[23]:


sns.regplot(x="stroke" , y="price", data=df)
plt.ylim(0,)


# Output23: Stroke is not good predictor because the linear regression line it looks more horizontal.Also  scatter plots are far from  the line.

# In[7]:


# Relationship  between body-style and price
sns.boxplot(x="body-style", y="price", data=df)
plt.ylim(0,)


# Output:4: It shows  body_style  categories  have a significant  overlap, and so body-style would not be  a good predictor  of price .

# In[6]:


# Relationship between engine  location  and price
sns.boxplot(x="engine-location",y="price", data=df)
plt.ylim(0,)


# Output 6: The distribution  of price  between  these two engine -location categories,front and rear, are distinct enough to take  engine-location as a potential good predictor of price.

# In[8]:


#drive- wheels
sns.boxplot(x="drive-wheels" , y="price" , data=df)


# Output 8 ; The  distribution of price  between the different  drive-wheels categories differs,as  such drive-wheels could potentially be predictor of  price.

# In[4]:


df.describe()


# In[5]:


df.describe(include=['object'])


# In[6]:


# calculating value counts  for drive wheels
df['drive-wheels'].value_counts()


# In[7]:


# converting  the series to a dataframe
df['drive-wheels'].value_counts().to_frame()


# In[8]:


# Renaming drive_wheels to value_counts
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels':'value_counts'}, inplace=True)
drive_wheels_counts


# In[9]:


#Renaming -location as variable
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)


# In[10]:


# Grouping by variable wheels
df['drive-wheels'].unique()


# In[11]:


# creating  df_group_one
df_group_one = df[['drive-wheels', 'body-style' , 'price']]


# In[12]:


#grouping results
df_group_one = df_group_one.groupby(['drive-wheels'], as_index=False).mean()
df_group_one


# Out 18: From the data ,it seems  rear-wheel drive vehicles are,on average,the most expensive,while 4-wheel and front-wheel are approximately the same in price.

# In[13]:


#Grouping the results
df_gptest = df[['drive-wheels','body-style' , 'price']]
grouped_test1 =df_gptest.groupby(['drive-wheels' , 'body-style'], as_index=False).mean()
grouped_test1


# This grouped data is much easier to visualize when it is made into a pivot table.We can conver the dataframe to apivot table using the method  "pivot" to  create a pivot table  from the groups. 

# In[14]:


# Creating  a pivot table.
grouped_pivot =grouped_test1.pivot(index='drive-wheels',columns = 'body-style')
grouped_pivot


# In[18]:


grouped_pivot = grouped_pivot.fillna(0)
grouped_pivot


# In[20]:


df_gptest2 = df[['body-style','price']]
grouped_test_bodystyle =df_gptest2.groupby(['body-style'],as_index=False).mean()
grouped_test_bodystyle


# In[24]:


import matplotlib.pyplot as plt


# In[25]:


# Relationship  between Body Style vs Price
plt.pcolor(grouped_pivot, cmap='Rdbu')
plt.colorbar()
plt.show()


# In[26]:


from scipy import stats


# In[28]:


# Calculating the Pearson Correlation Coefficient  and P-value of 'wheel-base'
pearson_coef,p_value =stats.pearsonr(df['horsepower'], df['price'])
print("The Pearson Correlation Coefficient  is", pearson_coef, "with a P-value of P =", p_value)


# Conclusion:p-value is <0.001 , the correlation  between wheel -base and price is statistically significant ,although the linear relationship isn"t extremely strong (~0.586)

# In[ ]:




