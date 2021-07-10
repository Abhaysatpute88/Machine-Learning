#!/usr/bin/env python
# coding: utf-8

# In[71]:


# IMPORT THE GIEVN LIBRARY TO PERFORM LINEAR REGRESSION
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# In[72]:


# DATA
x=np.array([2600,3000,3200,3600,4000])
y=np.array([550000,565000,610000,680000,725000])


# In[73]:


linreg=LinearRegression()


# In[74]:


x=x.reshape(-1,1)


# In[75]:


# FITTING THE DATA
linreg.fit(x,y)


# In[76]:


y_pre=linreg.predict(x)


# In[77]:


# PLOTTING THE GIVEN DATA AND BEST FIT LINE
plt.scatter(x,y)
plt.plot(x,y_pre, color="red")
plt.xlabel("area(sq ft)")
plt.ylabel("price(us$)")
plt.title("Linear regression model")
plt.show
plt.grid()


# In[78]:


print(linreg.coef_)


# In[79]:


print(linreg.intercept_)


# In[80]:


linreg.predict([[3300]])


# In[81]:


m=135.78767123
x=2600
c=180616.43835616432
y=m*x+c
y


# In[82]:


linreg.predict([[2600]])


# In[ ]:




