
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')


# In[5]:


x_train = dataset_train.iloc[:,:-1].values
Y = dataset_train.iloc[:,-1].values
x_test = dataset_test.iloc[:,:].values
X = np.concatenate((x_train, x_test), axis = 0)


# In[6]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
non_categorical_data = [0,1,3,4,17,18,19,20,26,34,36,37,38,43,44,45,46,47,48,49,50,51,52,54,56,59,61,62,66,67,68,69,70,71,75,76,77]
non_categorical_data = np.asarray(non_categorical_data)
total_data = np.arange(X.shape[1])
categorial_data = np.setdiff1d(total_data,non_categorical_data)
for i in categorial_data:
    X[:, i] = labelencoder.fit_transform(X[:, i].astype(str))


# In[7]:


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:, non_categorical_data])
X[:, non_categorical_data] = imputer.transform(X[:, non_categorical_data])


# In[8]:


onehotencoder = OneHotEncoder(categorical_features=categorial_data)
X = onehotencoder.fit_transform(X).toarray()


# In[9]:


x_t = X[0:x_train.shape[0]]
x_t = np.delete(x_t, [275], axis=1)
x_t2 = X[x_train.shape[0]:X.shape[0]]
x_t3 = np.delete(x_t2, [275], axis=1)


# In[10]:


Y = Y.reshape((Y.shape[0],1))
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=250, random_state=0)
regressor.fit(x_t,Y)
y_pred = regressor.predict(x_t3)


# In[11]:


print(y_pred)

