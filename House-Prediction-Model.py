
# coding: utf-8

# # Machine Learning model to predict the prices of home in IOWA

# ## Import the Libraries

# In[1]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')


# ## Functions to preprocess the data

# In[18]:


#loading the .csv file into dataset
def load_data(filename):
    dataset = pd.read_csv(filename)
    return dataset

#splitting the data into target and feature variables
def feature_target_data(dataset, feature):
    target = dataset[feature]
    feature = dataset[dataset.columns.tolist()[:-1]]
    return feature, target

#finding if any features are almost null in all the examples
def finding_null_variables(dataset):
    null_variables = []
    for i in dataset.columns.tolist():
        if dataset[i].isnull().sum() > 700 :
            null_variables.append(i)
    return null_variables

#handling the null values
def replacing_the_null_values(dataset, numerical_cols, categorical_cols):
    for i in numerical_cols:
        dataset.loc[dataset[i].isnull() , i] = 0
    for i in categorical_cols:
        dataset.loc[dataset[i].isnull(), i] = 'Not-Available'
    return dataset

#remove some irrelavant features from dataset
def update_dataset(dataset, features_to_remove):
    new_features = set(dataset.columns.tolist()) - set(features_to_remove)
    updated_dataset = dataset[list(new_features)]
    return updated_dataset

#categorizing numerical and categorical features
def split_categorical_numerical_variables(dataset):
    numerical_cols = dataset.describe(include=[np.number]).columns.tolist()
    categorical_cols = list(set(dataset.columns.tolist()) - set(numerical_cols) - {'Street'})
    numerical_cols = list(set(numerical_cols) - {'Id', 'SalePrice'})
    return numerical_cols, categorical_cols

#merge train and test data if we can see the difference in unique category of both the datasets
def consolidate(train, test):
    train_test = pd.concat([train, test])
    return train_test

#drop some feature from dataset
def drop_feature(dataset, feature):
    updated_dataset = dataset.drop(columns = feature)
    return updated_dataset

#one_hot_encoding of the categorical varaiables
def one_hot_encoding(dataset, categorical_cols):
    keys = range(len(dataset.columns.tolist()))
    keys_categorical_value = dict(zip(keys, dataset.columns.tolist() ))
    labelencoder = LabelEncoder()
    categorical_keys = []
    dataset_for_training = dataset.iloc[:,:].values
    for i in keys_categorical_value.keys():
        if keys_categorical_value[i] in categorical_cols:
            dataset_for_training[:,i] = labelencoder.fit_transform(dataset_for_training[:, i].astype(str))   
            categorical_keys.append(i)
    onehotencoder = OneHotEncoder(categorical_features=categorical_keys)
    dataset_for_training = onehotencoder.fit_transform(dataset_for_training).toarray()
    return dataset_for_training

#splitting the train and test datasets
def train_test_split(dataset):
    training_data = dataset[:1460]
    test_data = dataset[1460:]
    return training_data, test_data

#model training
def train_model(model, feature_df, target_df, num_procs, mean_mse, cv_std):
    neg_mse = cross_val_score(model, feature_df, target_df, cv=5, n_jobs=num_procs, scoring='neg_mean_squared_error')
    mean_mse = -1.0*np.mean(neg_mse)
    cv_std = np.std(neg_mse)
    print('\nModel:\n', model)
    print('Average MSE:\n', mean_mse)
    print('Standard deviation during CV:\n', cv_std)


# ### Loading the data into pandas dataframe

# In[39]:


dataset = load_data('train.csv')
testing = load_data('test.csv')


# ### Splitting the data into features and target

# In[40]:


training, target = feature_target_data(dataset, 'SalePrice')


# In[41]:


null_features = finding_null_variables(training) #features which are almost null in the dataset
print('******Null_Features******')
print(null_features)
print()
training = update_dataset(training, null_features) #update the training set
testing = update_dataset(testing, null_features) #update the testing set

#finding the numerical and categorical features
numerical_cols, categorical_cols = split_categorical_numerical_variables(training)

print('******Numerical_Columns******')
print(numerical_cols)
print()
print('******Categorical_Columns******')
print(categorical_cols)

#consolidating the train and test as unique features are missing in test dataset
train_test = consolidate(training, testing)

#handling the null_values or empty values from the consolidated dataset
train_test = replacing_the_null_values(train_test, numerical_cols, categorical_cols)

#drop some features which are not in use
train_test = drop_feature(train_test, 'Street')
train_test = drop_feature(train_test, 'Id')


# ### Preprocess the data like label encoding and onehotencoding to create the model

# In[42]:


dataset_for_training = one_hot_encoding(train_test, categorical_cols) #one hot encoding 
training_data, test_data = train_test_split(dataset_for_training) #splitting the data back to train and test


# ### Model Evaluation among Gradient Boosting and Random Forest Regression using K-Fold Cross Validation

# In[43]:


num_procs = 2 #number of process in parallel

#initialize some neccessary dicts and lists
models = []
mean_mse = {}
cv_std = {}
res = {}
verbose_lvl = 0
rf = RandomForestRegressor(n_estimators=150, n_jobs=num_procs, max_depth=25, min_samples_split=60, max_features=None)
gbm = GradientBoostingRegressor(n_estimators=150, max_depth=25, loss='ls', max_features=None, min_samples_split=60)
models.extend([rf, gbm])
#parallel cross-validate models, using MSE as evaluation metric, and print summaries
print("Beginning cross validation")
for model in models:
    train_model(model, training_data, target, num_procs, mean_mse, cv_std)


# ### According to results Gradient Boosting got better results so will proceed with Gradient Boosting and tune it using Grid Search
# 

# In[ ]:


parameters = [{'learning_rate': [0.16, 0.18, 0.185], 'n_estimators': [150, 175, 200], 'min_samples_split' : [5,10,15,20],
               'max_features': [100, 110, 150, None]}]
grid = GridSearchCV(estimator = gbm, param_grid= parameters, scoring= 'neg_mean_squared_error', cv = 10)
grid = grid.fit(training_data, target)
best_score = grid.best_score_
results = grid.cv_results_
print(best_score)
print(results)


# In[120]:


gbm_check = GradientBoostingRegressor(n_estimators=210, max_features=160, max_depth=3, random_state=0, 
                                       learning_rate = 0.18, min_samples_split=3)
train_model(gbm_check, training_data, target, num_procs, mean_mse, cv_std)


# ### Lets use Gradient Boosting with these features

# In[121]:


gbm = GradientBoostingRegressor(n_estimators=210, max_features=160, max_depth=3, random_state=0, 
                                       learning_rate = 0.18, min_samples_split=3)
gbm.fit(training_data, target)
y_pred = gbm.predict(test_data)


# ### Creation of a file 'results.csv' 

# In[122]:


index = list(range (1461, 2920))
with open('result.csv', 'a+') as r:
    r.write('Id,SalePrice\n')
    for i in range(len(y_pred)):
        r.write('%i,%f\n'%(index[i], y_pred[i]))


# ### Display some results

# In[123]:


result = pd.read_csv('result.csv')
result.head()

