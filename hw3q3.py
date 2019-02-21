
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import csv

from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error


# In[2]:


## (i)
# Load the data
datalab_df = pd.read_csv('hw3q3(b).csv')
feat_set = ['X1', 'X2', 'X3', 'X4']
label_set = ['Y']

model_feat_set = ['X1', 'X2', 'X3', 'X4']
model_para_set = ['Alpha', 'Beta', 'Gamma']

for i in range(3):
    # Calculate the coefficient of three models
    model_feat = model_feat_set
    model_para = model_para_set[i]
    x = np.append(np.array(datalab_df[model_feat]) ** (i+1), np.ones([len(datalab_df),1]), 1)
    y = np.array(datalab_df[['Y']])
    para = np.dot(np.linalg.pinv(x), y)
    print('Parameter', model_para, 'for model_' + str(i+1) +': \n', para)
print('\n')

   
## (ii)
for i in range(3):    
    # Predict labels and calculate RMSE of three models
    model_feat = model_feat_set
    model_para = model_para_set[i]
    x = np.append(np.array(datalab_df[model_feat]) ** (i+1), np.ones([len(datalab_df),1]), 1)
    y = np.array(datalab_df[['Y']])
    #kf = KFold(n_splits=10, shuffle=True)
    loo = LeaveOneOut()
    y_test_set = []; y_pred_set =[]
    
    count = 0
    for train_index, test_index in loo.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        para_train = np.dot(np.linalg.pinv(x_train), y_train)
        y_pred = np.dot(x_test, para_train)
        y_test_set.extend(y_test[0].tolist())
        y_pred_set.extend(y_pred[0].tolist())
        count = count + 1
        
    rmse = np.sqrt(mean_squared_error(y_test_set, y_pred_set))
    print('RMSE for model_' + str(i+1) +': \n', rmse, '\n')

