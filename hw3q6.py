
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import csv
import copy

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


# In[4]:


def ModelEvaluate(real_lab, pre_lab, verbo):
    conf_matrix = confusion_matrix(real_lab, pre_lab)
    acc = (conf_matrix[0][0] + conf_matrix[1][1])/(conf_matrix.sum())
    recall = conf_matrix[1][1]/(conf_matrix[1][0] + conf_matrix[1][1])
    precision = conf_matrix[1][1]/(conf_matrix[0][1] + conf_matrix[1][1])
    fscore = 2 * (recall * precision)/(recall + precision)
    if verbo == 1: 
        print('Perofrmance Measurements:')
        print('Confusion matrix: \\n', conf_matrix)
        print('Accuracy: ', acc)
        print('Recall: ', recall)
        print('Precision: ', precision)
        print('F-score: ', fscore)
    return conf_matrix, [acc, recall, precision, fscore]


# In[5]:


## (a)
# Load data
data_df = pd.read_csv('hw3q6.csv')
print('Number of positive: ', len(data_df.loc[data_df['Class'] == 1]))
print('Number of negative: ', len(data_df.loc[data_df['Class'] == 0]))

data = np.array(data_df[data_df.columns[:-1]])
label = np.array(data_df[data_df.columns[-1]])
x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.25, random_state = 0, stratify=label)
print('Positive samples in training data: ', sum(y_train == 1))
print('Negative samples in training data: ', sum(y_train == 0))
print('Positive samples in testing data: ', sum(y_test == 1))
print('Negative samples in testing : ', sum(y_test == 0))


# In[6]:


## (b)
# Change C to check the # of SVs and the accuracy
c_set = [0.1, 0.5, 1, 5, 10, 50, 100]
sv_num_set = []
acc_set = []
for i in range(len(c_set)):
    paraC = c_set[i]
    clf = SVC(kernel='linear', C = paraC)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    [_, [acc, _, _, _]] = ModelEvaluate(y_test,y_pred,verbo=0)
    sv_num_set.append(len(clf.support_))
    acc_set.append(acc)

# Plot the number of SVs
fig = plt.figure(figsize=(8,6))
ax = plt.gca()
ax.grid(linestyle='--')
plt.plot(c_set, sv_num_set)
plt.xlabel('Regularization parameter C', fontsize=16)
plt.ylabel('Number of SVs', fontsize=16)
plt.title('Number of SVs with different C', fontsize=16)
plt.savefig('SVs.png')
plt.show(block=True)


# In[7]:


## (c)
kernel_set = ['linear', 'poly', 'rbf', 'sigmoid']
clf_set = [SVC(kernel='linear',C = 10), 
           SVC(kernel='poly', C=100, degree=1, coef0=0.01), 
           SVC(kernel='rbf', C=5, gamma=0.1),
           SVC(kernel='sigmoid', C=10000, gamma=0.001, coef0=0.01)]

for i in range(len(clf_set)): 
    kernel_tmp = kernel_set[i]
    clf_tmp = clf_set[i]
    print('\nKernel: ', kernel_tmp)
    clf_tmp.fit(x_train, y_train)
    y_pred = clf_tmp.predict(x_test)
    ModelEvaluate(y_test,y_pred,verbo=1)

