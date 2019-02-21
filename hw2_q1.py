import numpy as np
from numpy import linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.neighbors as sk
from sklearn import metrics as mt

training_dataset=pd.read_csv("./Data/hw2q1_train.csv")
test_dataset=pd.read_csv("./Data/hw2q1_test.csv")
training_class=training_dataset["Class"]
training_dataset=training_dataset.drop(labels='Class',axis=1)
test_class=test_dataset["Class"]
test_dataset=test_dataset.drop(labels='Class',axis=1)

#a
print("Size of training data set: ",training_dataset.shape)
print("Size of testing data set: ",test_dataset.shape)
count_R=0
for i in range(len(training_class)):
    if training_class.iloc[i]=="R":
        count_R+=1
print("R in training dataset: ",count_R)
print("M in training dataset: ",len(training_class)-count_R)

count_R=0
for i in range(0,len(test_class)):
    if test_class.iloc[i]=="R":
        count_R+=1
print("R in testing dataset: ",count_R)
print("R in testing dataset: ",len(test_class)-count_R)

#b
#i
normalized_dataset=training_dataset
normalized_test_dataset=test_dataset

normalized_dataset=(training_dataset-training_dataset.min())/((training_dataset).max()-(training_dataset).min())
normalized_test_dataset=(test_dataset-(training_dataset).min())/((training_dataset).max()-(training_dataset).min())
print(normalized_dataset)
print(normalized_test_dataset)
covariance=normalized_dataset.cov()
eigen_values, eigen_vectors=LA.eig(covariance)
print("Covariance Matrix size", covariance)
cov_test=normalized_test_dataset.cov()
test_eigen_values,test_eigen_vectors = LA.eig(cov_test)

#ii
max_eigen=[]
max_eigen=np.copy(eigen_values)
arr = [i for i in range(len(eigen_values))]
max_eigenTr_pos=[]
print("Max eigenvalues")
for i in range(5):
    pos = np.where(max_eigen == max(max_eigen))
    print(max(max_eigen))
    max_eigen[pos] = -1
    max_eigenTr_pos.append(pos)

plt.plot(eigen_values)
plt.xlabel("Number of Eigenvalues")
plt.ylabel("Eigenvalues")
plt.show()
# print(test_result)
# test_result=np.real(test_result)
p=[2,4,8,10,20,40,60]
# p=[10]
csv_print=""
accuracy_rate=[]
for k in range(len(p)):
    max_eigen=np.copy(eigen_values)
    max_eigenTr_pos=[]
    print("Max eigenvalues")
    for i in range(p[k]):
        pos = np.where(max_eigen == max(max_eigen))
        max_eigen[pos] = -1
        print(max(max_eigen))
        max_eigenTr_pos.append(pos[0])
    feature_vector =[]
    for i in range(len(max_eigenTr_pos)):
        feature_vector.append(eigen_vectors[:,i])
        # print(eigen_vectors[max_eigenTr_pos[i]])
    # print(feature_vector)
    # print(eigen_vectors)

    max_eigen = np.copy(test_eigen_values)
    max_eigenTs_pos = []
    for i in range(p[k]):
        pos = np.where(max_eigen == max(max_eigen))
        # print(max(max_eigen))
        max_eigen[pos] = -1
        max_eigenTs_pos.append(pos[0])
    feature_vector_testing = []
    for i in range(len(max_eigenTs_pos)):
        # print(test_eigen_vectors[max_eigenTs_pos[i]])
        feature_vector_testing.extend(test_eigen_vectors[max_eigenTs_pos[i]])
    print(len(feature_vector_testing))
    # print(feature_vector,"FV")
    test_result = np.matmul(feature_vector, np.transpose(normalized_test_dataset))
    result=np.matmul(feature_vector,np.transpose(normalized_dataset))
    # print(np.transpose(result).shape)

    result=np.transpose(result)
    nbrs=sk.KNeighborsClassifier(n_neighbors=3)
    nbrs.fit(result,training_class)
    accuracy=nbrs.predict(np.transpose(np.real(test_result)))
    accuracy_rate.append(mt.accuracy_score(accuracy,test_class)*100)
    print(accuracy.shape)
    if p[k]==10:
        for i in range(len(accuracy)):
            for j in range(10):
                csv_print+=str(test_result[j,i])+","
            csv_print+=str(test_class.iloc[i])+","+str(accuracy[i])+"\n"
print(accuracy_rate)
print(csv_print)
plt.plot(p,accuracy_rate)
plt.xlabel("Principal Components")
plt.ylabel("Accuracy")

plt.show()
