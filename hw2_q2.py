import pandas as pd
import math
"""
To calculate the entropy of continuous variables
"""

def entropy(n,p):
    if n==0 and p==0:
        return 0
    elif n==0:
        print(-float(p/(n+p))*math.log((p/(n+p)),2))
        return -float(p/(n+p))*math.log((p/(n+p)),2)
    elif p==0:
        print(-float(n / (n + p)) * math.log((n / (n + p)), 2))
        return -float(n / (n + p)) * math.log((n / (n + p)), 2)
    else:
        print(-(n/(n+p))*math.log((n/(n+p)),2)-(p/(n+p))*math.log((p/(n+p)),2))
        return -(n/(n+p))*math.log((n/(n+p)),2)-(p/(n+p))*math.log((p/(n+p)),2)
dataset=pd.read_csv("./Data/hw2q2.csv")

class_f=class_t=0
for i in range(int(len(dataset)/2)):
    if dataset.iloc[i][5]=="F":
        class_f+=1
    else: class_t+=1
print(class_t," ",class_f)
entrpy=entropy(class_f,class_t)
print(entrpy)

max=0
which_id=0
for i in range(int(len(dataset)/2)):
    id=dataset.iloc[i][0]
    class1_f=class1_t=class2_f=class2_t=0
    for j in range(int(len(dataset)/2)):
        if int(dataset.iloc[j][0])<id and dataset.iloc[j][5]=="F":
            class1_f+=1
        elif int(dataset.iloc[j][0])<id and dataset.iloc[j][5]=="T":
            class1_t+=1
        elif int(dataset.iloc[j][0]) >= id and dataset.iloc[j][5] == "F":
            class2_f += 1
        elif int(dataset.iloc[j][0]) >= id and dataset.iloc[j][5] == "T":
            class2_t += 1
    print(class1_t,class1_f,class2_t,class2_f)
    print("id",int(dataset.iloc[i][0]))
    IG=entrpy-((class1_t+class1_f)*entropy(class1_t,class1_f)+(class2_t+class2_f)*entropy(class2_f,class2_t))/(int(len(dataset)/2))
    print(IG)
    if IG>max:
        max=IG
        which_id=int(dataset.iloc[i][0])
print(which_id)
