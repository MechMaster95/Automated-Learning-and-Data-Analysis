import sys
import numpy as np
import matplotlib.pyplot as plt
#Data
data=np.array([[27,-6,2,36,-8,40,35,30,20,-1],
              [6,2,2,2,4,2,4,2,6,4],
              [0,0,1,1,0,1,0,1,1,0],
              [1,2,3,4,5,6,7,8,9,10]])
euclidean_dist=sys.maxsize
data=np.transpose(data)
###############################################################
# Wrote a function to find K neighbours
###############################################################
def findNeighbours(ID, no_of_neighbours, test_data):
    global euclidean_dist
    x_coor=data[ID,0]
    y_coor=data[ID,1]
    arr=[]
    duplicate=np.copy(test_data)
    i=0
    pos=0
    id=0
    while i<len(duplicate):
        j=0
        euclidean_dist=sys.maxsize
        test=[]
        while j<len(duplicate):
            x=(x_coor-duplicate[j,0])**2
            y=(y_coor-duplicate[j, 1])**2
            e_dist=(x+y)**0.5
            test.append(e_dist)
            if e_dist < euclidean_dist:
                euclidean_dist=e_dist
                id=duplicate[j,3]
                pos=j
            j+=1
        arr.append(id)  # List of nearest neighbours
        duplicate[pos,0]=10000
        duplicate[pos,1]=10000
        i+=1
        if len(arr)==no_of_neighbours+1:
            break
    return arr


#a
print("Points nearest to the given point")
point_5 = findNeighbours(4,3,data)  # since ID starts from 1, but indexes from 0
point_10 = findNeighbours(9,3,data)  # since ID starts from 1, but indexes from 0
point_5=point_5[1:]
point_10=point_10[1:]
print(point_5)
print(point_10)

#b-
error=[]

out1x=[]
out2x=[]
out1y=[]
out2y=[]
for i in range(len(data)):
    if data[i, 2] == 1:
        out1x.append(data[i,0])
        out1y.append(data[i, 1])
    else:
        out2x.append(data[i, 0])
        out2y.append(data[i, 1])
    print(out1x)
print(out1x)
plt.scatter(out1x,out1y,c='b')
plt.scatter(out2x,out2y,c='r')
plt.xlim(-10,40)
plt.ylim(0,50)
plt.title("x1 vs x2")
plt.show()
for i in range(len(data)):
    id=findNeighbours(i,1,np.concatenate((data[0:i],data[i+1:]),axis=0))
    id=id[0]
    print(id)
    if data[i,2]!=data[id-1,2]:
        error.append(1)
    else: error.append(0)
print("Error array")
print(error)
error_rate=sum(error)/len(error)
print("Error rate",error_rate)


#c
folds=5
mse_sum=0
for i in range(0,folds):
    test=[]
    # Point 1= (4+i)%5
    # Point 2= 5+(4+i)%5
    duplicate=np.delete(data,(4+i)%5,0)
    duplicate = np.delete(duplicate, 4+(4 + i) % 5, 0)
    NN3_1=findNeighbours((4+i)%5,3,duplicate)
    NN3_2=findNeighbours(4+(4+i)%5,3,data)
    print("Test data tuple1 ID: ",1+ (4 + i) % 5)
    print("3NN:", NN3_1[:3])
    print("Test data tuple1 ID: ", 6 + (4 + i) % 5)
    print("3NN:", NN3_2[:3])
    sum_classes=0
    fold_sum=0
    for k in range(3):
        sum_classes+=data[NN3_2[k]-1,2]
    if sum_classes>=2 and data[(4+i)%5,2]!=1:
        fold_sum+=1
    elif sum_classes<2 and data[(4+i)%5,2]==1:
        fold_sum+=1
    print(fold_sum)
    mse_sum+=fold_sum/2 #Misclassification error in 1 fold
mse_average=mse_sum/folds
print(mse_average)

#d
# 5-fold cross validation with 3NN gave much lesser error than LOOCV with 1-NN
# This is because 3-NN considers classes of three nearest neighbours and takes majority of the class
# But in 1-NN the nearest neighbour could be another class datapoint which is very near to the test data point
# 3-NN considers the vicinity of the test point, hence reducing the probability of an outlier influencing the class label
