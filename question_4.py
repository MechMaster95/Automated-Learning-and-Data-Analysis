# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
## 3rd question
## (a) 
import numpy as np
a=np.identity(5)
a
## (b)
a[:,1]=3*np.ones(5)
a
x,y=a.shape
x,y
## (c)
sum=0
for i in range(1,x):
    for j in range(1,y): 
        sum=sum+a[i,j]
        
sum
#for i in matrix_a:
    sum+=np.sum(i)

## (d)
At=np.transpose(a)
##(e)
## sum of the third row
z=np.sum(a[3,:])
## sum of diagonal elements
np.trance(matrix_a)
k=1
z=0
for k in range(1,x):
    z=z+a[k,k]
z
####part(i)
### covariance of the matrices x,y,z
##Degrees of freedom
## np.stack
X=[2,4,6,8]
Y=[6,5,4,3]
Z=[1,3,5,7]
c=np.stack((X,Y,Z))
print(np.cov(c))

#####part(j)
x=np.arange(2,22,2)
a=np.mean(np.square(x))
b=np.square(np.mean(x))
c=np.var(x)
#### we can see that a=b+c

###missing part f,g and h
#part(f)
fB=3*np.random.rand(5,5)+5
print(fB)
##part(g)
fC=fB
fC[0,:]=fB[0,:]*fB[1,:]
fC[1,:]=(fB[2,:]+fB[3,:])-fB[4,:]
fC[1,:]
##part(h)
fD=np.identity(5)
fD
for i in range(0,5):
    fD[:,i]=(i+2)*fC[:,i]
print(fD)

