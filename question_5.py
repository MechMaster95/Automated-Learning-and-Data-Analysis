# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 14:22:38 2018

@author: Nadim
"""
## importing and filtering to two columns
import pandas as pd
import numpy as np
filename = 'seeds.csv'
data= pd.read_csv(filename)
data1= data[['area_A','kernel_width']]
data1.shape
b=np.sum(data1)
b
### Normalised_Data
np.max(data1)
np.mean(data1)
data_norm=(data1-np.min(data1))/(np.max(data1)-np.min(data1))
data_norm
##Standardised_Data
data_std=(data1-np.mean(data1))/(np.std(data1))
data_std
np.max(data_norm)-np.min(data_norm)
np.max(data_std)-np.min(data_std)
###PartB(i)
import matplotlib.pyplot as plt
##Raw_Data
plt.xlabel("area_A")
plt.ylabel("kernel_width")
plt.scatter(data1['area_A'],data1['kernel_width'])
plt.show()
## normalised_data
plt.xlabel("area_A")
plt.ylabel("kernel_width")
plt.scatter(data_norm['area_A'],data_norm['kernel_width'])
plt.show()

##standardised 
plt.xlabel("area_A")
plt.ylabel("kernel_width")
plt.scatter(data_std['area_A'],data_std['kernel_width'])
plt.show()

##partb(ii)
x=np.mean(data1)
print(x[0])
y=np.mean(data_norm)
z=np.mean(data_std)
##part(iii)
plt.scatter(data1['area_A'],data1['kernel_width'],c="black",1,2,c="red")
plt.show()




