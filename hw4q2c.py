import pandas as pd
from scipy.spatial import distance_matrix

data = [[1,8], [1, 1], [2, 4], [3, 3], [4, 9], [4, 6], [6, 4], [7, 7], [9, 9], [9, 1]]
points = ['A', 'B', 'C','D','E', 'F', 'G', 'H','I', 'J']

df = pd.DataFrame(data, columns=['xcord', 'ycord'], index=points)
#Hierarchical (Single)
c1 = df.iloc[8:10,]
c2 = df.iloc[1:4,]
c3 = df.iloc[[0,4,5,6,7],]

#K-Means AEF BCD GHIJ
#c1 = df.iloc[6:10,]
#c2 = df.iloc[1:4,]
#c3 = df.iloc[[0,4,5],]



#WSS
def WSS(c1, c2, c3): # WSS same as SSE
    def each_cluster(c):
        cMean = c.mean()
        c_mean = pd.DataFrame([[cMean[0],cMean[1]]], columns=['xcord', 'ycord'])  
        print("cluster mean:", c_mean)
        sum_c = 0
        for i in range(len(c)):
            sum_c += (distance_matrix(c.iloc[i:i+1,], c_mean, p=2)**2)
            print("Value " )
            print(c.iloc[i,])
            print(" wss for: " )
            print("Cluster ")
            print(c)
            print(" is  ")
            
        return(sum_c)
    sum_c1 = each_cluster(c1)
    sum_c2 = each_cluster(c2)
    sum_c3 = each_cluster(c3)  
    print(sum_c1)
    print(sum_c2)
    print(sum_c3)
    return(sum_c1+sum_c2+sum_c3)
wss = WSS(c1, c2, c3)
print("Total wss is ")
print(wss)
            
        
        
#BSS    
df_mean = df.mean()
M = pd.DataFrame([[df_mean[0],df_mean[1]]], columns=['xcord', 'ycord']) 
def BSS(c1, c2, c3):
    
    def each_cluster(c):
        cMean = c.mean()
        c_mean = pd.DataFrame([[cMean[0],cMean[1]]], columns=['xcord', 'ycord']) 
        print("larger mean: " )
        print(M)
        print("cluster mean:", c_mean)
        sum_c = len(c)*(distance_matrix(M, c_mean, p=2)**2)
        return(sum_c)
    sum_c1 = each_cluster(c1)
    sum_c2 = each_cluster(c2)
    sum_c3 = each_cluster(c3)        
    return(sum_c1+sum_c2+sum_c3)
bss = BSS(c1, c2, c3)
print("Total bss is ")
print(bss)      
    
        
    
    