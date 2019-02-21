from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
from scipy.spatial import distance_matrix

x1 = np.array([1,1,2,3,4,4,6,7,9,9])
x2 = np.array([8,1,4,3,9,6,4,7,9,1])

data = [[1,8], [1, 1], [2, 4], [3, 3], [4, 9], [4, 6], [6, 4], [7, 7], [9, 9], [9, 1]]
points = ['A', 'B', 'C','D','E', 'F', 'G', 'H','I', 'J']
df = pd.DataFrame(data, columns=['xcord', 'ycord'], index=points)
final_df = pd.DataFrame(distance_matrix(df.values, df.values, p=2), index=df.index, columns=df.index)
final_ = final_df.round(2)
print(final_)
plt.plot()
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('Dataset')
plt.scatter(x1, x2)
plt.show()

# create new plot and data
plt.plot()
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
colors = ['b', 'g', 'r']
markers = ['o', 'v', 's']

# k means determine k
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()



# generate the linkage matrix - Single
Z = linkage(final_ , 'single')

# set cut-off to 150
#max_d = 7.08                # max_d as in max_distance
max_d = 9
plt.figure(figsize=(15, 8))
plt.title('Hierarchical Clustering Dendrogram - Single Linkage')
plt.xlabel('x')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='none',  # show only the last p merged clusters
    p=150,                  # Try changing values of p
    leaf_rotation=90.,      # rotates the x axis labels
    leaf_font_size=15.,      # font size for the x axis labels
    labels = points
)
plt.axhline(y=max_d, c='k')
plt.show()



# generate the linkage matrix - Complete
Z = linkage(final_ , 'complete')

# set cut-off to 150
#max_d = 7.08                # max_d as in max_distance
max_d = 15.0
plt.figure(figsize=(15, 8))
plt.title('Hierarchical Clustering Dendrogram - Complete Linkage')
plt.xlabel('x')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='none',  # show only the last p merged clusters
    p=150,                  # Try changing values of p
    leaf_rotation=90.,      # rotates the x axis labels
    leaf_font_size=15.,      # font size for the x axis labels
    labels = points
)
plt.axhline(y=max_d, c='k')
plt.show()