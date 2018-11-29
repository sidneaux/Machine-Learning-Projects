"""
Created on Sun Jun 17 19:10:10 2018

@author: Sidneaux
"""

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset = pd.read_csv('mall.csv')
X = dataset.iloc[:, [3,4]].values
#Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()

#Fitting hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(N_clusters = 2, affinity='euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

#Visualizing the clusters
plt.scatter(x[y_hc==0, 0], x[y_kmeans == 0, 1], size = 100, c = 'red',label = 'Cluster 1')
plt.scatter(x[y_hc==1, 0], x[y_kmeans == 1, 1], size = 100, c = 'blue',label = 'Cluster 2')
plt.scatter(x[y_hc==2, 0], x[y_kmeans == 2, 1], size = 100, c = 'green',label = 'Cluster 3')
plt.scatter(x[y_hc==3, 0], x[y_kmeans == 3, 1], size = 100, c = 'cyan',label = 'Cluster 4')
plt.scatter(x[y_hc==4, 0], x[y_kmeans == 4, 1], size = 100, c = 'magenta',label = 'Cluster 5')
plt.title('Cluster of clients')
plt.xlabel('Annual Income (K$)')
plt.ylabel('Spending Score (K$)')
plt.legend()
plt.show()