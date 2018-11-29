"""
Created on Sun Jun 17 08:57:04 2018

@author: Sidneaux
K-Means 
"""

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('mall.csv')
X = dataset.iloc[:, [3,4]].values

#using the elbow method to find optimal number of clusters
from sklearn.clusters import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter=300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title ('the elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#Applying k-means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter=300, n_init = 10, random_state = 0)
y_kmeans.fit_predict(X)

#Visualizing the clusters
plt.scatter(x[y_kmean==0, 0], x[y_kmeans == 0, 1], size = 100, c = 'red',label = 'Cluster 1')
plt.scatter(x[y_kmean==1, 0], x[y_kmeans == 1, 1], size = 100, c = 'blue',label = 'Cluster 2')
plt.scatter(x[y_kmean==2, 0], x[y_kmeans == 2, 1], size = 100, c = 'green',label = 'Cluster 3')
plt.scatter(x[y_kmean==3, 0], x[y_kmeans == 3, 1], size = 100, c = 'cyan',label = 'Cluster 4')
plt.scatter(x[y_kmean==4, 0], x[y_kmeans == 4, 1], size = 100, c = 'magenta',label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], size = 300, c = 'yellow',label = 'Centroids')
plt.title('Cluster of clients')
plt.xlabel('Annual Income (K$)')
plt.ylabel('Spending Score (K$)')
plt.legend()
plt.show()



