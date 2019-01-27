# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"F:\Udemy\Machine Learning\Machine Learning A-Z Template Folder\Part 4 - Clustering\Section 24 - K-Means Clustering\Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42, max_iter=300, n_init=10)
y_means = kmeans.fit_predict(X)


plt.scatter(X[y_means==0,0],X[y_means==0,1],s=100, c='red', label='C1')
plt.scatter(X[y_means==1,0],X[y_means==1,1],s=100, c='blue', label='C2')
plt.scatter(X[y_means==2,0],X[y_means==2,1],s=100, c='green', label='C3')
plt.scatter(X[y_means==3,0],X[y_means==3,1],s=100, c='cyan', label='C4')
plt.scatter(X[y_means==4,0],X[y_means==4,1],s=100, c='magenta', label='C5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.legend()
plt.show()
