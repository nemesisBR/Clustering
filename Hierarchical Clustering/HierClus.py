# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"F:\Udemy\Machine Learning\Machine Learning A-Z Template Folder\Part 4 - Clustering\Section 25 - Hierarchical Clustering\Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values

# Using Dendogram

import scipy.cluster.hierarchy as sch

dendogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.title('Ddgram')
plt.xlabel('Customer')
plt.ylabel('Distance')
plt.show()

#fitting hierarchial

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_hc = hc.fit_predict(X)

#Visualising

plt.scatter(X[y_hc==0,0],X[y_hc==0,1],s=100, c='red', label='C1')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100, c='blue', label='C2')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100, c='green', label='C3')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100, c='cyan', label='C4')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100, c='magenta', label='C5')
plt.legend()
plt.show()
plt.scatter