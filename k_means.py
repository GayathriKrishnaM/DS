import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data=pd.read_csv('/home/student/Desktop/iris.csv')
print(data.head())

x=data.iloc[:,:4]
print(x.head())

km=KMeans(n_clusters=3,n_init=10)
km.fit(x)
y=km.predict(x)
print(y)
centroid=km.cluster_centers_
print(centroid)