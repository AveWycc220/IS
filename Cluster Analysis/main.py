import os
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


""" CONSTS """
PATH = os.path.dirname(os.path.abspath(__file__)) + '\\'


data = pd.read_table(PATH + 'data\\computer.dat',  sep='\t', encoding='utf-8')
data_scale = pd.DataFrame(preprocessing.scale(data), columns=data.columns)
dist_data = pd.DataFrame({'Index': data.columns}, index=data.columns, columns=data.columns)
for i in range(0, len(data.columns)):
    for j in range(0, len(data.columns)):
        dist_data.iloc[:, j][i] = np.sum([(a-b)*(a-b) for a, b in zip(data_scale.iloc[:, j], data_scale.iloc[:, i])])
dist_data.to_excel(PATH + 'data\\dist.xls', columns=dist_data.columns)
km = KMeans(n_clusters=2, init='random')
data_km = km.fit_predict(pd.DataFrame(data_scale))
print(data_km)

plt.scatter([i for i in range(len(data_km))], data_km, c='red')
plt.grid()
plt.show()
