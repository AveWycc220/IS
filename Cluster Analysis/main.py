import os
import pandas as pd
from sklearn import preprocessing
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


""" CONSTS """
PATH = os.path.dirname(os.path.abspath(__file__)) + '\\'


def plot_scatter(data, title=None, cluster=None):
    x = [i for i in range(0, len(data))]
    plt.title(f'{title}') if title else plt.title('')
    plots, names = [], []
    for i in range(0, len(data.columns)):
        plots.append(plt.scatter(x, data[data.columns[i]]))
        names.append(f'{data.columns[i]}')
    if cluster is not None:
        plots.append(plt.scatter(x, cluster))
        names.append('Cluster')
    plt.legend(plots, names)
    plt.show()


def drop_vars(data_drop, type_columns: int = 1, row_drop: int = 0):
    if row_drop != 0 and row_drop < len(data_drop.index):
        data_drop = data_drop.drop(data_drop.index[0: row_drop])
    if type_columns == 1:
        return data_drop.drop(columns=data_drop.columns[3:len(data_drop.columns) - 1])
    elif type_columns == 2:
        return data_drop.drop(columns=[*data_drop.columns[0:3], *data_drop.columns[9:len(data_drop.columns) - 1]])
    elif type_columns == 3:
        return data_drop.drop(columns=[*data_drop.columns[0:9], *data_drop.columns[12:len(data_drop.columns) - 1]])
    elif type_columns == 4:
        return data_drop.drop(columns=[*data_drop.columns[0:12], *data_drop.columns[17:len(data_drop.columns) - 1]])
    elif type_columns == 5:
        return data_drop.drop(columns=data_drop.columns[0:17])
    else:
        return data_drop


def elbow_method(data_elbow):
    distortions = []
    for i in range(1, 20):
        km_eblow = KMeans(n_clusters=i,
                    n_init=10,
                    max_iter=300,
                    random_state=0)
        km_eblow.fit(data_elbow)
        distortions.append(km_eblow.inertia_)
    plt.plot(range(1, 20), distortions, marker='o')
    plt.title('Eblow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distortion')
    plt.show()


input_data = pd.read_table(PATH + 'data\\computer.dat',  sep='\t', encoding='utf-8')
input_data = drop_vars(input_data, 1)
plot_scatter(input_data, 'Input Data')
data_scaled = pd.DataFrame(preprocessing.scale(input_data), columns=input_data.columns)
plot_scatter(data_scaled, 'Standardized Data')
elbow_method(data_scaled)

#km = KMeans(n_clusters=8)
#data_clusters = pd.DataFrame([*km.fit_predict(data_scaled)])
#plot_scatter(data_scaled, 'Input Data with Clusters', cluster=data_clusters)