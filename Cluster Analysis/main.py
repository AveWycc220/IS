import os
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


""" CONSTS """
PATH = os.path.dirname(os.path.abspath(__file__)) + '\\'


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


def value_sex(input_data, cluster):
    input_data = input_data.drop(columns=['ALTER', 'HERKU', 'FB'])
    data = pd.DataFrame(preprocessing.scale(input_data), columns=['SEX', 'VALUE'])
    elbow_method(data)
    clusters = cluster
    plt.xlabel('VALUE')
    plt.ylabel('SEX')
    plt.scatter(data['VALUE'], data['SEX'])
    plt.show()
    km = KMeans(n_clusters=clusters)
    data_clusters = km.fit_predict(data)
    data = np.array(data)
    plt.xlabel('VALUE')
    plt.ylabel('SEX')
    input_data = np.array(input_data)
    centr = km.cluster_centers_
    mean_of_array = input_data.mean(axis=0)
    std_of_array = input_data.std(axis=0)
    centr = (centr * std_of_array) + mean_of_array
    for i in range(0, clusters):
        plt.scatter(input_data[data_clusters == i, 1], input_data[data_clusters == i, 0])
        plt.scatter(centr[i, 1], centr[i, 0], marker='*', c=list(mcolors.TABLEAU_COLORS.values())[i], linewidths=5)
    plt.show()


def value_alter(input_data, cluster):
    input_data = input_data.drop(columns=['SEX', 'HERKU', 'FB'])
    data = pd.DataFrame(preprocessing.scale(input_data), columns=['ALTER', 'VALUE'])
    elbow_method(data)
    clusters = cluster
    plt.xlabel('VALUE')
    plt.ylabel('ALTER')
    plt.scatter(input_data['VALUE'], input_data['ALTER'])
    plt.show()
    km = KMeans(n_clusters=clusters)
    data_clusters = km.fit_predict(data)
    input_data = np.array(input_data)
    centr = km.cluster_centers_
    mean_of_array = input_data.mean(axis=0)
    std_of_array = input_data.std(axis=0)
    centr = (centr * std_of_array) + mean_of_array
    plt.xlabel('VALUE')
    plt.ylabel('ALTER')
    for i in range(0, clusters):
        plt.scatter(input_data[data_clusters == i, 1], input_data[data_clusters == i, 0])
        plt.scatter(centr[i, 1], centr[i, 0], marker='*', c=list(mcolors.TABLEAU_COLORS.values())[i], linewidths=5)
    plt.show()


def value_herku(input_data, cluster):
    input_data = input_data.drop(columns=['SEX', 'ALTER', 'FB'])
    data = pd.DataFrame(preprocessing.scale(input_data), columns=['HERKU', 'VALUE'])
    elbow_method(data)
    clusters = cluster
    plt.xlabel('VALUE')
    plt.ylabel('HERKU')
    plt.scatter(input_data['VALUE'], input_data['HERKU'])
    plt.show()
    km = KMeans(n_clusters=clusters)
    data_clusters = km.fit_predict(data)
    data = np.array(data)
    input_data = np.array(input_data)
    centr = km.cluster_centers_
    mean_of_array = input_data.mean(axis=0)
    std_of_array = input_data.std(axis=0)
    centr = (centr * std_of_array) + mean_of_array
    plt.xlabel('VALUE')
    plt.ylabel('HERKU')
    scatter_list = []
    for i in range(0, clusters):
        scatter_list.append(plt.scatter(input_data[data_clusters == i, 1], input_data[data_clusters == i, 0], c=list(mcolors.TABLEAU_COLORS.values())[i]))
        plt.scatter(centr[i, 1], centr[i, 0], marker='*', c=list(mcolors.TABLEAU_COLORS.values())[i], linewidths=5)
    plt.legend(scatter_list, [i for i in range(0, clusters)])
    plt.show()


def value_fb(input_data, cluster):
    input_data = input_data.drop(columns=['SEX', 'ALTER', 'HERKU'])
    data = pd.DataFrame(preprocessing.scale(input_data), columns=['FB', 'VALUE'])
    elbow_method(data)
    clusters = cluster
    plt.xlabel('VALUE')
    plt.ylabel('FB')
    plt.scatter(input_data['VALUE'], input_data['FB'])
    plt.show()
    km = KMeans(n_clusters=clusters)
    data_clusters = km.fit_predict(data)
    data = np.array(data)
    input_data = np.array(input_data)
    centr = km.cluster_centers_
    mean_of_array = input_data.mean(axis=0)
    std_of_array = input_data.std(axis=0)
    centr = (centr * std_of_array) + mean_of_array
    plt.xlabel('VALUE')
    plt.ylabel('FB')
    scatter_list = []
    for i in range(0, clusters):
        scatter_list.append(plt.scatter(input_data[data_clusters == i, 1], input_data[data_clusters == i, 0], c=list(mcolors.TABLEAU_COLORS.values())[i]))
        plt.scatter(centr[i, 1], centr[i, 0], marker='*', c=list(mcolors.TABLEAU_COLORS.values())[i], linewidths=5)
    plt.show()

def hierarchical_number_of_cluster(input_data, q):
    input_data = input_data.drop(columns=['SEX', 'ALTER', 'FB'])
    data = preprocessing.scale(input_data)
    dist = pdist(data, 'euclidean')
    link = linkage(dist, 'ward')
    dendrogram(link, p=3, truncate_mode='level')
    plt.show()
    plt.xlabel('VALUE')
    plt.ylabel('HERKU')
    input_data = np.array(input_data)
    clusters = fcluster(link, q, criterion='maxclust')
    scatter_list = []
    for i in range(0, q):
        scatter_list.append(plt.scatter(input_data[clusters == i+1, 1], input_data[clusters == i+1, 0], c=list(mcolors.TABLEAU_COLORS.values())[i]))
    plt.legend(scatter_list, [i for i in range(0, q)])
    plt.show()

def hierarchical_dist(input_data, d):
    input_data = input_data.drop(columns=['SEX', 'ALTER', 'FB'])
    data = preprocessing.scale(input_data)
    dist = pdist(data, 'euclidean')
    link = linkage(dist, 'ward')
    dendrogram(link, p=3, truncate_mode='level')
    plt.show()
    plt.xlabel('VALUE')
    plt.ylabel('HERKU')
    input_data = np.array(input_data)
    clusters = fcluster(link, d, criterion='distance')
    print('Number of Cluster = ', max(clusters))
    scatter_list = []
    for i in range(0, max(clusters)):
        scatter_list.append(plt.scatter(input_data[clusters == i+1, 1], input_data[clusters == i+1, 0]))
    plt.show()


input_data = pd.read_table(PATH + 'data\\computer.dat',  sep='\t', encoding='utf-8')
new_column = []
for i in range(0, len(input_data[input_data.columns[0]])):
    temp = 0
    for j in range(0, len(input_data.columns)):
        temp = temp + input_data[input_data.columns[j]][i]
    new_column.append(temp)
input_data['VALUE'] = new_column
input_data = input_data.drop(columns=input_data.columns[0:19])
value_fb(input_data, 5)
#hierarchical_number_of_cluster(input_data, 5)
#№ierarchical_dist(input_data, 15)