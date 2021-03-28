import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize

""" CONSTS """
PATH = os.path.dirname(os.path.abspath(__file__)) + '\\'

data = pd.read_csv(PATH + 'data\\milk.csv')
for i in range(0, 15):
    plt.scatter([j+(12*i) for j in range(0, len(data['milk'][i*12:12+12*i]))], data['milk'][i*12:12+12*i])
plt.xlabel('MONTH')
plt.ylabel('MILK')
plt.show()
x = np.array([i for i in range(0, len(data['milk']))]).reshape((-1, 1))
norm_data = normalize(np.array(data['milk']).reshape((1, -1)))
lr = LinearRegression().fit(x, norm_data[0])
x_predict = np.array([i for i in range(len(data['milk']), len(data['milk'])+24)]).reshape((-1, 1))
y_predict = lr.predict(x_predict)
for i in range(0, 15):
    plt.scatter([j+(12*i) for j in range(0, len(norm_data[0][i*12:12+12*i]))], norm_data[0][i*12:12+12*i])
plt.scatter(x_predict, y_predict, marker='X')
plt.show()