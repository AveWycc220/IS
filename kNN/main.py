import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

""" CONSTS """
PATH = os.path.dirname(os.path.abspath(__file__)) + '\\'

input_data = pd.read_table(PATH + 'data\\Credit_Screening.dat',  sep=';', encoding='utf-8')
X = input_data.iloc[:, 0:46].values
y = np.array(list(map(lambda x: x[0], input_data.iloc[:, 47:48].values)))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
knn = KNeighborsClassifier(n_neighbors=39, p=2, metric='minkowski')
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
cmat = confusion_matrix(y_test, y_pred)
sns.set(font_scale=1.4)
sns.heatmap(cmat, annot=True, fmt="d")
plt.show()
print(classification_report(y_test, y_pred))
acc = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    acc.append(knn.score(X_test, y_test))
plt.figure(figsize=(10, 4))
plt.plot(range(1, 40), acc, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Accuracy vs. K-Values')
plt.xlabel('K-Values')
plt.ylabel('Accuracy')
plt.show()