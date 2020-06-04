import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from sklearn.linear_model import LinearRegression,LogisticRegression
import matplotlib as mpl

import seaborn as sns
data  =make_blobs(n_samples=1000,centers=[[0,0],[10,10]],n_features=2)
noise = make_blobs(n_samples=500,centers=[[6,-6]],n_features=2)
model = LogisticRegression()
model.fit(data[0],data[1])


def get_z(X,Y):
    result = np.ones(shape=X.shape)
    for i1 in range(X.shape[0]):
        for i2 in range(X.shape[1]):
            result[i1,i2] = model.predict_proba([[X[i1,i2],Y[i1,i2]]])[0][1]
            # result[i1,i2] = model.predict([[X[i1,i2],Y[i1,i2]]])[0]
    return result

x = np.arange(-3,13,0.1)
y = np.arange(-3,13,0.1)
X,Y = np.meshgrid(x,y)
z = get_z(X,Y)
N = np.arange(-0.2, 1.5, 0.1)
# plt.imshow(z)
sns.heatmap(z, cmap='Reds')
# CS=plt.contour(z,[0,0.5,1],cmap=mpl.cm.jet)
plt.show()

























