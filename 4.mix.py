from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
d1 = np.random.normal(5,3,7000)
d2 = np.random.normal(25,5,7000)
d3 = np.hstack((d1,d2))

plt.hist(d3,density=False,bins=50)

model = GaussianMixture(n_components=2)
model.fit(d3.reshape([-1,1]))


