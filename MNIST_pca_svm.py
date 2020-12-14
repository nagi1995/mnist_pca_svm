# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 11:00:26 2020

@author: Nagesh
"""
#%%

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import pickle


#%%


df = pd.read_csv("mnist_data.csv", delimiter = ',')
xy = np.array(df, dtype = 'float')
x, y = xy[:, :-1], xy[:, -1]

#%%

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.99, random_state = 4)

#%%
# checking whether all hand written numbers are present in training set or not
y_train_df = pd.Series(y_train)
print(np.sort(pd.unique(y_train_df))) # This statement should print all numbers from 0 - 9.

#%%

pca = PCA()
pca.fit(x_train)
plt.plot(pca.explained_variance_ratio_.cumsum())

#%%

print(np.where(pca.explained_variance_ratio_.cumsum() > 0.9))
print(np.where(pca.explained_variance_ratio_.cumsum() > 0.95))
print(np.where(pca.explained_variance_ratio_.cumsum() > 0.99))

#%%

param_grid = {"n_components" : [74, 123, 255]}
pca = PCA()
gs = GridSearchCV(estimator = pca, param_grid = param_grid)
gs.fit(x_train, y_train)

print(gs.best_params_)

#%%

pca = PCA(n_components = 123).fit(x_train)

#%%

with open('pca_model.pkl', 'wb') as f:
    pickle.dump(pca, f)

#%%

x_train_pca = pca.transform(x_train)

#%%

model = SVC(kernel = 'rbf').fit(x_train_pca, y_train)

#%%


with open('mnist_pca_svm.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('mnist_pca_svm.pkl', 'rb') as f:
    model = pickle.load(f)

#%%

x_test_pca = pca.transform(x_test)
y_pred = model.predict(x_test_pca)
print(accuracy_score(y_test, y_pred))







