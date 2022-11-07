# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 2022
@author: pnguyen340 (modified from jtay)
"""

# %% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# from helpers import nn_arch, nn_reg
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import FastICA

out = './Output/ICA/'
out1 = './Output/ICA/ICA/'
f_path = './Output/BASE/'

np.random.seed(9)
nn_reg = [(10 ** -3)]
nn_arch = [(14,)]
nn_arch2 = [(14,)]

# Read in data
# Credit Card Fraud dataset
ccf_train = pd.read_hdf(f_path + 'datasets.hdf', 'train_ccf')
ccf_test = pd.read_hdf(f_path + 'datasets.hdf', 'test_ccf')

ccfX_train = ccf_train.drop('labels', 1).copy().values
ccfY_train = ccf_train['labels'].copy().values
ccfX_test = ccf_test.drop('labels', 1).copy().values
ccfY_test = ccf_test['labels'].copy().values

# Breast Cancer dataset
bc_train = pd.read_hdf(f_path + 'datasets.hdf', 'train_bc')
bc_test = pd.read_hdf(f_path + 'datasets.hdf', 'test_bc')

bcX_train = bc_train.drop(['diagnosis'], 1).copy().values
bcY_train = bc_train['diagnosis'].copy().values
bcX_test = bc_test.drop(['diagnosis'], 1).copy().values
bcY_test = bc_test['diagnosis'].copy().values

ccfX_train = StandardScaler().fit_transform(ccfX_train)
bcX_train = StandardScaler().fit_transform(bcX_train)
ccfX_test = StandardScaler().fit_transform(ccfX_test)
bcX_test = StandardScaler().fit_transform(bcX_test)

clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
dims = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15]

# %% data for 1
ica = FastICA(random_state=4)
kurt = {}
for dim in dims:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(ccfX_train)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt)
kurt.to_csv(out1 + 'ccf scree.csv')

ica = FastICA(random_state=4)
kurt = {}
for dim in dims:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(bcX_train)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt)
kurt.to_csv(out1 + 'bc scree.csv')

# %% Data for 2
grid = {'ica__n_components': dims, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
ica = FastICA(random_state=4)
mlp = MLPClassifier(activation='relu', max_iter=200, early_stopping=True, random_state=5, learning_rate_init=0.1, momentum=0.3)
pipe = Pipeline([('ica', ica), ('NN', mlp)])
gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

gs.fit(ccfX_train, ccfY_train)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out1 + 'ccf dim red.csv')

grid = {'ica__n_components': dims, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch2}
ica = FastICA(random_state=4)
mlp = MLPClassifier(activation='relu', max_iter=200, early_stopping=True, random_state=5, learning_rate_init=0.1, momentum=0.3)
pipe = Pipeline([('ica', ica), ('NN', mlp)])
gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

gs.fit(bcX_train, bcY_train)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out1 + 'bc dim red.csv')

# %% data for 3
# Set this from chart 2 and dump, use clustering script to finish up
dim = 3     # 4
for ccfX, ccfY, key_name in zip([ccfX_train, ccfX_test], [ccfY_train, ccfY_test], ["train_ccf", "test_ccf"]):
    ica = FastICA(n_components=dim, random_state=8)
    ccfX2 = ica.fit_transform(ccfX)
    ccf2 = pd.DataFrame(np.hstack((ccfX2, np.atleast_2d(ccfY).T)))
    cols = list(range(ccf2.shape[1]))
    cols[-1] = 'labels'
    ccf2.columns = cols
    ccf2.to_hdf(out + 'datasets.hdf', key_name, complib='blosc', complevel=9)

dim = 15
for bcX, bcY, key_name in zip([bcX_train, bcX_test], [bcY_train, bcY_test], ["train_bc", "test_bc"]):
    ica = FastICA(n_components=dim, random_state=8)
    bcX2 = ica.fit_transform(bcX)
    bc2 = pd.DataFrame(np.hstack((bcX2, np.atleast_2d(bcY).T)))
    cols = list(range(bc2.shape[1]))
    cols[-1] = 'diagnosis'
    bc2.columns = cols
    bc2.to_hdf(out + 'datasets.hdf', key_name, complib='blosc', complevel=9)
