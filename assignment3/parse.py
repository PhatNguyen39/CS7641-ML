# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 10:39:27 2017

@author: jtay
"""

import pandas as pd
import os


# Make directories if not created
for d in ['BASE', 'RP', 'PCA', 'ICA', 'RF']:
    n = './Output/{}/{}/'.format(d, d)
    if not os.path.exists(n):
        os.makedirs(n)

# Set output folder
OUT = './Output/BASE/'

# Convert datasets into training and test into HDF output ##

# For Breast Cancer dataset
train_bc = pd.read_csv('./Datasets/train_bc.csv')
test_bc = pd.read_csv('./Datasets/test_bc.csv')
train_bc = train_bc.drop(['id'], 1)
test_bc = test_bc.drop(['id'], 1)

train_bc.to_hdf(OUT + 'datasets.hdf', 'train_bc', complib='blosc', complevel=9)
test_bc.to_hdf(OUT + 'datasets.hdf', 'test_bc', complib='blosc', complevel=9)

# For Credit Card Fraud dataset
train_f = pd.read_csv('./Datasets/train_ccf.csv', header=None)
test_f = pd.read_csv('./Datasets/test_ccf.csv', header=None)

template = "feature_{}"
column_names = [template.format(i) for i in range(30)]
column_names.append("labels")

train_f.columns = column_names
test_f.columns = column_names

train_f.loc[train_f["labels"] == -1, "labels"] = 0
test_f.loc[test_f["labels"] == -1, "labels"] = 0

train_f.to_hdf(OUT + 'datasets.hdf', 'train_ccf', complib='blosc', complevel=9)
test_f.to_hdf(OUT + 'datasets.hdf', 'test_ccf', complib='blosc', complevel=9)

