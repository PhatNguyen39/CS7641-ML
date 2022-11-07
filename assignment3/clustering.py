# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 2022
@author: pnguyen340 (modified from jtay)
"""

import sys
# from time import clock
import time
from collections import defaultdict

import numpy as np
# %% Imports
import pandas as pd
from sklearn.cluster import KMeans as kmeans
from sklearn.metrics import normalized_mutual_info_score as ami
from sklearn.metrics import silhouette_score as ss
from sklearn.mixture import GaussianMixture as GMM
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from helpers import cluster_adjusted_rand


# sys.argv[1] represents the first command-line argument (as a string) supplied to the script in question.
out = './Output/{}/'.format(sys.argv[1])

# Set hidden layer neurons based on dimensions of results
arg = sys.argv[1]

np.random.seed(9)

# Read in data
# ccf dataset
ccf = pd.read_hdf(out + 'datasets.hdf', 'train_ccf')
ccfX = ccf.drop('labels', 1).copy().values
ccfY = ccf['labels'].copy().values

# bc dataset
bc = pd.read_hdf(out + 'datasets.hdf', 'train_bc')
bcX = bc.drop(['diagnosis'], 1).copy().values
bc.loc[bc["diagnosis"] == "B", "diagnosis"] = 0
bc.loc[bc["diagnosis"] == "M", "diagnosis"] = 1
bcY = bc['diagnosis'].copy().values.astype('int')

ccfX = StandardScaler().fit_transform(ccfX)
bcX = StandardScaler().fit_transform(bcX)


'''
    Dummy classifier to validate reduced datasets
'''
# Dataframe ccftrain with label
ccf_X_train, ccf_X_test, ccf_y_train, ccf_y_test = train_test_split(ccfX, ccfY, test_size=0.2, random_state=42)
ccf_DT = DecisionTreeClassifier(random_state=42)  # max_depth =3
st = time.time()
ccf_DT.fit(ccf_X_train, ccf_y_train)
ccf_fit_time = time.time() - st

ccf_DT_score = ccf_DT.score(ccf_X_test, ccf_y_test)
# ccf_X_predict = ccf_DT.predict(ccf_X_test)

with open("Dummy Classifier note.txt", "a") as file:
    file.write("\n")
    file.write("\n")
    file.write("Dummy Decision Tree Classifier score of {} Credit Card dataset: {:.3}".format(arg, ccf_DT_score))
    file.write("\n")
    file.write("Dummy Decision Tree Classifier fit time of {} Credit Card dataset: {:.3}".format(arg, ccf_fit_time))
    file.write("\n")

bc_X_train, bc_X_test, bc_y_train, bc_y_test = train_test_split(bcX, bcY, test_size=0.2, random_state=42)
bc_DT = DecisionTreeClassifier(random_state=42)  # max_depth =3
st = time.time()
bc_DT.fit(bc_X_train, bc_y_train)
bc_fit_time = time.time() - st

bc_DT_score = bc_DT.score(bc_X_test, bc_y_test)
with open("Dummy Classifier note.txt", "a") as file:
    file.write("Dummy Decision Tree Classifier score of {} Breast Cancer dataset: {:.3} ".format(arg, bc_DT_score))
    file.write("\n")
    file.write("Dummy Decision Tree Classifier fit time of {} Breast Cancer dataset: {:.3}".format(arg, bc_fit_time))
    file.write("\n")

clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]

# Data for 1-3
SSE = defaultdict(dict)
ll = defaultdict(dict)  # log likelihood
SS = defaultdict(lambda: defaultdict(dict))  # silhouette score
acc = defaultdict(lambda: defaultdict(dict))  # TODO change name acc
adjMI = defaultdict(lambda: defaultdict(dict))
km = kmeans(random_state=5)  # TODO change random_state  #5
gmm = GMM(random_state=5)  # 5

# st = clock()
st = time.time()
fit_time_dict = {"k": [], "BC_kmean": [], "BC_em": [], "CCF_kmean": [], "CCF_em": []}
fit_time = pd.DataFrame()

for k in clusters:
    km.set_params(n_clusters=k)
    gmm.set_params(n_components=k)

    start = time.time()
    km.fit(ccfX)
    km_fit_time = time.time() - start

    start = time.time()
    gmm.fit(ccfX)
    gmm_fit_time = time.time() - start

    # Credit Card Fraud dataset
    # Visual Measurements
    # fit_time[k]['CreditCardF_km'] = km_fit_time
    # fit_time[k]['CreditCardF_gmm'] = gmm_fit_time
    fit_time_dict["k"].append(k)
    fit_time_dict["CCF_kmean"].append(km_fit_time)
    fit_time_dict["CCF_em"].append(gmm_fit_time)

    # Sum of Squared Errors for K-means
    SSE[k]['CreditCardF'] = km.score(ccfX)

    # Log-Likelihood for GMM
    ll[k]['CreditCardF'] = gmm.score(ccfX)

    # Silhouette Score
    # The best value is 1 and the worst value is -1.
    # Silhouette analysis can be used to study the separation distance between the resulting clusters.
    SS[k]['CreditCardF']['Kmeans'] = ss(ccfX, km.predict(ccfX))
    SS[k]['CreditCardF']['GMM'] = ss(ccfX, gmm.predict(ccfX))

    # Cluster adjusted rand score
    acc[k]['CreditCardF']['Kmeans'] = cluster_adjusted_rand(ccfY, km.predict(ccfX))  # cluster_acc # TODO change acc
    acc[k]['CreditCardF']['GMM'] = cluster_adjusted_rand(ccfY, gmm.predict(ccfX))  # cluster_acc

    # Normalized Mutual Information
    adjMI[k]['CreditCardF']['Kmeans'] = ami(ccfY, km.predict(ccfX))  # TODO change adj
    adjMI[k]['CreditCardF']['GMM'] = ami(ccfY, gmm.predict(ccfX))

    # Breast Cancer dataset
    start = time.time()
    km.fit(bcX)
    bc_km_fit_time = time.time() - start

    start = time.time()
    gmm.fit(bcX)
    bc_gmm_fit_time = time.time() - start

    fit_time_dict["BC_kmean"].append(bc_km_fit_time)
    fit_time_dict["BC_em"].append(bc_gmm_fit_time)

    SSE[k]['BreastC'] = km.score(bcX)
    ll[k]['BreastC'] = gmm.score(bcX)
    SS[k]['BreastC']['Kmeans'] = ss(bcX, km.predict(bcX))
    SS[k]['BreastC']['GMM'] = ss(bcX, gmm.predict(bcX))
    acc[k]['BreastC']['Kmeans'] = cluster_adjusted_rand(bcY, km.predict(bcX))  # cluster_acc
    acc[k]['BreastC']['GMM'] = cluster_adjusted_rand(bcY, gmm.predict(bcX))  # cluster_acc
    adjMI[k]['BreastC']['Kmeans'] = ami(bcY, km.predict(bcX))
    adjMI[k]['BreastC']['GMM'] = ami(bcY, gmm.predict(bcX))
    print(k, time.time() - st)  # TODO maybe have to save time or this timme is useless

fit_time = pd.DataFrame(fit_time_dict)
fit_time.to_csv(out + "Clustering_fit_time.csv", index=False)

SSE = (-pd.DataFrame(SSE)).T
SSE.rename(columns=lambda x: x + ' SSE (left)', inplace=True)
ll = pd.DataFrame(ll).T
ll.rename(columns=lambda x: x + ' log-likelihood', inplace=True)

SS = pd.DataFrame(SS)
acc = pd.DataFrame(acc)
adjMI = pd.DataFrame(adjMI)

SSE.to_csv(out + 'SSE.csv')
ll.to_csv(out + 'loglikelihood.csv')

temp = pd.json_normalize(SS.loc['CreditCardF']).T.reindex(['GMM', 'Kmeans'])
temp.columns = SS.columns
temp.to_csv(out + 'CreditCardF Silhouette.csv')

temp = pd.json_normalize(SS.loc['BreastC']).T.reindex(['GMM', 'Kmeans'])
temp.columns = SS.columns
temp.to_csv(out + 'BreastC Silhouette.csv')

temp = pd.json_normalize(acc.loc['CreditCardF']).T.reindex(['GMM', 'Kmeans'])
temp.columns = SS.columns
temp.to_csv(out + 'CreditCardF acc.csv')

temp = pd.json_normalize(acc.loc['BreastC']).T.reindex(['GMM', 'Kmeans'])
temp.columns = SS.columns
temp.to_csv(out + 'BreastC acc.csv')

temp = pd.json_normalize(adjMI.loc['CreditCardF']).T.reindex(['GMM', 'Kmeans'])
temp.columns = SS.columns
temp.to_csv(out + 'CreditCardF adjMI.csv')

temp = pd.json_normalize(adjMI.loc['BreastC']).T.reindex(['GMM', 'Kmeans'])
temp.columns = SS.columns
temp.to_csv(out + 'BreastC adjMI.csv')


