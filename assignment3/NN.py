# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 2022
@author: pnguyen340 (modified from jtay)
"""

import os.path
import sys

import numpy as np
# %% Imports
import pandas as pd
import sklearn.model_selection
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import compute_sample_weight
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM

import time

# import wandb
# wandb.init(project="visualize-sklearn")


def balanced_accuracy(truth, pred):
    wts = compute_sample_weight('balanced', truth)
    return accuracy_score(truth, pred, sample_weight=wts)


def perform_grid_search_cv(X_train, y_train, clustering_method=None):
    # L2 Penalty Parameter Regularization for NN
    alphas = [0.001, 0.01, 0.1]
    hidden_layer_sizes = [(h,) * l for l in np.arange(1, 3, 1) for h in [5, 10, 15]]

    np.random.seed(9)

    # Part 4 - Apply dim red on one of your datasets
    grid = {'alpha': alphas,
            'hidden_layer_sizes': hidden_layer_sizes,
            "max_iter": [2 ** x for x in range(12)] + [2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]}

    mlp = MLPClassifier(activation='relu', early_stopping=True, random_state=9, learning_rate_init=0.001, momentum=0.3)
    gs = GridSearchCV(mlp, grid, verbose=10, cv=5, scoring=make_scorer(balanced_accuracy))
    gs.fit(X_train, y_train)

    tmp = pd.DataFrame(gs.cv_results_)

    if clustering_method is None:
        tmp.to_csv(out1 + f'{dim_red_method}_CCF_NN_dim_red.csv')
    else:
        tmp.to_csv(out1 + f'{clustering_method}_{dim_red_method}_CCF_NN_dim_red.csv')


def train_and_plot_best_model(model_name="NN", dataset_name="CCF"):
    # Note that iterations here are equivalent to epochs
    if dim_red_method != "BASE":
        if dim_red_method == "PCA":
            best_params = {'alpha': 0.1, 'hidden_layer_sizes': (15,), 'max_iter': 64}  # PCA
        elif dim_red_method == "ICA":
            best_params = {'alpha': 0.1, 'hidden_layer_sizes': (15, 15), 'max_iter': 16}  # ICA
        elif dim_red_method == "RP":
            best_params = {'alpha': 0.1, 'hidden_layer_sizes': (15,), 'max_iter': 64}  # RP   or 0.1, (15,), 128
        # RF
        else:
            best_params = {'alpha': 0.1, 'hidden_layer_sizes': (15, 15), 'max_iter': 32}  # RF

    elif clustering_method is not None:
        if clustering_method == "kmeans":
            best_params = {'alpha': 0.1, 'hidden_layer_sizes': (10, 10), 'max_iter': 64}    # kmeans
        elif clustering_method == "GMM":
            best_params = {'alpha': 0.01, 'hidden_layer_sizes': (15, 15), 'max_iter': 64}   # GMM
        else:
            return
    else:
        # Best params for NN on original CCF dataset
        best_params = {'alpha': 0.1, 'hidden_layer_sizes': (15,), 'max_iter': 64}   # BASE

    mlp = MLPClassifier(activation='relu', early_stopping=True, random_state=9, batch_size=200,
                        learning_rate_init=0.001, momentum=0.3, **best_params, verbose=True)
    mlp.fit(ccfX_train, ccfY_train)
    ccfY_test_pred = mlp.predict(ccfX_test)

    if clustering_method is None:
        print("Test accuracy of {} = {}".format(dim_red_method, balanced_accuracy(ccfY_test, ccfY_test_pred)))
    else:
        print("Test accuracy of {} -- {} = {}".format(dim_red_method, clustering_method, balanced_accuracy(ccfY_test, ccfY_test_pred)))

    # wandb.sklearn.plot_learning_curve(mlp, ccfX_train, ccfY_train)
    colors = ["tab:red", "tab:green"]
    fig, ax1 = plt.subplots(figsize=(9.6, 6.4), dpi=200)
    ax2 = ax1.twinx()

    ln1 = ax1.plot([acc * 100 for acc in mlp.validation_scores_], color=colors[1], marker=".", label="validation accuracy")
    ln2 = ax2.plot(mlp.loss_curve_, color=colors[0], marker="*", label="logistic loss")
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="best", fontsize=16)

    ax1.set_xlabel("# iterations (epochs)", fontsize=16)
    ax1.set_ylabel("accuracy (%)", fontsize=16)
    ax2.set_ylabel("loss", fontsize=16)

    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 1)

    ax1.xaxis.set_tick_params(labelsize=16)
    ax1.yaxis.set_tick_params(labelsize=16)
    ax2.yaxis.set_tick_params(labelsize=16)

    plot_name = dim_red_method if dim_red_method != "BASE" else "Original"
    plot_name += " - " + clustering_method if clustering_method is not None else ""
    plt.title(f"Loss curve (NN & CCF) - {plot_name}", fontsize=16)

    if clustering_method is None:
        fig.savefig('{}/{}_{}_{}_LossC.png'.format(out1, dim_red_method, model_name, dataset_name), format='png', dpi=200)
    else:
        fig.savefig('{}/{}_{}_{}_{}_LossC.png'.format(out1, dim_red_method, clustering_method, model_name, dataset_name), format='png', dpi=200)

    make_learning_curve(mlp, ccfX_train, ccfY_train, dim_red_method=dim_red_method, model_name=model_name, dataset_name=dataset_name)
    make_timing_curve(mlp, ccfX_train, ccfY_train, dim_red_method=dim_red_method, model_name=model_name, dataset_name=dataset_name)

    # wandb.log({"plot": plt})

def plot_learning_curve(title, train_sizes, train_scores, test_scores, ylim=None, multiple_runs=True,
                        x_scale='linear', y_scale='linear',
                        x_label='Training examples (%)', y_label='accuracy (%)'):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    title : string
        Title for the chart.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    train_sizes : list, array
        The training sizes

    train_scores : list, array
        The training scores

    test_scores : list, array
        The testing sizes

    multiple_runs : boolean
        If True, assume the given train and test scores represent multiple runs of a given test (the default)

    x_scale: string
        The x scale to use (defaults to None)

    y_scale: string
        The y scale to use (defaults to None)

    x_label: string
        Label fo the x-axis

    y_label: string
        Label fo the y-axis
    """

    fig = plt.figure(figsize=(9.6, 6.4), dpi=200)
    fig.tight_layout()

    plot_name = dim_red_method if dim_red_method != "BASE" else "Original"
    plot_name += " - " + clustering_method if clustering_method is not None else ""
    plt.title(title + f" - {plot_name}", fontsize=16)
    plt.xlabel(x_label, fontsize=16)
    plt.ylabel(y_label, fontsize=16)

    if ylim is not None:
        plt.ylim(*ylim)

    train_points = train_scores
    test_points = test_scores

    if x_scale is not None or y_scale is not None:
        ax = plt.gca()
        # ax.autoscale('y')
        if x_scale is not None:
            ax.set_xscale(x_scale)
        if y_scale is not None:
            ax.set_yscale(y_scale)

    if multiple_runs:
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        train_points = train_scores_mean
        test_points = test_scores_mean

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.2)
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.2)

    plt.plot(train_sizes, [acc * 100 for acc in train_points], 'o-', linewidth=1, markersize=4, label="Training accuracy")
    plt.plot(train_sizes, [acc * 100 for acc in test_points], 'o-', linewidth=1, markersize=4, label="Cross-validation accuracy")

    # Zoom in the plots for better reading
    plt.ylim([40 if min(list(train_points) + list(test_points)) >= 0.4 else 20, 100])

    plt.grid()
    plt.legend(loc="best", fontsize=16)

    return plt

def make_learning_curve(clf, X, y, dim_red_method, model_name="NN", dataset_name="CCF"):
    n = y.shape[0]

    train_size_fracs = np.linspace(0, 1.0, 21, endpoint=True)[1:]  # 0 should not be included
    print(" - n: {}, train_sizes: {}".format(n, train_size_fracs))

    train_sizes, train_scores, test_scores = learning_curve(clf, X, y, cv=5, train_sizes=train_size_fracs,
                                                            scoring=make_scorer(balanced_accuracy), verbose=10,
                                                            n_jobs=-1, random_state=seed)

    print(" - n: {}, train_sizes: {}".format(n, train_sizes))
    # curve_train_scores = pd.DataFrame(index=train_sizes, data=train_scores)
    # curve_test_scores = pd.DataFrame(index=train_sizes, data=test_scores)
    # curve_train_scores.to_csv('{}/{}_{}_LC_train.csv'.format(out1, model_name, dataset_name))
    # curve_test_scores.to_csv('{}/{}_{}_LC_test.csv'.format(out1, model_name, dataset_name))

    plt = plot_learning_curve('Learning Curve: {} - {}'.format(model_name, dataset_name),
                              [frac * 100 for frac in train_size_fracs], train_scores, test_scores)

    if clustering_method is None:
        plt.savefig('{}/{}_{}_{}_LC.png'.format(out1, dim_red_method, model_name, dataset_name), format='png', dpi=200)
    else:
        plt.savefig('{}/{}_{}_{}_{}_LC.png'.format(out1, dim_red_method, clustering_method, model_name, dataset_name), format='png', dpi=200)

    print(" - Learning curve complete")

def plot_model_timing(title, data_sizes, fit_scores, predict_scores, ylim=None, mean_std_axis=1):
    """
    Generate a simple plot of the given model timing data

    Parameters
    ----------
    title : string
        Title for the chart.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    data_sizes : list, array
        The data sizes

    fit_scores : list, array
        The fit/train times

    predict_scores : list, array
        The predict times

    """

    fit_scores_mean = np.mean(fit_scores, axis=mean_std_axis)
    fit_scores_std = np.std(fit_scores, axis=mean_std_axis)
    predict_scores_mean = np.mean(predict_scores, axis=mean_std_axis)
    predict_scores_std = np.std(predict_scores, axis=mean_std_axis)

    fig = plt.figure(figsize=(9.6, 6.4), dpi=200)
    fig.tight_layout()

    plot_name = dim_red_method if dim_red_method != "BASE" else "Original"
    plot_name += " - " + clustering_method if clustering_method is not None else ""
    plt.title(title + f" - {plot_name}", fontsize=16)
    plt.xlabel("Training Data Size (% of total)", fontsize=16)
    plt.ylabel("Log-scale Time (s)", fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    if ylim is not None:
        plt.ylim(*ylim)

    ind = data_sizes
    bar_width = 2.5

    plt.bar(ind, fit_scores_mean, bar_width, yerr=fit_scores_std, label="Train time")
    plt.bar(ind + bar_width, predict_scores_mean, bar_width, yerr=predict_scores_std, label="Test time")

    plt.grid()
    plt.yscale("log")
    plt.legend(loc="best", fontsize=16)

    return fig
def make_timing_curve(clf, X, y, dim_red_method, model_name="NN", dataset_name="CCF"):
    print("Building timing curve")

    sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tests = 5

    out = dict()
    out['train'] = np.zeros(shape=(len(sizes), tests))
    out['test'] = np.zeros(shape=(len(sizes), tests))
    for i, frac in enumerate(sizes):
        for j in range(tests):
            np.random.seed(seed)
            x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1-frac, random_state=seed)

            st = time.time()
            clf.fit(x_train, y_train)
            out['train'][i, j] = (time.time() - st)

            st = time.time()
            clf.predict(x_test)
            out['test'][i, j] = (time.time() - st)

    train_df = pd.DataFrame(out['train'], index=sizes)
    test_df = pd.DataFrame(out['test'], index=sizes)
    mean_std_axis = 1

    fig = plot_model_timing('Time Curve: {} - {}'.format(model_name, dataset_name),
                            np.array(sizes) * 100, train_df, test_df, mean_std_axis=mean_std_axis)

    if clustering_method is None:
        fig.savefig('{}/{}_{}_{}_TC.png'.format(out1, dim_red_method, model_name, dataset_name), format='png', dpi=200)
    else:
        fig.savefig('{}/{}_{}_{}_{}_TC.png'.format(out1, dim_red_method, clustering_method, model_name, dataset_name), format='png', dpi=200)

    print(" - Timing curve complete")


def generate_cluster_features(feature_only=False):
    # Add KMeans features
    km = kmeans(n_clusters=8, random_state=seed)
    km.fit(ccfX_train)

    # Add new features to both train and test of the original dataset
    ccfX_train_kmeans = np.concatenate((ccfX_train, np.expand_dims(km.predict(ccfX_train), 1)), axis=1)
    ccfX_test_kmeans = np.concatenate((ccfX_test, np.expand_dims(km.predict(ccfX_test), 1)), axis=1)

    if not feature_only:
        perform_grid_search_cv(ccfX_train_kmeans, ccfY_train, clustering_method="kmeans")

    # Add GMM features
    gmm = GMM(n_components=7, random_state=seed)
    gmm.fit(ccfX_train)

    # Add new features to both train and test of the original dataset
    ccfX_train_gmm = np.concatenate((ccfX_train, np.expand_dims(gmm.predict(ccfX_train), 1)), axis=1)
    ccfX_test_gmm = np.concatenate((ccfX_test, np.expand_dims(gmm.predict(ccfX_test), 1)), axis=1)

    if not feature_only:
        perform_grid_search_cv(ccfX_train_gmm, ccfY_train, clustering_method="GMM")

    if feature_only:
        return ccfX_train_kmeans, ccfX_test_kmeans, ccfX_train_gmm, ccfX_test_gmm


if __name__ == '__main__':
    # sys.argv[1] represents the first command-line argument (as a string) supplied to the script in question.
    dim_red_method = sys.argv[1] if len(sys.argv) >= 2 else None
    clustering_method = sys.argv[2] if len(sys.argv) >= 3 else None

    out = './Output/{}/'.format(dim_red_method)
    out1 = './Part4and5output/'
    seed = 9

    if not os.path.exists(out1):
        os.makedirs(out1, exist_ok=True)

    # ccf dataset
    ccf_train = pd.read_hdf(out + 'datasets.hdf', 'train_ccf')
    ccfX_train = ccf_train.drop('labels', 1).copy().values
    ccfY_train = ccf_train['labels'].copy().values

    ccf_test = pd.read_hdf(out + 'datasets.hdf', 'test_ccf')
    ccfX_test = ccf_test.drop('labels', 1).copy().values
    ccfY_test = ccf_test['labels'].copy().values

    # bc dataset
    '''
    bc_train = pd.read_hdf(out + 'datasets.hdf', 'train_bc')
    bcX_train = bc_train.drop(['diagnosis'], 1).copy().values

    bc_train.loc[bc_train["diagnosis"] == "B", "diagnosis"] = 0
    bc_train.loc[bc_train["diagnosis"] == "M", "diagnosis"] = 1
    bcY_train = bc_train['diagnosis'].copy().values

    bc_test = pd.read_hdf(out + 'datasets.hdf', 'test_bc')
    bcX_test = bc_test.drop(['diagnosis'], 1).copy().values

    bc_test.loc[bc_test["diagnosis"] == "B", "diagnosis"] = 0
    bc_test.loc[bc_test["diagnosis"] == "M", "diagnosis"] = 1
    bcY_test = bc_test['diagnosis'].copy().values

    ccfX_train = StandardScaler().fit_transform(ccfX_train)
    ccfX_test = StandardScaler().fit_transform(ccfX_test)
    bcX_train = StandardScaler().fit_transform(bcX_train)
    bcX_test = StandardScaler().fit_transform(bcX_test)
    '''

    # perform_grid_search_cv(ccfX_train, ccfY_train)

    if clustering_method is not None:
        ccfX_train_kmeans, ccfX_test_kmeans, ccfX_train_gmm, ccfX_test_gmm = generate_cluster_features(feature_only=True)

        # Update dataset using Kmeans features
        if clustering_method == "kmeans":
            ccfX_train = ccfX_train_kmeans
            ccfX_test = ccfX_test_kmeans

        # Update dataset using GMM features
        elif clustering_method == "GMM":
            ccfX_train = ccfX_train_gmm
            ccfX_test = ccfX_test_gmm

    train_and_plot_best_model()


'''
Test accuracy of BASE = 0.8675530818387962
Test accuracy of PCA = 0.8928571428571428
Test accuracy of ICA = 0.6277056277056278
Test accuracy of RP = 0.8629663986806844
Test accuracy of RF = 0.8776025561739846
Test accuracy of BASE -- kmeans = 0.8623479694908266
Test accuracy of BASE -- GMM = 0.8627602556173984
'''