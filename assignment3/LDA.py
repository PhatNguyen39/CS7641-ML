# %% Imports
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import ImportanceSelect
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    out = './Output/LDA/'
    out1 = './Output/LDA/LDA/'
    f_path = './Output/BASE/'

    np.random.seed(0)
    ## Read in data ##
    nn_reg = [(10 ** -3)]
    nn_arch = [(14,)]
    nn_arch2 = [(14,)]
    # nn_arch= [(12,),(14,),(16,)]
    # nn_arch2= [(4,),(6,),(8,)]

    # Credit Card Fraud dataset
    ccf = pd.read_hdf(f_path + 'datasets.hdf', 'train_ccf')
    ccfX = ccf.drop('labels', 1).copy().values
    ccfY = ccf['labels'].copy().values

    # Breast Cancer dataset
    bc = pd.read_hdf(f_path + 'datasets.hdf', 'train_bc')
    bcX = bc.drop(['diagnosis'], 1).copy().values
    bcY = bc['diagnosis'].copy().values

    ccfX = StandardScaler().fit_transform(ccfX)
    bcX = StandardScaler().fit_transform(bcX)

    clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    dims = [1]

    # %% data for 1
    lda_classifier = LDA()
    fs_ccf = lda_classifier.fit(ccfX, ccfY).explained_variance_ratio_
    fs_bc = lda_classifier.fit(bcX, bcY).explained_variance_ratio_

    # tmp = pd.Series(np.sort(fs_ccf)[::-1])
    tmp = pd.Series(data=fs_ccf)
    tmp.to_csv(out1 + 'ccf scree.csv')

    # tmp = pd.Series(np.sort(fs_bc)[::-1])
    tmp = pd.Series(data=fs_bc)
    tmp.to_csv(out1 + 'bc scree.csv')

    # %% Data for 2
    # filtr = ImportanceSelect(lda_classifier)
    grid = {'lda__n_components': dims, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
    mlp = MLPClassifier(activation='relu', max_iter=200, early_stopping=True, random_state=5, learning_rate_init=0.1, momentum=0.3)
    # pipe = Pipeline([('filter', filtr), ('NN', mlp)])
    pipe = Pipeline([('lda', lda_classifier), ('NN', mlp)])
    gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

    gs.fit(ccfX, ccfY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out1 + 'ccf dim red.csv')

    grid = {'lda__n_components': dims, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch2}
    mlp = MLPClassifier(activation='relu', max_iter=200, early_stopping=True, random_state=5, learning_rate_init=0.1, momentum=0.3)
    pipe = Pipeline([('lda', lda_classifier), ('NN', mlp)])
    gs = GridSearchCV(pipe, grid, verbose=10, cv=5)

    gs.fit(bcX, bcY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out1 + 'bc dim red.csv')

    # %% data for 3
    # Set this from chart 2 and dump, use clustering script to finish up
    dim = 1
    # filtr = ImportanceSelect(lda_classifier, dim)
    lda_classifier = LDA(n_components=dim)
    ccfX2 = lda_classifier.fit_transform(ccfX, ccfY)
    ccf2 = pd.DataFrame(np.hstack((ccfX2, np.atleast_2d(ccfY).T)))
    cols = list(range(ccf2.shape[1]))
    cols[-1] = 'labels'
    ccf2.columns = cols
    ccf2.to_hdf(out + 'datasets.hdf', 'train_ccf', complib='blosc', complevel=9)

    dim = 1
    # filtr = ImportanceSelect(lda_classifier, dim)
    lda_classifier = LDA(n_components=dim)
    ccfX2 = lda_classifier.fit_transform(ccfX, ccfY)
    bcX2 = lda_classifier.fit_transform(bcX, bcY)
    bc2 = pd.DataFrame(np.hstack((bcX2, np.atleast_2d(bcY).T)))
    cols = list(range(bc2.shape[1]))
    cols[-1] = 'diagnosis'
    bc2.columns = cols
    bc2.to_hdf(out + 'datasets.hdf', 'train_bc', complib='blosc', complevel=9)
