#!/bin/bash

python NN.py BASE   # Original dataset
python NN.py PCA    # Reduced dataset
python NN.py ICA    # Reduced dataset
python NN.py RP     # Reduced dataset
python NN.py RF     # Reduced dataset

python NN.py BASE kmeans  # kmean-feature-added dataset
python NN.py BASE GMM     # GMM-feature-added dataset

