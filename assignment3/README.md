# Machine Learning CS7641

* #### Student: Phat Nguyen
* #### GTID: pnguyen340

## Assignment 3

Please follow the below steps if you want to reproduce my results:

1. Clone the code by the following command

```
  git clone https://github.com/PhatNguyen39/CS7641-ML
```

2. Change directory

```
  cd assignment3/
```

3. Create conda environment

```
  conda create -n cs7641 python=3.9
```

4. Install libraries

```
  pip install -r requirements.txt
```

5. Run experiments

* Part 1: Clustering
```
  python clustering.py BASE
```

* Part 2 & 3: Dimensionality reduction and Clustering
```
  bash dim_red.sh
  bash clustering.py
```

* Part 4: Neural Network on dimensionality-reduced datasets
```
  bash NN.sh
```

* Part 5: Neural Network on cluster features (k-means and EM)
```
  python NN.py BASE kmeans
  python NN.py BASE GMM
```

6. Plots for parts 1-3: Run jupyter notebooks in `Charts` folder.
