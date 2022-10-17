# Machine Learning CS7641

* #### Student: Phat Nguyen
* #### GTID: pnguyen340

## Assignment 2

Please follow the below steps if you want to reproduce my results:

0. Follow the instructions to cython: https://www.jython.org/installation

1. Clone the code by the following command

```
  git clone https://github.com/PhatNguyen39/CS7641-ML
```
  
2. Change directory

```
  cd assignment2/
```

3. Create conda environment

```
  conda create -n cs7641 python=3.9
```

4. Install libraries

```
  pip install -r requirements.txt
```

5. Run experiments for each algorithm as follows

a. Part 1: Optimization problems and randomized algorithms

```
  jython continuouspeaks.py
  jython tsp.py
  jython knapsack.py
```

b. Part 2: Neural Network

```
  jython NN-Backprop.py
  jython NN-RHC.py
  jython NN-SA.py
  jython NN-GA.py
```

6. Convert the results stored in CSV file to the figures

```
  python plotting.py
```
