# Machine Learning CS7641

* #### Student: Phat Nguyen
* #### GTID: pnguyen340

## Assignment 4

Please follow the below steps if you want to reproduce my results:

1. Clone the code by the following command

```
  git clone https://github.com/PhatNguyen39/CS7641-ML
```

2. Change directory

```
  cd assignment4/
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

* Part 1: Tuning hyperparameters for VI, PI and Q-learning in **Frozen Lake** (small and large state)
```
  python run_frozen_lake.py                        # VI + PI for small FL (8x8)
  python run_frozen_lake.py --large                # VI + PI for large FL (25x25)
  python run_frozen_lake.py --q_learning           # Q-learning for small FL (8x8)
  python run_frozen_lake.py --q_learning --large   # Q-learning for large FL (25x25)
```

* Part 2: Tuning hyperparameters for VI, PI and Q-learning in **Forest Management** (small and large state)
```
  python run_forest_management.py                        # VI + PI for small FM (8 states)
  python run_forest_management.py --large                # VI + PI for large FM (625 states)
  python run_forest_management.py --q_learning           # Q-learning for small FM (8 states)
  python run_forest_management.py --q_learning --large   # Q-learning for large FM (625 states)
```

* Part 3: Plotting
```
  python plotting.py
```

My results (tuning data and plots) for 3 parts are stored under `outputs` folder: https://github.com/PhatNguyen39/CS7641-ML/tree/main/assignment4/outputs



