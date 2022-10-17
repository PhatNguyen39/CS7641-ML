"""
Backprop NN training
"""
# Adapted from https://github.com/JonathanTay/CS-7641-assignment-2/blob/master/NN0.py

import sys

sys.path.append("./ABAGAIL.jar")
from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from func.nn.backprop import RPROPUpdateRule, BatchBackPropagationTrainer
from func.nn.activation import RELU
from base import *

# Network parameters found "optimal" in Assignment 1
'''
{'activation': 'relu', 'alpha': 0.01, 'batch_size': 'auto', 'beta_1': 0.9, 'beta_2': 0.999, 
'early_stopping': False, 'epsilon': 1e-08, 'hidden_layer_sizes': (15, 15), 'learning_rate': 'constant', 
'learning_rate_init': 0.001, 'max_fun': 15000, 'max_iter': 1000, 'momentum': 0.9, 'n_iter_no_change': 10, 
'nesterovs_momentum': True, 'power_t': 0.5, 'random_state': None, 'shuffle': True, 'solver': 'adam', 
'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': False, 'warm_start': False}
'''

INPUT_LAYER = 31 if DS_NAME == "WiscosinBreastCancerData" else 30
HIDDEN_LAYER1 = 15
HIDDEN_LAYER2 = 15
OUTPUT_LAYER = 1
TRAINING_ITERATIONS = 1000
# OUTFILE = OUTPUT_DIRECTORY + '/NN_OUTPUT/NN_{}_LOG.csv'
OUTFILE = OUTPUT_DIRECTORY + '/NN_OUTPUT/' + DS_NAME + '/' + 'NN_{}_LOG.csv'


def main():
    """Run this experiment"""
    training_ints = initialize_instances(TRAIN_DATA_FILE)
    testing_ints = initialize_instances(TEST_DATA_FILE)
    validation_ints = initialize_instances(VALIDATE_DATA_FILE)
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()
    data_set = DataSet(training_ints)
    relu = RELU()

    # 50 and 0.000001 are the defaults from RPROPUpdateRule.java
    rule = RPROPUpdateRule(0.064, 50, 0.000001)
    oa_names = ["Backprop"]

    classification_network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER1, HIDDEN_LAYER2, OUTPUT_LAYER], relu)
    train(BatchBackPropagationTrainer(data_set, classification_network, measure, rule), classification_network, 'Backprop', training_ints, validation_ints, testing_ints, measure, TRAINING_ITERATIONS, OUTFILE.format('Backprop'))


if __name__ == "__main__":
    with open(OUTFILE.format('Backprop'), 'a+') as f:
        f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format('iteration', 'MSE_trg', 'MSE_val', 'MSE_tst', 'acc_trg',
                                                            'acc_val', 'acc_tst', 'f1_trg', 'f1_val', 'f1_tst', 'elapsed'))
    main()
