from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

def concatenate_datasets(datasets):
    result = datasets[0]
    for dataset in datasets[1:]:
        result = np.concatenate((result, dataset), axis=0)
    return result

def preprocess( dataset, div = 1 ):
    row_index_end = dataset.shape[0] - dataset.shape[0] % div  # divisible by div, but What is div for?
    data_x = dataset[:row_index_end, 4:-1]

    data_y = dataset[:row_index_end, -1]
    # Change training labels
    inds1 = np.where(data_y == -1)
    data_y[inds1] = 2
    return data_x,data_y

from src.VFBLS_v110.bls.processing.one_hot_m import one_hot_m
from src.VFBLS_v110.bls.model.bls_train import bls_train_realtime

print("======================= BLS =======================\n")
def xpr_train_test(train_x, train_y, test_x):
    # Set parameters
    mem = 'low'
    # mem = 'high'
    # BLS parameters
    seed = 1  # set the seed for generating random numbers
    num_class = 2  # number of the classes
    epochs = 1  # number of epochs
    C = 2 ** -15  # parameter for sparse regularization
    s = 0.6  # the shrinkage parameter for enhancement nodes
    train_y = one_hot_m(train_y, num_class)
    # test_y = one_hot_m(test_y, num_class);
    #######################
    # N1* - the number of mapped feature nodes
    # N2* - the groups of mapped features
    # N3* - the number of enhancement nodes
    if mem == 'low':
        N1_bls = 20
        N2_bls = 5
        N3_bls = 100
    else:
        N1_bls = 200
        N2_bls = 10
        N3_bls = 100
    #######################

    train_err = np.zeros((1, epochs))
    train_time = np.zeros((1, epochs))
    test_time = np.zeros((1, epochs))
    np.random.seed(seed)  # set the seed for generating random numbers
    for j in range(0, epochs):
        trainingAccuracy, trainingTime, testingTime, predicted = \
        bls_train_realtime(train_x, train_y, test_x,
                            s, C,
                            N1_bls, N2_bls, N3_bls)

        train_err[0, j] = trainingAccuracy * 100
        train_time[0, j] = trainingTime
        test_time[0, j] = testingTime
    # predicted = [[1.], [2.], [2.], [2.], [2.]]
    predicted_list = []
    for label in predicted:
        predicted_list.append(label[0])
    return predicted_list

import os
import sys
sys.path.append('./src/VFBLS_v110')
from src.xpr_feature_reshaping import *
from sklearn.metrics import precision_score, recall_score, confusion_matrix

def blockPrint():
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    return old_stdout
def enablePrint(old_stdout):
    sys.stdout = old_stdout


def xpr_test( raw_train_datasets, raw_test_dataset, time_span, normtype="Power", slide_window=False):
    train_datasets = aggregate_datasets(raw_train_datasets, time_span, slide_window)
    test_dataset = aggregate_rows( raw_test_dataset, time_span, slide_window)
    # train_datasets = aggregate_datasets(raw_train_datasets, time_span, True)
    # test_dataset = aggregate_rows(datasets[combo["test"]], time_span, True)
    train_dataset = concatenate_datasets(train_datasets)
            
    train_x, train_y = preprocess(train_dataset)
    test_x, test_y = preprocess(test_dataset)

    train_x, test_x = norm( train_x, test_x, normtype)
    

    predicted_list = xpr_train_test(train_x, train_y, test_x)
    return predicted_list
