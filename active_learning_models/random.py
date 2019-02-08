from evaluation.metrics import evaluate, eval
from predict.alpredictor import sampling_predict
from scipy.sparse import csr_matrix
from utils.progress import WorkSplitter, inhour

import numpy as np
import time


class Random(object):
    def __init__(self, num_rows, num_cols):
        self.num_rows = num_rows
        self.num_cols = num_cols

    def predict(self):
        return np.random.random((self.num_rows, self.num_cols))

    def update_matrix(self, prediction, matrix_test, matrix_input, result, test_index):
        start_time = time.time()
        # Query ‘k’ samples's labels from the test set and mark predicted
        # positive feedback as ones in the train set
        index = np.tile(np.arange(prediction.shape[0]),(prediction.shape[1],1)).T
        index_prediction = np.dstack((index, prediction)).reshape((prediction.shape[0]*prediction.shape[1]), 2)
        index_test_ones = np.dstack((matrix_test.nonzero()[0], matrix_test.nonzero()[1]))[0]

        index_prediction_set = set([tuple(x) for x in index_prediction])
        index_test_ones_set = set([tuple(x) for x in index_test_ones])
        prediction_test_ones_intersect = np.array([x for x in index_prediction_set & index_test_ones_set])
        print('The number of ones predicted is {}'.format(len(prediction_test_ones_intersect)))
        prediction_test_zero_intersect = np.array([x for x in index_prediction_set - index_test_ones_set])
        print('The number of zeros predicted is {}'.format(len(prediction_test_zero_intersect)))

        result['Num_Ones_In_Train'] = len(matrix_input.nonzero()[0])
        result['Num_Ones_In_Valid'] = len(matrix_test.nonzero()[0])
        result['Num_Ones_In_Prediction'] = len(prediction_test_ones_intersect)
        result['Num_Zeros_In_Prediction'] = len(prediction_test_zero_intersect)

        if len(prediction_test_ones_intersect) > 0:
            mask_row = prediction_test_ones_intersect[:, 0]
            mask_col = prediction_test_ones_intersect[:, 1]
            mask_data = np.full(len(prediction_test_ones_intersect), True)
            mask = csr_matrix((mask_data, (mask_row, mask_col)), shape=matrix_input.shape)

            matrix_input[mask] = 1

        print("Elapsed: {0}".format(inhour(time.time() - start_time)))

        return result, matrix_input


def random(matrix_train, matrix_test, topk, test_index, total_steps,
           embedded_matrix=np.empty((0)), gpu_on=True, **unused):
    progress = WorkSplitter()
    matrix_input = matrix_train
    if embedded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embedded_matrix.T))

    m, n = matrix_input.shape

    matrix_input = matrix_input.tolil()
    matrix_test = matrix_test.tolil()

    metrics_result = []

    random_selection = Random(num_rows=m, num_cols=n)

    for i in range(total_steps):
        print('This is step {} \n'.format(i))
        print('The number of ones in train set is {}'.format(len(matrix_input.nonzero()[0])))
        print('The number of ones in test set is {}'.format(len(matrix_test.nonzero()[0])))

        progress.section("Sampling")
        prediction_scores = random_selection.predict()[:test_index]

        prediction = sampling_predict(prediction_scores=prediction_scores,
                                      topK=topk,
                                      matrix_train=matrix_input[:test_index],
                                      gpu=gpu_on)

        progress.section("Create Metrics")
        result = eval(prediction, matrix_test[:test_index], topk)

        progress.section("Update Train Set and Valid Set Based On Sampling Results")
        result, matrix_input = random_selection.update_matrix(prediction, matrix_test, matrix_input, result, test_index)

        metrics_result.append(result)

    return metrics_result

