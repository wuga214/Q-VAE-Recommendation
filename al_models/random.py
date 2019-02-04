from evaluation.metrics import evaluate
from models.alpredictor import sampling_predict
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

    def eval(self, prediction, matrix_valid, topk):
        start_time = time.time()

        metric_names = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision', 'MAP']

        result = evaluate(prediction, matrix_valid, metric_names, [topk])

        print("-")
        for metric in result.keys():
            print("{0}:{1}".format(metric, result[metric]))
        print("Elapsed: {0}".format(inhour(time.time() - start_time)))

        return result

    def update(self, prediction, matrix_valid, matrix_input, result):
        start_time = time.time()
        # Move these ‘k’ samples from the validation set to the train-set
        # and query their labels.
        index = np.tile(np.arange(prediction.shape[0]),(prediction.shape[1],1)).T
        index_prediction = np.dstack((index, prediction)).reshape((prediction.shape[0]*prediction.shape[1]), 2)
        index_valid_nonzero = np.dstack((matrix_valid.nonzero()[0], matrix_valid.nonzero()[1]))[0]

        index_prediction_set = set([tuple(x) for x in index_prediction])
        index_valid_nonzero_set = set([tuple(x) for x in index_valid_nonzero])
        prediction_valid_nonzero_intersect = np.array([x for x in index_prediction_set & index_valid_nonzero_set])
        print('The number of unmasked positive data is {}'.format(len(prediction_valid_nonzero_intersect)))
        prediction_valid_zero_intersect = np.array([x for x in index_prediction_set - index_valid_nonzero_set])
        print('The number of unmasked negative data is {}'.format(len(prediction_valid_zero_intersect)))

        result['Num_Nonzero_In_Train'] = len(matrix_input.nonzero()[0])
        result['Num_Nonzero_In_Valid'] = len(matrix_valid.nonzero()[0])
        result['Num_Unmasked_Positive'] = len(prediction_valid_nonzero_intersect)
        result['Num_Unmasked_Negative'] = len(prediction_valid_zero_intersect)

        if len(prediction_valid_nonzero_intersect) > 0:
            mask_row = prediction_valid_nonzero_intersect[:, 0]
            mask_col = prediction_valid_nonzero_intersect[:, 1]
            mask_data = np.full(len(prediction_valid_nonzero_intersect), True)
            mask = csr_matrix((mask_data, (mask_row, mask_col)), shape=matrix_input.shape)

            matrix_input[mask] = 1
            matrix_valid[mask] = 0

        print("Elapsed: {0}".format(inhour(time.time() - start_time)))

        return result, matrix_input, matrix_valid


def random(matrix_train, matrix_valid, topk, total_steps,
           embedded_matrix=np.empty((0)), gpu_on=True, **unused):
    progress = WorkSplitter()
    matrix_input = matrix_train
    if embedded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embedded_matrix.T))

    m, n = matrix_input.shape

    matrix_input = matrix_input.tolil()
    matrix_valid = matrix_valid.tolil()

    metrics_result = []

    random_selection = Random(num_rows=m, num_cols=n)

    for i in range(total_steps):
        print('This is step {} \n'.format(i))
        print('The number of nonzero in train set is {}'.format(len(matrix_input.nonzero()[0])))
        print('The number of nonzero in valid set is {}'.format(len(matrix_valid.nonzero()[0])))

        progress.section("Sampling")
        prediction_scores = random_selection.predict()

        prediction = sampling_predict(prediction_scores=prediction_scores,
                                      topK=topk,
                                      matrix_train=matrix_input,
                                      gpu=gpu_on)

        progress.section("Create Metrics")
        result = random_selection.eval(prediction, matrix_valid, topk)

        progress.section("Update Train Set and Valid Set Based On Sampling Results")
        result, matrix_input, matrix_valid = random_selection.update(prediction, matrix_valid, matrix_input, result)

        metrics_result.append(result)

    return metrics_result

