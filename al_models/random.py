from utils.progress import WorkSplitter, inhour
from models.alpredictor import sampling_predict

import numpy as np

def random_sampling(num_rows, num_cols):
    return np.random.random((num_rows, num_cols))

def random(matrix_train, matrix_valid, topk, total_steps, validation,
           embedded_matrix=np.empty((0)), gpu_on=True, **unused):
    progress = WorkSplitter()
    matrix_input = matrix_train
    if embedded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embedded_matrix.T))

    m, n = matrix_input.shape

    matrix_input_with_negative = matrix_input.copy()

    metrics_result = []

    for i in range(total_steps):
        print('This is step {} \n'.format(i))
        print('The number of nonzero in train set is {}'.format(len(matrix_input.nonzero()[0])))
        print('The number of nonzero in valid set is {}'.format(len(matrix_valid.nonzero()[0])))

        progress.section("Sampling")
        prediction_scores = random_sampling(m, n)

        prediction = sampling_predict(prediction_scores=prediction_scores,
                                      topK=topk,
                                      matrix_train=matrix_input,
                                      gpu=gpu_on)

        if validation:
            progress.section("Create Metrics")
            import time
            start_time = time.time()

            from evaluation.metrics import evaluate
            metric_names = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision', 'MAP']

            result = evaluate(prediction, matrix_valid, metric_names, [topk])

            print("-")
            for metric in result.keys():
                print("{0}:{1}".format(metric, result[metric]))
            print("Elapsed: {0}".format(inhour(time.time() - start_time)))


        progress.section("Update Train Set and Valid Set Based On Sampling Results")
        start_time = time.time()
        # TODO: Move these ‘k’ samples from the validation set to the train-set
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
        metrics_result.append(result)

        if len(prediction_valid_nonzero_intersect) > 0:
            mask_row = prediction_valid_nonzero_intersect[:, 0]
            mask_col = prediction_valid_nonzero_intersect[:, 1]
            mask_data = np.full(len(prediction_valid_nonzero_intersect), True)
            from scipy.sparse import csr_matrix
            mask = csr_matrix((mask_data, (mask_row, mask_col)), shape=matrix_input.shape)
            matrix_input = matrix_input.tolil()
            matrix_input_with_negative = matrix_input_with_negative.tolil()
            matrix_valid = matrix_valid.tolil()
            matrix_input[mask] = 1
            matrix_valid[mask] = 0
            matrix_input_with_negative[mask] = 1
            matrix_input = matrix_input.tocsr()
            matrix_valid = matrix_valid.tocsr()
            matrix_input_with_negative = matrix_input_with_negative.tocsr()

        if len(prediction_valid_zero_intersect) > 0:
            mask_row = prediction_valid_zero_intersect[:, 0]
            mask_col = prediction_valid_zero_intersect[:, 1]
            mask_data = np.full(len(prediction_valid_zero_intersect), True)
            from scipy.sparse import csr_matrix
            mask = csr_matrix((mask_data, (mask_row, mask_col)), shape=matrix_input.shape)
            matrix_input_with_negative = matrix_input_with_negative.tolil()
            matrix_input_with_negative[mask] = -0.1
            matrix_input_with_negative = matrix_input_with_negative.tocsr()

        print("Elapsed: {0}".format(inhour(time.time() - start_time)))

    return metrics_result
