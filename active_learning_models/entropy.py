from evaluation.metrics import evaluate, eval
from predict.alpredictor import sampling_predict
from recommendation_models.ifvae import IFVAE, get_gaussian_parameters, logsumexp_pdf
from scipy.sparse import csr_matrix
from tqdm import tqdm
from utils.progress import WorkSplitter, inhour
from utils.regularizers import Regularizer

import math
import numpy as np
import tensorflow as tf
import time


class Entropy(object):
    def __init__(self):
        return

    def predict(self, item_mu, user_mu, user_sigma):
        normalized_pdf = logsumexp_pdf(item_mu, user_mu, user_sigma)
        p_log2_p = np.multiply(normalized_pdf, np.log2(normalized_pdf))
        one_minus_p_log2_one_minus_p = np.multiply(1-normalized_pdf, 1-np.log2(normalized_pdf))
        return np.negative(np.nan_to_num(p_log2_p) + np.nan_to_num(one_minus_p_log2_one_minus_p))

    def eval(self, prediction_scores, matrix_input, matrix_test, topk, gpu_on):
        start_time = time.time()

        metric_names = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision', 'MAP']

        evaluation_items = sampling_predict(prediction_scores=-prediction_scores,
                                            topK=topk,
                                            matrix_train=matrix_input,
                                            gpu=gpu_on)
        result = evaluate(evaluation_items, matrix_test, metric_names, [topk])

        print("-")
        for metric in result.keys():
            print("{0}:{1}".format(metric, result[metric]))
        print("Elapsed: {0}".format(inhour(time.time() - start_time)))

        return result

    def update_matrix(self, prediction, matrix_test, result, matrix_input, result, test_index):
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
        prediction_test_zeros_intersect = np.array([x for x in index_prediction_set - index_test_ones_set])
        print('The number of zeros predicted is {}'.format(len(prediction_test_zeros_intersect)))

        result['Num_Ones_In_Train'] = len(matrix_input.nonzero()[0])
        result['Num_Ones_In_Test'] = len(matrix_test.nonzero()[0])
        result['Num_Ones_In_Prediction'] = len(prediction_test_ones_intersect)
        result['Num_Zeros_In_Prediction'] = len(prediction_test_zeros_intersect)

        if len(prediction_test_ones_intersect) > 0:
            mask_row = prediction_test_ones_intersect[:, 0]
            mask_col = prediction_test_ones_intersect[:, 1]
            mask_data = np.full(len(prediction_test_ones_intersect), True)
            mask = csr_matrix((mask_data, (mask_row, mask_col)), shape=matrix_input.shape)

            matrix_input = matrix_input.tolil()
            matrix_input[mask] = 1

        print("Elapsed: {0}".format(inhour(time.time() - start_time)))

        return result, matrix_input.tocsr()

def entropy(matrix_train, matrix_test, rec_model, topk, test_index, total_steps,
            latent, embedded_matrix=np.empty((0)), iteration=100,
            rank=200, corruption=0.2, gpu_on=True, lam=80, optimizer="RMSProp",
            beta=1.0, **unused):

    progress = WorkSplitter()
    matrix_input = matrix_train
    if embedded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embedded_matrix.T))

    m, n = matrix_input.shape

    metrics_result = []

    model = IFVAE(n, rank, 100, lamb=lam, beta=beta,
                  observation_distribution="Gaussian",
                  optimizer=Regularizer[optimizer])

    progress.section("Training")
    model.train_model(matrix_input[test_index:], corruption, iteration)

    progress.section("Get Item Distribution")
    # Get all item distribution by feedforward passing one hot encoding vector
    # through encoder
    item_gaussian_mu, \
        item_gaussian_sigma = get_gaussian_parameters(model=model,
                                                      is_item=True,
                                                      size=n)

    entropy_selection = Entropy()

    for i in range(total_steps):
        print('This is step {} \n'.format(i))
        print('The number of nonzero in train set is {}'.format(len(matrix_input.nonzero()[0])))
        print('The number of nonzero in test set is {}'.format(len(matrix_test.nonzero()[0])))

        progress.section("Get User Distribution")
        # Get all user distribution by feedforward passing user vector through
        # encoder
        user_gaussian_mu, \
            user_gaussian_sigma = get_gaussian_parameters(model=model,
                                                          is_item=False,
                                                          matrix=matrix_input[:test_index].A)

        progress.section("Sampling")
        prediction_scores = entropy_selection.predict(item_gaussian_mu, user_gaussian_mu, user_gaussian_sigma)

        prediction = sampling_predict(prediction_scores=prediction_scores,
                                      topK=topk,
                                      matrix_train=matrix_input,
                                      gpu=gpu_on)
        import ipdb; ipdb.set_trace()

        progress.section("Create Metrics")
        result = entropy_selection.eval(prediction_scores, matrix_input, matrix_test, topk, gpu_on)

        evaluation_scores = sampling_predict(prediction_scores=-prediction_scores,
                                             topK=topk,
                                             matrix_train=matrix_input[:test_index],
                                             gpu=gpu_on)
        result = eval(prediction=evaluation_scores, matrix_test[:test_index], topk)

        progress.section("Update Train Set and Test Set Based On Sampling Results")
        result, matrix_input = entropy_selection.update_matrix(prediction, matrix_test, result, matrix_input, result, test_index)

        metrics_result.append(result)

    model.sess.close()
    tf.reset_default_graph()

    return metrics_result

