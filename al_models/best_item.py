from evaluation.metrics import evaluate
from models.alpredictor import sampling_predict
from models.ifvae import IFVAE, get_gaussian_parameters, calculate_gaussian_log_pdf
from scipy.sparse import csr_matrix
from tqdm import tqdm
from utils.progress import WorkSplitter, inhour
from utils.regularizers import Regularizer

import numpy as np
import tensorflow as tf
import time

class BestItem(object):
    def __init__(self):
        return

    def predict(self, item_mu, user_mu, user_sigma):
        log_pdf = calculate_gaussian_log_pdf(item_mu, user_mu, user_sigma)
        return np.exp(log_pdf)

    def eval(self, prediction, matrix_valid, topk):
        start_time = time.time()

        metric_names = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision', 'MAP']

        result = evaluate(prediction, matrix_valid, metric_names, [topk])

        print("-")
        for metric in result.keys():
            print("{0}:{1}".format(metric, result[metric]))
        print("Elapsed: {0}".format(inhour(time.time() - start_time)))

        return result

    def update_matrix(self, prediction, matrix_valid, matrix_input, result):
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

        result['Num_Nonzero_In_Train'] = np.unique(matrix_input[matrix_input.nonzero()].A[0], return_counts=True)[1][np.where(np.unique(matrix_input[matrix_input.nonzero()].A[0], return_counts=True)[0] == 1.)][0]
        result['Num_Nonzero_In_Valid'] = len(matrix_valid.nonzero()[0])
        result['Num_Unmasked_Positive'] = len(prediction_valid_nonzero_intersect)
        result['Num_Unmasked_Negative'] = len(prediction_valid_zero_intersect)

        if len(prediction_valid_nonzero_intersect) > 0:
            mask_row = prediction_valid_nonzero_intersect[:, 0]
            mask_col = prediction_valid_nonzero_intersect[:, 1]
            mask_data = np.full(len(prediction_valid_nonzero_intersect), True)
            mask = csr_matrix((mask_data, (mask_row, mask_col)), shape=matrix_input.shape)

            matrix_input = matrix_input.tolil()
            matrix_valid = matrix_valid.tolil()
            matrix_input[mask] = 1
            matrix_valid[mask] = 0

        if len(prediction_valid_zero_intersect) > 0:
            mask_row = prediction_valid_zero_intersect[:, 0]
            mask_col = prediction_valid_zero_intersect[:, 1]
            mask_data = np.full(len(prediction_valid_zero_intersect), True)
            mask = csr_matrix((mask_data, (mask_row, mask_col)), shape=matrix_input.shape)

            matrix_input[mask] = -0.1

        print("Elapsed: {0}".format(inhour(time.time() - start_time)))

        return result, matrix_input.tocsr(), matrix_valid.tocsr()


def best_item(matrix_train, matrix_valid, topk, total_steps,
            retrain_interval, embedded_matrix=np.empty((0)), iteration=100,
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

    best_item_selection = BestItem()

    for i in range(total_steps):
        print('This is step {} \n'.format(i))
        print('The number of nonzero in train set is {}'.format(np.unique(matrix_input[matrix_input.nonzero()].A[0], return_counts=True)[1][np.where(np.unique(matrix_input[matrix_input.nonzero()].A[0], return_counts=True)[0] == 1.)][0]))
        print('The number of nonzero in valid set is {}'.format(len(matrix_valid.nonzero()[0])))

        if i % retrain_interval == 0:
            progress.section("Training")
            model.train_model(matrix_input, corruption, iteration)

            progress.section("Get Item Distribution")
            # Get all item distribution by feedforward passing one hot encoding vector
            # through encoder
            item_gaussian_mu, \
                item_gaussian_sigma = get_gaussian_parameters(model=model,
                                                              size=n,
                                                              is_item=True,
                                                              is_user=False)

        progress.section("Get User Distribution")
        # Get all user distribution by feedforward passing user vector through
        # encoder
        user_gaussian_mu, \
            user_gaussian_sigma = get_gaussian_parameters(model=model,
                                                          size=m,
                                                          is_item=False,
                                                          is_user=True,
                                                          matrix=matrix_input)

        progress.section("Sampling")
        prediction_scores = best_item_selection.predict(item_gaussian_mu, user_gaussian_mu, user_gaussian_sigma)

        prediction = sampling_predict(prediction_scores=prediction_scores,
                                      topK=topk,
                                      matrix_train=matrix_input,
                                      gpu=gpu_on)

        progress.section("Create Metrics")
        result = best_item_selection.eval(prediction, matrix_valid, topk)

        progress.section("Update Train Set and Valid Set Based On Sampling Results")
        result, matrix_input, matrix_valid = best_item_selection.update_matrix(prediction, matrix_valid, matrix_input, result)

        metrics_result.append(result)

    model.sess.close()
    tf.reset_default_graph()

    return metrics_result

