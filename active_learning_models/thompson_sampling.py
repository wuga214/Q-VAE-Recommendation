from evaluation.metrics import evaluate
from predict.alpredictor import sampling_predict
from recommendation_models.ifvae import IFVAE, get_gaussian_parameters, logsumexp_pdf
from scipy.sparse import csr_matrix
from tqdm import tqdm
from utils.progress import WorkSplitter, inhour
from utils.regularizers import Regularizer

import numpy as np
import tensorflow as tf
import time


class ThompsonSampling(object):
    def __init__(self, initial_reward):
        self.alpha = 1. + initial_reward
        self.beta = 1. + 1. - initial_reward

    def update(self, reward):
        self.alpha = self.alpha + reward
        self.beta = self.beta + 1 - reward

    def predict(self):
        return np.random.beta(self.alpha, self.beta)

    @staticmethod
    def eval(prediction, matrix_valid, topk):
        start_time = time.time()

        metric_names = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision', 'MAP']

        result = evaluate(prediction, matrix_valid, metric_names, [topk])

        print("-")
        for metric in result.keys():
            print("{0}:{1}".format(metric, result[metric]))
        print("Elapsed: {0}".format(inhour(time.time() - start_time)))

        return result

    @staticmethod
    def update_matrix(prediction, matrix_valid, matrix_input, result):
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

        # np.unique(np.array(matrix_input[matrix_input.nonzero()])[0], return_counts=True)
        result['Num_Nonzero_In_Train'] = np.unique(matrix_input[matrix_input.nonzero()].A[0], return_counts=True)[1][np.where(np.unique(matrix_input[matrix_input.nonzero()].A[0], return_counts=True)[0] == 1.)][0]
        result['Num_Nonzero_In_Valid'] = len(matrix_valid.nonzero()[0])
        result['Num_Unmasked_Positive'] = len(prediction_valid_nonzero_intersect)
        result['Num_Unmasked_Negative'] = len(prediction_valid_zero_intersect)

        chosen_arms_row = []
        chosen_arms_col = []

        if len(prediction_valid_nonzero_intersect) > 0:
            mask_row = prediction_valid_nonzero_intersect[:, 0]
            mask_col = prediction_valid_nonzero_intersect[:, 1]
            mask_data = np.full(len(prediction_valid_nonzero_intersect), True)
            mask = csr_matrix((mask_data, (mask_row, mask_col)), shape=matrix_input.shape)

            matrix_input = matrix_input.tolil()
            matrix_valid = matrix_valid.tolil()
            matrix_input[mask] = 1
            matrix_valid[mask] = 0

            chosen_arms_row = np.append(chosen_arms_row, mask_row)
            chosen_arms_col = np.append(chosen_arms_col, mask_col)

        if len(prediction_valid_zero_intersect) > 0:
            mask_row = prediction_valid_zero_intersect[:, 0]
            mask_col = prediction_valid_zero_intersect[:, 1]
            mask_data = np.full(len(prediction_valid_zero_intersect), True)
            mask = csr_matrix((mask_data, (mask_row, mask_col)), shape=matrix_input.shape)

            matrix_input[mask] = -0.1

            chosen_arms_row = np.append(chosen_arms_row, mask_row)
            chosen_arms_col = np.append(chosen_arms_col, mask_col)

        print("Elapsed: {0}".format(inhour(time.time() - start_time)))

        return result, matrix_input.tocsr(), matrix_valid.tocsr(), chosen_arms_row, chosen_arms_col


def thompson_sampling(matrix_train, matrix_valid, topk, total_steps,
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

    progress.section("Training")
    model.train_model(matrix_input, corruption, iteration)

    progress.section("Get Item Distribution")
    # Get all item distribution by feedforward passing one hot encoding vector
    # through encoder
    item_gaussian_mu, \
        item_gaussian_sigma = get_gaussian_parameters(model=model, size=n,
                                                      is_item=True, is_user=False)

    for i in range(total_steps):
        print('This is step {} \n'.format(i))
        print("The number of nonzero in train set is {}".format(np.unique(matrix_input[matrix_input.nonzero()].A[0], return_counts=True)[1][np.where(np.unique(matrix_input[matrix_input.nonzero()].A[0], return_counts=True)[0] == 1.)][0]))
        print('The number of nonzero in valid set is {}'.format(len(matrix_valid.nonzero()[0])))

        progress.section("Get User Distribution")
        # Get all user distribution by feedforward passing user vector through
        # encoder
        user_gaussian_mu, \
            user_gaussian_sigma = get_gaussian_parameters(model=model, size=m,
                                                          is_item=False,
                                                          is_user=True,
                                                          matrix=matrix_input)

        progress.section("Sampling")
        # Get normalized probability
        normalized_pdf = logsumexp_pdf(item_gaussian_mu, user_gaussian_mu, user_gaussian_sigma).ravel()

        prediction_scores = []

        if i > 0:
            chosen_arms_index = (chosen_arms_row * n + chosen_arms_col).astype(np.int64)

            for arm in chosen_arms_index[:result['Num_Unmasked_Positive']]:
                ts_list[arm].update(normalized_pdf[arm])

            for arm in chosen_arms_index[-result['Num_Unmasked_Negative']:]:
                ts_list[arm].update(1-normalized_pdf[arm])

            for ts_index in range(len(normalized_pdf)):
                prediction_scores.append(ts_list[ts_index].predict())

        # Bandits start here
        if i == 0:
            ts_list = []

            for ts_index in range(len(normalized_pdf)):
                ts_list.append(ThompsonSampling(initial_reward=normalized_pdf[ts_index]))
                prediction_scores.append(ts_list[ts_index].predict())

        prediction_scores = np.array(prediction_scores).reshape((m, n))

        prediction = sampling_predict(prediction_scores=prediction_scores,
                                      topK=topk,
                                      matrix_train=matrix_input,
                                      gpu=gpu_on)

        progress.section("Create Metrics")
        result = ThompsonSampling.eval(prediction, matrix_valid, topk)

        progress.section("Update Train Set and Valid Set Based On Sampling Results")
        result, matrix_input, matrix_valid, chosen_arms_row, chosen_arms_col = ThompsonSampling.update_matrix(prediction, matrix_valid, matrix_input, result)
        # import ipdb; ipdb.set_trace()

        metrics_result.append(result)

    model.sess.close()
    tf.reset_default_graph()

    return metrics_result


