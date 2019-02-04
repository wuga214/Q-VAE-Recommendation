from evaluation.metrics import evaluate
from models.alpredictor import sampling_predict
from models.ifvae import IFVAE
from tqdm import tqdm
from utils.progress import WorkSplitter, inhour
from utils.regularizers import Regularizer

import math
import numpy as np
import tensorflow as tf
import time


class UCB1(object):
    def __init__(self, counts, average_reward, num_arms):
        self.counts = counts
        self.average_reward = average_reward
        self.num_arms = num_arms
        self.ucb_scores = None
        self.chosen_arm = None

    def set_initial_average_reward(self, average_reward):
        self.average_reward = average_reward

    def get_gaussian_parameters(self, model, size, is_item, is_user, matrix_input_with_negative=None):
        mu, sigma = [], []

        for i in tqdm(range(size)):
            # Can only get item or user distribution at one time
            if is_item & is_user == is_item | is_user:
                raise ValueError('Either get item distribution or user distribution.')
            elif is_item:
                vector = self.create_one_hot_vector(num_classes=size, nth_item=i)
            else:
                vector = matrix_input_with_negative[i, :].todense()

            Gaussian_Params = model.uncertainty(vector)
            mu.append(Gaussian_Params[0][0])
            sigma.append(Gaussian_Params[1][0])

        return np.array(mu), np.array(sigma)

    def create_one_hot_vector(self, num_classes, nth_item):
        return np.eye(num_classes)[[nth_item]]

    def calculate_gaussian_log_pdf(self, item_mu, user_mu, user_sigma):
        result = []
        for user_index in range(len(user_mu)):
            result.append(self.multivariate_normal_log_pdf(x=item_mu, mean=user_mu[user_index], cov=np.square(user_sigma[user_index])))
            # return np.negative(np.sum(np.divide(np.square(item_gaussian_mu-user_gaussian_mu[0]), 2 * np.square(user_gaussian_sigma[0])) + 0.5 * np.log(2 * math.pi * np.square(user_gaussian_sigma[0])), axis=1))
            # np.log(multivariate_normal.pdf(x=item_gaussian_mu[0], mean=user_gaussian_mu[0], cov=np.square(user_gaussian_sigma[0])))
        return result

    # log_p(I_Mu|U_Mu, U_Sigma)
    def multivariate_normal_log_pdf(self, x, mean, cov):
        return np.negative(np.sum(np.divide(np.square(x-mean), 2 * cov) + 0.5 * np.log(2 * math.pi * cov), axis=1))

    def eval(self, prediction, matrix_valid, topk):
        start_time = time.time()

        metric_names = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision', 'MAP']

        result = evaluate(prediction, matrix_valid, metric_names, [topk])

        print("-")
        for metric in result.keys():
            print("{0}:{1}".format(metric, result[metric]))
        print("Elapsed: {0}".format(inhour(time.time() - start_time)))

        return result


    def predict(self):
        total_counts = np.sum(self.counts, axis=1)

        bonus = np.sqrt((2 * np.log(total_counts))[:,None] / self.counts)
        self.ucb_scores = self.average_reward + bonus

        return self.ucb_scores

    def update(self, chosen_arm, immediate_reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1

        n = self.counts[chosen_arm]
        average_reward = self.average_reward[chosen_arm]

        new_average_reward = np.multiply((n - 1) / n, average_reward) + np.multiply(1 / n, immediate_reward)
        self.average_reward[chosen_arm] = new_average_reward

def ucb1(matrix_train, matrix_valid, topk, total_steps,
         retrain_interval,  embedded_matrix=np.empty((0)), iteration=100,
         rank=200, corruption=0.2, gpu_on=True, lam=80, optimizer="RMSProp",
         beta=1.0, **unused):

    progress = WorkSplitter()
    matrix_input = matrix_train
    if embedded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embedded_matrix.T))

    matrix_input_with_negative = matrix_input.copy()

    m, n = matrix_input.shape

    metrics_result = []

    model = IFVAE(n, rank, 100, lamb=lam, beta=beta,
                  observation_distribution="Gaussian",
                  optimizer=Regularizer[optimizer])

    progress.section("Training")
    model.train_model(matrix_input, corruption, iteration)

    ucb_selection = UCB1(counts=np.ones((m, n)),
                         average_reward=None,
                         num_arms=n)


    progress.section("Get Item Distribution")
    # Get all item distribution by feedforward passing one hot encoding vector
    # through encoder
    item_gaussian_mu, item_gaussian_sigma = ucb_selection.\
        get_gaussian_parameters(model=model, size=n,
                                is_item=True, is_user=False)

    progress.section("Get User Distribution")
    # Get all user distribution by feedforward passing user vector through
    # encoder
    user_gaussian_mu, user_gaussian_sigma = ucb_selection.\
        get_gaussian_parameters(model=model, size=m,
                                is_item=False, is_user=True,
                                matrix_input_with_negative=matrix_input_with_negative)


    model.sess.close()
    tf.reset_default_graph()

    progress.section("Sampling")
    log_prediction_scores = ucb_selection.calculate_gaussian_log_pdf(item_gaussian_mu, user_gaussian_mu, user_gaussian_sigma)

    ucb_selection.set_initial_average_reward(average_reward=np.exp(log_prediction_scores))
    import ipdb; ipdb.set_trace()

    for i in range(total_steps):
        print('This is step {} \n'.format(i))
        print('The number of nonzero in train set is {}'.format(len(matrix_input.nonzero()[0])))
        print('The number of nonzero in valid set is {}'.format(len(matrix_valid.nonzero()[0])))

        prediction_scores = ucb_selection.predict()

        prediction = sampling_predict(prediction_scores=prediction_scores,
                                    topK=topk,
                                    matrix_train=matrix_input,
                                    gpu=gpu_on)

        progress.section("Create Metrics")
        result = ucb_selection.eval(prediction, matrix_valid, topk)

        import ipdb; ipdb.set_trace()
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

        chosen_arms_row = []
        chosen_arms_col = []
        immediate_reward = []

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

            chosen_arms_row = np.append(chosen_arms_row, mask_row)
            chosen_arms_col = np.append(chosen_arms_col, mask_col)
            immediate_reward = np.append(immediate_reward, np.ones(len(prediction_valid_nonzero_intersect)))

        if len(prediction_valid_zero_intersect) > 0:
            mask_row = prediction_valid_zero_intersect[:, 0]
            mask_col = prediction_valid_zero_intersect[:, 1]
            mask_data = np.full(len(prediction_valid_zero_intersect), True)
            from scipy.sparse import csr_matrix
            mask = csr_matrix((mask_data, (mask_row, mask_col)), shape=matrix_input.shape)
            matrix_input_with_negative = matrix_input_with_negative.tolil()
            matrix_input_with_negative[mask] = -0.1
            matrix_input_with_negative = matrix_input_with_negative.tocsr()

            chosen_arms_row = np.append(chosen_arms_row, mask_row)
            chosen_arms_col = np.append(chosen_arms_col, mask_col)
            immediate_reward = np.append(immediate_reward, np.full(len(prediction_valid_zero_intersect), -0.1))

        print("Elapsed: {0}".format(inhour(time.time() - start_time)))

        ucb_selection.update(chosen_arm=(chosen_arms_row.astype(np.int64), chosen_arms_col.astype(np.int64)), immediate_reward=immediate_reward)
    # import ipdb; ipdb.set_trace()
    return metrics_result

