from evaluation.metrics import eval
from predict.alpredictor import sampling_predict, predict_gaussian_prob, get_latent_gaussian_params
from recommendation_models.ifvae import IFVAE
from scipy.sparse import csr_matrix
from tqdm import tqdm
from utils.progress import WorkSplitter, inhour
from utils.regularizers import Regularizer

import numpy as np
import tensorflow as tf
import time


class NonLinUCB(object):
    def __init__(self, counts, average_reward, num_arms):
        self.counts = counts
        self.average_reward = average_reward
        self.num_arms = num_arms
        self.ucb_scores = None
        self.chosen_arm = None

    def predict(self):
        total_counts = np.sum(self.counts, axis=1)

        bonus = np.sqrt((2 * np.log(total_counts))[:,None] / self.counts)
        self.ucb_scores = self.average_reward + bonus
        return self.ucb_scores

    def update(self, chosen_arm, immediate_reward):
        self.counts[chosen_arm] = self.counts[chosen_arm] + 1

        n = self.counts[chosen_arm]
        average_reward = self.average_reward[chosen_arm]

        new_average_reward = np.multiply((n - 1) / n, average_reward) + np.multiply(1 / n, immediate_reward[chosen_arm])
        self.average_reward[chosen_arm] = new_average_reward

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
        prediction_test_zeros_intersect = np.array([x for x in index_prediction_set - index_test_ones_set])
        print('The number of zeros predicted is {}'.format(len(prediction_test_zeros_intersect)))

        result['Num_Ones_In_Train'] = len(matrix_input[:test_index].nonzero()[0])
        result['Num_Ones_In_Test'] = len(matrix_test[:test_index].nonzero()[0])
        result['Num_Ones_In_Prediction'] = len(prediction_test_ones_intersect)
        result['Num_Zeros_In_Prediction'] = len(prediction_test_zeros_intersect)

        chosen_arms_row = []
        chosen_arms_col = []

        if len(prediction_test_ones_intersect) > 0:
            mask_row = prediction_test_ones_intersect[:, 0]
            mask_col = prediction_test_ones_intersect[:, 1]
            mask_data = np.full(len(prediction_test_ones_intersect), True)
            mask = csr_matrix((mask_data, (mask_row, mask_col)), shape=matrix_input.shape)

            matrix_input = matrix_input.tolil()
            matrix_input[mask] = 1

            chosen_arms_row = np.append(chosen_arms_row, mask_row)
            chosen_arms_col = np.append(chosen_arms_col, mask_col)
        # import ipdb; ipdb.set_trace()

        if len(prediction_test_zeros_intersect) > 0:
            mask_row = prediction_test_zeros_intersect[:, 0]
            mask_col = prediction_test_zeros_intersect[:, 1]

            chosen_arms_row = np.append(chosen_arms_row, mask_row)
            chosen_arms_col = np.append(chosen_arms_col, mask_col)
        # import ipdb; ipdb.set_trace()

        print("Elapsed: {0}".format(inhour(time.time() - start_time)))

        return result, matrix_input.tocsr(), chosen_arms_row, chosen_arms_col

def non_lin_ucb(matrix_train, matrix_test, rec_model, topk, test_index, total_steps,
         latent, embedded_matrix=np.empty((0)), iteration=100,
         rank=200, corruption=0.2, gpu=True, lam=80, optimizer="RMSProp",
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

    """
    input, wc, wc_normalized, hc, hc_normalized = model.return_input(matrix_input.A)
    print("check if input is the same as matrix input {}".format(np.array_equal(input, matrix_input.A)))
    print("check if input is the same as wc {}".format(np.array_equal(input, wc)))
    print("check if input is the same as hc {}".format(np.array_equal(input, hc)))
    print("check if wc is the same as hc {}".format(np.array_equal(wc, hc)))
    print("the number of positive ratings in input {}".format(input.nonzero()[0].shape))
    print("the number of positive ratings in wc {}".format(wc.nonzero()[0].shape))
    print("the number of positive ratings in hc {}".format(hc.nonzero()[0].shape))


    wc_row_sum = np.sum(wc, axis=1)
    wc_row_sum[wc_row_sum == 0.] = 1.
    wc_row_sum = wc_row_sum.reshape(-1, 1)
    python_result = wc / wc_row_sum


    wc_row_sum_dim = np.sum(wc, axis=1, keepdims=True)
    wc_row_sum_dim[wc_row_sum_dim == 0.] = 1.
    result = wc / wc_row_sum_dim

    hc_row_sum = np.sum(hc, axis=1)
    hc_row_sum[hc_row_sum == 0.] = 1.
    hc_row_sum = hc_row_sum.reshape(-1, 1)
    python_result2 = hc / hc_row_sum


    hc_row_sum_dim = np.sum(hc, axis=1, keepdims=True)
    hc_row_sum_dim[hc_row_sum_dim == 0.] = 1.
    result2 = hc / hc_row_sum_dim

    print("wc normalized result {}, {}".format(np.array_equal(wc_normalized, python_result), np.array_equal(python_result, result)))
    print("hc normalized result {}, {}".format(np.array_equal(hc_normalized, python_result2), np.array_equal(python_result2, result2)))

    import ipdb; ipdb.set_trace()
    """


    progress.section("Training")
    model.train_model(matrix_input[test_index:], corruption, iteration)

    progress.section("Get Item Distribution")
    # Get all item distribution by feedforward passing one hot encoding vector
    # through encoder
    item_latent_mu, \
        item_latent_sigma = get_latent_gaussian_params(model=model,
                                                       is_item=True,
                                                       size=n)

    for i in range(total_steps):
        print('This is step {} \n'.format(i))
        print('The number of ones in train set is {}'.format(len(matrix_input[:test_index].nonzero()[0])))
        print('The number of ones in test set is {}'.format(len(matrix_test[:test_index].nonzero()[0])))

        progress.section("Get User Distribution")
        # Get all user distribution by feedforward passing user vector through
        # encoder
        user_latent_mu, \
            user_latent_sigma = get_latent_gaussian_params(model=model,
                                                           is_item=False,
                                                           matrix=matrix_input[:test_index].A)
        # import ipdb; ipdb.set_trace()

        progress.section("Sampling")
        # Get normalized pdf
        predict_prob = predict_gaussian_prob(item_latent_mu, user_latent_mu, user_latent_sigma, model, matrix_input[:test_index], latent=latent)

        import ipdb; ipdb.set_trace()

        if i > 0:
            ucb_selection.update(chosen_arm=(chosen_arms_row.astype(np.int64), chosen_arms_col.astype(np.int64)), immediate_reward=predict_prob)

        # The bandit starts here
        if i == 0:
            ucb_selection = NonLinUCB(counts=np.ones((test_index, n)),
                                      average_reward=predict_prob,
                                      num_arms=n)

        prediction_scores = ucb_selection.predict()

        prediction = sampling_predict(prediction_scores=prediction_scores,
                                      topK=topk,
                                      matrix_train=matrix_train[:test_index],
                                      gpu=gpu)

        '''
        from predict.predictor import predict
        RQ = model.get_RQ(matrix_input)
        Y = model.get_Y().T
        Bias = model.get_Bias()
        user_inference = model.inference(matrix_input.A)
        pre = predict(matrix_U=RQ,
                         matrix_V=Y,
                         bias=Bias,
                         topK=50,
                         matrix_Train=matrix_input,
                         gpu=gpu)
        '''
        # import ipdb; ipdb.set_trace()

        print(matrix_train[:test_index].nonzero())
        progress.section("Create Metrics")
        result = eval(matrix_test[:test_index], topk, prediction)

        progress.section("Update Train Set and Test Set Based On Sampling Results")
        result, matrix_input, chosen_arms_row, chosen_arms_col = ucb_selection.update_matrix(prediction, matrix_test, matrix_input, result, test_index)

        metrics_result.append(result)

    model.sess.close()
    tf.reset_default_graph()

    return metrics_result

