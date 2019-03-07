from evaluation.metrics import eval
from predict.alpredictor import sampling_predict, predict_gaussian_prob, get_latent_gaussian_params
from recommendation_models.ifvae import IFVAE
from scipy.sparse import csr_matrix
from tqdm import tqdm
from utils.modelnames import vaes
from utils.progress import WorkSplitter, inhour
from utils.regularizers import Regularizer

import numpy as np
import tensorflow as tf
import time


class ThompsonSampling(object):
    def __init__(self, initial_reward, num_users, num_arms):
        self.counts = np.ones((num_users, num_arms))
        self.mu = initial_reward
        self.cummulative_x = initial_reward
        self.sigma = np.ones((num_users, num_arms))
        self.num_arms = num_arms

    def predict(self):
        return np.random.normal(self.mu, self.sigma)

    def update(self, chosen_arm, current_reward):
#        import ipdb; ipdb.set_trace()


        self.counts[chosen_arm] = self.counts[chosen_arm] + 1
#        import ipdb; ipdb.set_trace()

        # Assume variance sigma is known as 1
        posterior_variance = 1 / (1 / np.square(self.sigma) + self.counts / np.square(np.ones(self.counts.shape)))
#        import ipdb; ipdb.set_trace()


        prior_mu_divide_variance = self.mu / np.square(self.sigma)
#        import ipdb; ipdb.set_trace()

#        print(self.cummulative_x[chosen_arm])
        self.cummulative_x[chosen_arm] = self.cummulative_x[chosen_arm] + current_reward[chosen_arm]
#        print(self.cummulative_x[chosen_arm])
        sum_reward_divide_variance = self.cummulative_x / np.square(np.ones(self.counts.shape))
#        import ipdb; ipdb.set_trace()

        self.mu = posterior_variance * (prior_mu_divide_variance + sum_reward_divide_variance)
#        import ipdb; ipdb.set_trace()

        self.sigma = np.sqrt(posterior_variance)
#        import ipdb; ipdb.set_trace()

#        n = self.counts[chosen_arm]
#        average_reward = self.average_reward[chosen_arm]

#        new_average_reward = np.multiply((n - 1) / n, average_reward) + np.multiply(1 / n, immediate_reward[chosen_arm])
#        self.average_reward[chosen_arm] = new_average_reward

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
        # import ipdb; ipdb.set_trace()

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

def thompson_sampling(matrix_train, matrix_test, rec_model, topk, test_index, total_steps,
                      latent, sampling, evaluation_range, embedded_matrix=np.empty((0)), iteration=100,
                      rank=200, corruption=0.2, gpu=True, lam=80, optimizer="RMSProp",
                      **unused):

    progress = WorkSplitter()
    matrix_input = matrix_train
    if embedded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embedded_matrix.T))

    m, n = matrix_input.shape

    metrics_result = []

    user_case = []
#    import ipdb; ipdb.set_trace()


    model = vaes[rec_model](n, rank, 100, lamb=lam, optimizer=Regularizer[optimizer])

    progress.section("Training")
    model.train_model(matrix_input[test_index:], corruption, iteration)

    progress.section("Get Item Distribution")
    # Get all item distribution by feedforward passing one hot encoding vector
    # through encoder
    item_latent_mu, \
        item_latent_sigma = get_latent_gaussian_params(model=model,
                                                       is_item=True,
                                                       size=n)



    # Normalize item mu with the max number of positive ratings in train for an
    # item
    """
    train_item_pop = np.array(matrix_input.sum(axis=0))[0]
    scale = train_item_pop / train_item_pop.max()
    scale_reshape = scale.reshape(len(scale), 1)
    item_latent_mu = item_latent_mu * scale_reshape
    """

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
        predict_prob = predict_gaussian_prob(item_latent_mu, user_latent_mu, user_latent_sigma, model, matrix_input[:test_index], latent=latent, sampling=sampling)

        if i > 0:
            ts.update(chosen_arm=(chosen_arms_row.astype(np.int64), chosen_arms_col.astype(np.int64)), current_reward=predict_prob)

        # The bandit starts here
        if i == 0:
            ts = ThompsonSampling(initial_reward=predict_prob,
                                  num_users=test_index,
                                  num_arms=n)

        prediction_scores = ts.predict()

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
        result, matrix_input, chosen_arms_row, chosen_arms_col = ts.update_matrix(prediction, matrix_test, matrix_input, result, test_index)
#        import ipdb; ipdb.set_trace()
        result['active_learning_iteration'] = i + 1
        metrics_result.append(result)

        user_case.append({"357": prediction[357][0], "52": prediction[52][0], "314": prediction[314][0], "2456": prediction[2456][0], "1804": prediction[1804][0], "331": prediction[331][0]})

    model.sess.close()
    tf.reset_default_graph()

    return metrics_result, user_case

