import re
import random
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils.progress import WorkSplitter, inhour
from scipy.sparse import vstack, hstack
from utils.regularizers import Regularizer

class IFVAE(object):

    def __init__(self, observation_dim, latent_dim, batch_size,
                 lamb=0.01,
                 beta=1.0,
                 learning_rate=1e-4,
                 optimizer=tf.train.RMSPropOptimizer,
                 observation_distribution="Gaussian", # or Bernoulli or Multinomial
                 observation_std=0.01):

        self._lamb = lamb
        self._latent_dim = latent_dim
        self._beta = beta
        self._batch_size = batch_size
        self._observation_dim = observation_dim
        self._learning_rate = learning_rate
        self._optimizer = optimizer
        self._observation_distribution = observation_distribution
        self._observation_std = observation_std
        self._build_graph()

    def _build_graph(self):

        with tf.variable_scope('ifvae'):
            self.input = tf.placeholder(tf.float32, shape=[None, self._observation_dim], name='input')
            self.corruption = tf.placeholder(tf.float32)
            self.sampling = tf.placeholder(tf.bool)

            mask1 = tf.floor(tf.nn.dropout(tf.ones_like(self.input), 1-self.corruption))
            mask2 = tf.floor(tf.nn.dropout(tf.ones_like(self.input), 1-self.corruption))

            wc = self.input * mask1
            hc = wc * mask2
            with tf.variable_scope('network'):
                self.wc_mean, wc_log_std, self.wc_obs_mean = self._network(wc)
            with tf.variable_scope('network', reuse=True):
                self.hc_mean, hc_log_std, self.hc_obs_mean = self._network(hc)

            self.wc_std = tf.exp(wc_log_std)

            with tf.variable_scope('loss'):
                with tf.variable_scope('kl-divergence'):
                    kl1 = self._kl_diagnormal_stdnormal(self.wc_mean, wc_log_std,
                                                        self.hc_mean, hc_log_std)
                    kl2 = self._kl_diagnormal_stdnormal(self.hc_mean, hc_log_std,
                                                        tf.zeros_like(self.hc_mean),
                                                        tf.zeros_like(hc_log_std))

                if self._observation_distribution == 'Gaussian':
                    with tf.variable_scope('gaussian'):

                        obj1 = self._gaussian_log_likelihood(wc * (1 - mask2),
                                                             self.wc_obs_mean * mask1 * (1 - mask2),
                                                             self._observation_std)
                        obj2 = self._gaussian_log_likelihood(hc,
                                                             self.hc_obs_mean * mask1 * mask2,
                                                             self._observation_std)
                elif self._observation_distribution == 'Bernoulli':
                    with tf.variable_scope('bernoulli'):
                        obj1 = self._bernoulli_log_likelihood(self.input * tf.floor(mask1) * (1 - tf.floor(mask2)),
                                                              self.wc_obs_mean * tf.floor(mask1) * (1 - tf.floor(mask2)))
                        obj2 = self._bernoulli_log_likelihood(self.input * tf.floor(mask1) * tf.floor(mask2),
                                                              self.hc_obs_mean * tf.floor(mask1) * tf.floor(mask2))

                else:
                    with tf.variable_scope('multinomial'):
                        obj1 = self._multinomial_log_likelihood(self.input * tf.floor(mask1) * (1 - tf.floor(mask1)),
                                                                self.wc_obs_mean * tf.floor(mask1) * (1 - tf.floor(mask1)))
                        obj2 = self._multinomial_log_likelihood(self.input * tf.floor(mask1) * tf.floor(mask2),
                                                                self.hc_obs_mean * tf.floor(mask1) * tf.floor(mask2))

                with tf.variable_scope('l2'):
                    l2_loss = tf.reduce_mean(tf.nn.l2_loss(self.encode_weights) + tf.nn.l2_loss(self.decode_weights))

                self._loss = ((self._beta * (kl1 + kl2) + obj1 + obj2) + self._lamb * l2_loss) * (1 - self.corruption)

            with tf.variable_scope('optimizer'):
                optimizer = self._optimizer(learning_rate=self._learning_rate)
            with tf.variable_scope('training-step'):
                var_list = [self.encode_weights, self.encode_bias, self.decode_weights, self.decode_bias]
                self._train = optimizer.minimize(self._loss, var_list=var_list)

            self.sess = tf.Session()
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def _network(self, x):
        with tf.variable_scope('encoder'):
            self.encode_weights = tf.get_variable("Weights", initializer=tf.truncated_normal([self._observation_dim,
                                                                                              self._latent_dim * 2],
                                                                                             stddev=1 / 500.0))
            self.encode_bias = tf.get_variable("Bias", initializer=tf.constant(0., shape=[self._latent_dim * 2]))

            encoded = tf.matmul(x, self.encode_weights) + self.encode_bias

        with tf.variable_scope('latent'):
            mean = tf.nn.relu(encoded[:, :self._latent_dim])
            logstd = encoded[:, self._latent_dim:]

            logstd = tf.where(tf.less(logstd, -3), tf.ones(tf.shape(logstd))*-3, logstd)

            std = tf.exp(logstd)
            epsilon = tf.random_normal(tf.shape(std))
            z = mean

            z = tf.cond(self.sampling, lambda: z + std * epsilon, lambda: z)

        with tf.variable_scope('decoder'):
            self.decode_weights = tf.get_variable("Weights", initializer=tf.truncated_normal([self._latent_dim,
                                                                                              self._observation_dim],
                                                                                             stddev=1 / 500.0))
            self.decode_bias = tf.get_variable("Bias", initializer=tf.constant(0., shape=[self._observation_dim]))
            decoded = tf.matmul(z, self.decode_weights) + self.decode_bias

            obs_mean = decoded

        return mean, logstd, obs_mean

    @staticmethod
    def _kl_diagnormal_stdnormal(mu_1, log_std_1, mu_2=0, log_std_2=0):

        var_square_1 = tf.exp(2. * log_std_1)
        var_square_2 = tf.exp(2. * log_std_2)
        kl = 0.5 * tf.reduce_mean(2 * log_std_2 - 2. * log_std_1
                                  + tf.divide(var_square_1 + tf.square(mu_1 - mu_2), var_square_2) - 1.)
        return kl

    @staticmethod
    def _gaussian_log_likelihood(targets, mean, std):
        se = 0.5 * tf.reduce_mean(tf.square(targets - mean)) / (2 * tf.square(std)) + tf.log(std)
        return se

    @staticmethod
    def _bernoulli_log_likelihood(targets, outputs, eps=1e-8):

        log_like = -tf.reduce_mean(targets * tf.log(outputs + eps)
                                   + (1. - targets) * tf.log((1. - outputs) + eps))
        return log_like

    @staticmethod
    def _multinomial_log_likelihood(target, outputs, eps=1e-8):
        log_softmax_output = tf.nn.log_softmax(outputs)
        log_like = -tf.reduce_sum(tf.reduce_sum(log_softmax_output * target, axis=1))
        return log_like

    def update(self, x, corruption):
        _, loss = self.sess.run([self._train, self._loss],
                                feed_dict={self.input: x, self.corruption: corruption, self.sampling: True})
        return loss

    def inference(self, x):
        predict = self.sess.run(self.wc_obs_mean,
                                feed_dict={self.input: x, self.corruption: 0, self.sampling: False})
        return predict

    def uncertainty(self, x):
        gaussian_parameters = self.sess.run([self.wc_mean, self.wc_std],
                                            feed_dict={self.input: x, self.corruption: 0, self.sampling: False})

        return gaussian_parameters

    def train_model(self, rating_matrix, corruption, epoch=100, batches=None, **unused):
        if batches is None:
            batches = self.get_batches(rating_matrix, self._batch_size)

        # Training
        pbar = tqdm(range(epoch))
        for i in pbar:
            for step in range(len(batches)):
                corrupt_rate = random.uniform(0.1, 0.3)
                #corrupt_rate = corruption
                feed_dict = {self.input: batches[step].todense(), self.corruption: corrupt_rate,
                             self.sampling: True}
                training, loss = self.sess.run([self._train, self.wc_std], feed_dict=feed_dict)
                pbar.set_description("loss: {:.4f}".format(np.mean(loss)))

    def get_batches(self, rating_matrix, batch_size):
        remaining_size = rating_matrix.shape[0]
        batch_index = 0
        batches = []
        while remaining_size > 0:
            if remaining_size < batch_size:
                batches.append(rating_matrix[batch_index * batch_size:])
            else:
                batches.append(rating_matrix[batch_index * batch_size:(batch_index + 1) * batch_size])
            batch_index += 1
            remaining_size -= batch_size
        return batches

    def get_RQ(self, rating_matrix):
        batches = self.get_batches(rating_matrix, self._batch_size)
        RQ = []
        for step in range(len(batches)):
            feed_dict = {self.input: batches[step].todense(), self.corruption: 0, self.sampling: False}
            embedding = self.sess.run(self.wc_mean, feed_dict=feed_dict)
            RQ.append(embedding)

        return np.vstack(RQ)

    def get_Y(self):
        return self.sess.run(self.decode_weights)

    def get_Bias(self):
        return self.sess.run(self.decode_bias)


def ifvae(matrix_train, matrix_valid, topk, al_model, total_steps, retrain_interval, validation, embedded_matrix=np.empty((0)),
          iteration=100, lam=80, rank=200, corruption=0.2, optimizer="RMSProp",
          beta=1.0, seed=1, gpu_on=True, **unused):
    progress = WorkSplitter()
    matrix_input = matrix_train
    if embedded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embedded_matrix.T))

    matrix_input_with_negative = matrix_input.copy()

    m, n = matrix_input.shape

    metrics_result = []

    model = IFVAE(n, rank, 100, lamb=lam, beta=beta,
                    observation_distribution="Gaussian", optimizer=Regularizer[optimizer])

    export_metrics_df_name = al_model + "_" + str(total_steps) + "steps_" + str(topk) + "items_per_step_per_user_retrain_every_" + str(retrain_interval) + "steps"

    for i in range(total_steps):
        print('This is step {} \n'.format(i))
        print('The number of nonzero in train set is {}'.format(len(matrix_input.nonzero()[0])))
        print('The number of nonzero in valid set is {}'.format(len(matrix_valid.nonzero()[0])))

        if i % retrain_interval == 0:
            progress.section("Training")
            model.train_model(matrix_input, corruption, iteration)

            progress.section("Get Item Distribution")
            # Get all item distribution by feedforward passing one hot encoding vector
            # through encoder
            item_gaussian_mu, item_gaussian_sigma = [], []
            for nth_item in tqdm(range(n)):
                # print(nth_item)
                one_hot_vector = create_one_hot_vector(num_classes=n, nth_item=nth_item)
                # print(one_hot_vector)
                Gaussian_Params = model.uncertainty(one_hot_vector)
                item_gaussian_mu.append(Gaussian_Params[0][0])
                item_gaussian_sigma.append(Gaussian_Params[1][0])

            item_gaussian_mu, item_gaussian_sigma = np.array(item_gaussian_mu), np.array(item_gaussian_sigma)

        progress.section("Get User Distribution")

        user_gaussian_mu, user_gaussian_sigma = [], []
        for nth_user in tqdm(range(m)):
            user_vector = matrix_input_with_negative[nth_user, :]
            Gaussian_Params = model.uncertainty(user_vector.todense())
            user_gaussian_mu.append(Gaussian_Params[0][0])
            user_gaussian_sigma.append(Gaussian_Params[1][0])
        user_gaussian_mu, user_gaussian_sigma = np.array(user_gaussian_mu), np.array(user_gaussian_sigma)

        progress.section("Sampling")

        if al_model == "Entropy":
            prediction_scores = entropy_sampling(item_gaussian_mu, user_gaussian_mu, user_gaussian_sigma)
        elif al_model == "Random":
            prediction_scores = random_sampling(m, n)
        else:
            print("Don't have this active learning approaches!")

        # print(prediction_scores)

        prediction = sampling_predict(prediction_scores=prediction_scores,
                                      topK=topk,
                                      matrix_Train=matrix_input,
                                      gpu=gpu_on)

        if len(prediction) == 0:
            import pandas as pd
            pd.DataFrame(metrics_result).to_pickle(export_metrics_df_name)
            import ipdb; ipdb.set_trace()

        # TODO: Use the trained model with test set, get performance measures
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

        if len(prediction_valid_nonzero_intersect) + len(prediction_valid_zero_intersect) == 0:
            import pandas as pd
            pd.DataFrame(metrics_result).to_pickle(export_metrics_df_name)
            import ipdb; ipdb.set_trace()

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

    import pandas as pd
    pd.DataFrame(metrics_result).to_pickle(export_metrics_df_name)
    import ipdb; ipdb.set_trace()

    model.sess.close()
    tf.reset_default_graph()

    return metrics_result

def create_one_hot_vector(num_classes, nth_item):
    return np.eye(num_classes)[[nth_item]]

def create_one_hot_matrix(num_rows, num_classes, nth_item):
    target_row_index = np.full(num_rows, nth_item, dtype=int)
    return np.eye(num_classes)[target_row_index]

def random_sampling(num_rows, num_cols):
    return np.random.random((num_rows, num_cols))

def entropy_sampling(item_mu, user_mu, user_sigma):
    log_pdf = calculate_gaussian_log_pdf(item_mu, user_mu, user_sigma)
    return log_pdf

import math
# log_p(I_Mu|U_Mu, U_Sigma)
def multivariate_normal_log_pdf(x, mean, cov):
    return np.negative(np.sum(np.divide(np.square(x-mean), 2 * cov) + 0.5 * np.log(2 * math.pi * cov), axis=1))

def calculate_gaussian_log_pdf(item_mu, user_mu, user_sigma):
    result = []
    for user_index in range(len(user_mu)):
       result.append(multivariate_normal_log_pdf(x=item_mu, mean=user_mu[user_index], cov=np.square(user_sigma[user_index])))
#    return np.negative(np.sum(np.divide(np.square(item_gaussian_mu-user_gaussian_mu[0]), 2 * np.square(user_gaussian_sigma[0])) + 0.5 * np.log(2 * math.pi * np.square(user_gaussian_sigma[0])), axis=1))
#    np.log(multivariate_normal.pdf(x=item_gaussian_mu[0], mean=user_gaussian_mu[0], cov=np.square(user_gaussian_sigma[0])))
    return result

def sampling_predict(prediction_scores, topK, matrix_Train, gpu=False):
    prediction = []

    from tqdm import tqdm
    for user_index in tqdm(range(len(prediction_scores))):
        vector_train = matrix_Train[user_index]
        if len(vector_train.nonzero()[0]) > 0:
            vector_predict = sub_routine(prediction_scores[user_index], vector_train, topK=topK, gpu=gpu)
        else:
            vector_predict = np.zeros(topK, dtype=np.float32)

        # Return empty list when there is a user has less than topK items to
        # recommend. The main program will stop.
        if len(vector_predict) != topK:
            import ipdb; ipdb.set_trace()
            return []

        prediction.append(vector_predict)
    return np.vstack(prediction)

def sub_routine(vector_predict, vector_train, topK=500, gpu=False):

    train_index = vector_train.nonzero()[1]
    sort_length = topK + len(train_index)
    # print('original sort length is {}'.format(sort_length))

    if sort_length + 1 > len(vector_predict):
        sort_length = len(vector_predict) - 1
    # print('modified sort length is {}'.format(sort_length))

    if gpu:
        import cupy as cp
        candidate_index = cp.argpartition(-vector_predict, sort_length)[:sort_length]
        vector_predict = candidate_index[vector_predict[candidate_index].argsort()[::-1]]
        vector_predict = cp.asnumpy(vector_predict).astype(np.float32)
    else:
        candidate_index = np.argpartition(-vector_predict, sort_length)[:sort_length]
        vector_predict = candidate_index[vector_predict[candidate_index].argsort()[::-1]]
    vector_predict = np.delete(vector_predict, np.isin(vector_predict, train_index).nonzero()[0])

    return vector_predict[:topK]

