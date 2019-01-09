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
                 beta=0.2,
                 learning_rate=1e-4,
                 optimizer=tf.train.RMSPropOptimizer,
                 observation_distribution="Gaussian", # or Bernoulli or Multinomial
                 observation_std=0.01):

        self._lamb = lamb
        self._latent_dim = latent_dim
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

            mask1 = tf.nn.dropout(tf.ones_like(self.input), 1-self.corruption)
            mask2 = tf.nn.dropout(tf.ones_like(self.input), 1-self.corruption)

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
                                                        tf.ones_like(hc_log_std))

                if self._observation_distribution == 'Gaussian':
                    with tf.variable_scope('gaussian'):
                        obj1 = self._gaussian_log_likelihood(self.input * tf.floor(mask1),
                                                             self.wc_obs_mean,
                                                             self._observation_std)
                        obj2 = self._gaussian_log_likelihood(self.input * tf.floor(mask2),
                                                             self.hc_obs_mean,
                                                             self._observation_std)
                elif self._observation_distribution == 'Bernoulli':
                    with tf.variable_scope('bernoulli'):
                        obj1 = self._bernoulli_log_likelihood(self.input * tf.floor(mask1) * (1 - tf.floor(mask1)),
                                                              self.wc_obs_mean * tf.floor(mask1) * (1 - tf.floor(mask1)))
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

                self._loss = ((kl1 + kl2 + obj1 + obj2) + self._lamb * l2_loss) * (1 - self.corruption)

            with tf.variable_scope('optimizer'):
                optimizer = self._optimizer(learning_rate=self._learning_rate)
            with tf.variable_scope('training-step'):
                self._train = optimizer.minimize(self._loss)

            self.sess = tf.Session()
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def _network(self, x):
        with tf.variable_scope('encoder'):
            self.encode_weights = tf.Variable(tf.truncated_normal([self._observation_dim, self._latent_dim * 2],
                                                                   stddev=1 / 500.0),
                                         name="Weights")
            encode_bias = tf.Variable(tf.constant(0., shape=[self._latent_dim * 2]), name="Bias")

            encoded = tf.matmul(x, self.encode_weights) + encode_bias

        with tf.variable_scope('latent'):
            mean = tf.nn.relu(encoded[:, :self._latent_dim])
            logstd = encoded[:, self._latent_dim:]
            std = tf.exp(logstd)
            epsilon = tf.random_normal(tf.shape(std))
            z = mean

            z = tf.cond(self.sampling, lambda: z + std * epsilon, lambda: z)

        with tf.variable_scope('decoder'):
            self.decode_weights = tf.Variable(
                tf.truncated_normal([self._latent_dim, self._observation_dim], stddev=1 / 500.0),
                name="Weights")
            self.decode_bias = tf.Variable(tf.constant(0., shape=[self._observation_dim]), name="Bias")
            decoded = tf.matmul(z, self.decode_weights) + self.decode_bias

            obs_mean = decoded

        return mean, logstd, obs_mean

    # @staticmethod
    # def _kl_diagnormal_stdnormal(mu_1, std_1, mu_2=0, std_2=1):
    #
    #     kl = tf.reduce_mean(tf.log(std_2) - tf.log(std_1)
    #                         + tf.divide((tf.square(std_1) + tf.square(mu_1-mu_2)), 2*tf.square(std_2))
    #                         - 0.5)
    #     return kl

    @staticmethod
    def _kl_diagnormal_stdnormal(mu_1, log_std_1, mu_2=0, log_std_2=0):

        log_std_2 = tf.where(tf.less(0., log_std_2), tf.zeros(tf.shape(log_std_2)), log_std_2)

        var_square_1 = tf.exp(2. * log_std_1)
        var_square_2 = tf.exp(2. * log_std_2)
        kl = 0.5 * tf.reduce_mean(2 * log_std_2 - 2. * log_std_1
                                  + tf.divide(var_square_1 + tf.square(mu_1-mu_2), var_square_2) - 1.)
        return kl

    @staticmethod
    def _gaussian_log_likelihood(targets, mean, std):
        se = 0.5 * tf.reduce_mean(tf.square(targets - mean)) / (2*tf.square(std)) + tf.log(std)
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
                corrupt_rate = random.uniform(0.1, 0.5)
                feed_dict = {self.input: batches[step].todense(), self.corruption: corrupt_rate, self.sampling: True}
                training, loss = self.sess.run([self._train, self.wc_std], feed_dict=feed_dict)
                pbar.set_description("loss: {:.4f}".format(np.mean(loss)))

    def get_batches(self, rating_matrix, batch_size):
        remaining_size = rating_matrix.shape[0]
        batch_index = 0
        batches = []
        while remaining_size > 0:
            if remaining_size < batch_size:
                batches.append(rating_matrix[batch_index*batch_size:])
            else:
                batches.append(rating_matrix[batch_index*batch_size:(batch_index+1)*batch_size])
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


def ifvae(matrix_train, embeded_matrix=np.empty((0)), iteration=100,
          lam=80, rank=200, corruption=0.2, optimizer="RMSProp", seed=1, **unused):
    progress = WorkSplitter()
    matrix_input = matrix_train
    if embeded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embeded_matrix.T))

    m, n = matrix_input.shape
    model = IFVAE(n, rank, 100, lamb=lam, observation_distribution="Gaussian", optimizer=Regularizer[optimizer])

    model.train_model(matrix_input, corruption, iteration)

    RQ = model.get_RQ(matrix_input)
    Y = model.get_Y()
    Bias = model.get_Bias()
    model.sess.close()
    tf.reset_default_graph()

    return RQ, Y, Bias