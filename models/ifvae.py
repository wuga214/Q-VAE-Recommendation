import re
import tensorflow as tf


class IFVAE(object):

    def __init__(self, latent_dim, batch_size, encoder, decoder,
                 observation_dim=784,
                 learning_rate=1e-4,
                 optimizer=tf.train.RMSPropOptimizer,
                 observation_distribution="Bernoulli", # or Gaussian
                 observation_std=0.01):

        self._latent_dim = latent_dim
        self._batch_size = batch_size
        self._encode = encoder
        self._decode = decoder
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
                self.wc_mean, self.wc_std, self.wc_obs_mean = self._network(wc)
            with tf.variable_scope('network', reuse=True):
                self.hc_mean, self.hc_std, self.hc_obs_mean = self._network(hc)

            with tf.variable_scope('loss'):
                with tf.variable_scope('kl-divergence'):
                    kl1 = self._kl_diagnormal_stdnormal(self.wc_mean, self.wc_std, self.hc_mean, self.hc_std)
                    kl2 = self._kl_diagnormal_stdnormal(self.hc_mean, self.hc_std, tf.zeros_like(self.hc_mean), tf.ones_like(self.hc_std))

                if self._observation_distribution == 'Gaussian':
                    with tf.variable_scope('gaussian'):
                        obj1 = self._gaussian_log_likelihood(self.input*tf.floor(mask1),
                                                             self.wc_obs_mean,
                                                             self._observation_std)
                        obj2 = self._gaussian_log_likelihood(self.input*tf.floor(mask2),
                                                             self.hc_obs_mean,
                                                             self._observation_std)
                else:
                    with tf.variable_scope('bernoulli'):
                        obj1 = self._bernoulli_log_likelihood(self.input * tf.floor(mask1), self.wc_obs_mean)
                        obj2 = self._bernoulli_log_likelihood(self.input * tf.floor(mask2), self.hc_obs_mean)

                self._loss = (kl1 + kl2 + obj1 + obj2) / self._batch_size

            with tf.variable_scope('optimizer'):
                optimizer = tf.train.RMSPropOptimizer(learning_rate=self._learning_rate)
            with tf.variable_scope('training-step'):
                self._train = optimizer.minimize(self._loss)

            self._sesh = tf.Session()
            init = tf.global_variables_initializer()
            self._sesh.run(init)

    def _network(self, x):
        with tf.variable_scope('encoder'):
            encoded = self._encode(x, self._latent_dim)

        with tf.variable_scope('latent'):
            mean = encoded[:, :self._latent_dim]
            logvar = encoded[:, self._latent_dim:]
            std = tf.sqrt(tf.exp(logvar))
            epsilon = tf.random_normal([self._batch_size, self._latent_dim])
            z = mean

            z = tf.cond(self.sampling, lambda: z + std * epsilon, lambda: z)

        with tf.variable_scope('decoder'):
            decoded = self._decode(z, self._observation_dim)
            obs_mean = decoded

        return mean, std, obs_mean#, sample

    @staticmethod
    def _kl_diagnormal_stdnormal(mu_1, std_1, mu_2=0, std_2=1):

        kl = tf.reduce_sum(tf.log(std_2) - tf.log(std_1)
                           + tf.divide((tf.square(std_1) + tf.square(mu_1-mu_2)), 2*tf.square(std_2))
                           - 0.5)
        return kl

    @staticmethod
    def _gaussian_log_likelihood(targets, mean, std):
        se = 0.5 * tf.reduce_sum(tf.square(targets - mean)) / (2*tf.square(std)) + tf.log(std)
        return se

    @staticmethod
    def _bernoulli_log_likelihood(targets, outputs, eps=1e-8):

        log_like = -tf.reduce_sum(targets * tf.log(outputs + eps)
                                  + (1. - targets) * tf.log((1. - outputs) + eps))
        return log_like

    def update(self, x, corruption):
        _, loss = self._sesh.run([self._train, self._loss],
                                 feed_dict={self.input: x, self.corruption: corruption, self.sampling: True})
        return loss

    def inference(self, x):
        predict = self._sesh.run(self.wc_obs_mean,
                                 feed_dict={self.input: x, self.corruption: 0, self.sampling: False})
        return predict

    def save_generator(self, path, prefix="in/generator"):
        variables = tf.trainable_variables()
        var_dict = {}
        for v in variables:
            if "decoder" in v.name:
                name = prefix+re.sub("vae/decoder", "", v.name)
                name = re.sub(":0", "", name)
                var_dict[name] = v
        for k, v in var_dict.items():
            print(k)
            print(v)
        saver = tf.train.Saver(var_dict)
        saver.save(self._sesh, path)