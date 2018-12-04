import tensorflow as tf
import numpy as np
from tqdm import tqdm
from utils.progress import WorkSplitter, inhour
from scipy.sparse import vstack, hstack


class WRMF(object):
    def __init__(self,
                 num_users,
                 num_items,
                 embed_dim,
                 batch_size=500,
                 lam=0.01,
                 **unused):
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.lam = lam
        self.get_graph()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def get_graph(self):

        # Placehoders

        # M users
        self.observation = tf.placeholder(tf.float32, [None, None])
        self.alpha = tf.placeholder(tf.float32)
        self.user_idx = tf.placeholder(tf.int32, [2])
        self.item_idx = tf.placeholder(tf.int32, [2])

        # Variable to learn
        self.user_embeddings = tf.Variable(tf.random_normal([self.num_users, self.embed_dim],
                                                            stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))

        self.item_embeddings = tf.Variable(tf.random_normal([self.num_items, self.embed_dim],
                                                            stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))

        users = self.user_embeddings[self.user_idx[0]:self.user_idx[1]]
        items = self.item_embeddings[self.item_idx[0]:self.item_idx[1]]

        with tf.variable_scope('loss'):
            confidence = 1 + self.alpha * self.observation

            user_l2 = tf.nn.l2_loss(users)
            item_l2 = tf.nn.l2_loss(items)

            mse = tf.square(self.observation-tf.matmul(users, tf.transpose(items)))

            loss_user = tf.reduce_mean(confidence * mse) + self.lam * tf.reduce_mean(user_l2)
            loss_item = tf.reduce_mean(confidence * mse) + self.lam * tf.reduce_mean(item_l2)

        with tf.variable_scope('optimizer'):
            self.optimizer_user = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(loss_user,
                                                                    var_list=[self.user_embeddings])
            self.optimizer_item = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(loss_item,
                                                                    var_list=[self.item_embeddings])

            # self.optimizer_user = tf.contrib.opt.ScipyOptimizerInterface(loss=loss_user,
            #                                                              method='L-BFGS-B',
            #                                                              var_list=[self.user_embeddings],
            #                                                              options={'maxiter': 10})
            #
            # self.optimizer_item = tf.contrib.opt.ScipyOptimizerInterface(loss=loss_item,
            #                                                              method='L-BFGS-B',
            #                                                              var_list=[self.item_embeddings],
            #                                                              options={'maxiter': 10})


    def get_batches(self, rating_matrix, batch_size, item=True):
        batch_index=0
        batches = []
        if item:
            remaining_size = rating_matrix.shape[1]
            while(remaining_size>0):
                if remaining_size<batch_size:
                    batches.append([rating_matrix[:, batch_index*batch_size:],
                                    [batch_index*batch_size, self.num_items]])
                else:
                    batches.append([rating_matrix[:, batch_index*batch_size:(batch_index+1)*batch_size],
                                   [batch_index*batch_size, (batch_index+1)*batch_size]])
                batch_index += 1
                remaining_size -= batch_size
        else:
            remaining_size = rating_matrix.shape[0]
            while(remaining_size>0):
                if remaining_size<batch_size:
                    batches.append([rating_matrix[batch_index*batch_size:],
                                    [batch_index*batch_size, self.num_users]])
                else:
                    batches.append([rating_matrix[batch_index*batch_size:(batch_index+1)*batch_size],
                                   [batch_index*batch_size, (batch_index+1)*batch_size]])
                batch_index += 1
                remaining_size -= batch_size
        return batches

    def train_model(self, rating_matrix, alpha=1.0, iter=7, epoch=20):
        user_batches = self.get_batches(rating_matrix, self.batch_size, item=False)
        item_batches = self.get_batches(rating_matrix, self.batch_size, item=True)

        # Training
        pbar = tqdm(range(iter))
        for i in pbar:

            for _ in tqdm(range(epoch)):
                for step in tqdm(range(len(user_batches))):
                    feed_dict = {self.observation: user_batches[step][0].todense(),
                                 self.user_idx: user_batches[step][1],
                                 self.item_idx: [0, self.num_items],
                                 self.alpha: alpha}
                    self.sess.run([self.optimizer_user], feed_dict=feed_dict)

                    # self.optimizer_user.minimize(session=self.sess,
                    #                              feed_dict=feed_dict)

            for _ in tqdm(range(epoch)):
                for step in tqdm(range(len(item_batches))):
                    feed_dict = {self.observation: item_batches[step][0].todense(),
                                 self.user_idx: [0, self.num_users],
                                 self.item_idx: item_batches[step][1],
                                 self.alpha: alpha}
                    self.sess.run([self.optimizer_item], feed_dict=feed_dict)
                    # self.optimizer_item.minimize(session=self.sess,
                    #                              feed_dict=feed_dict)

    def get_RQ(self):
        return self.sess.run(self.user_embeddings)

    def get_Y(self):
        return self.sess.run(self.item_embeddings)


def als_tf(matrix_train, embeded_matrix=np.empty((0)), iteration=4, lam=80, rank=200, alpha=100, **unused):
    progress = WorkSplitter()
    matrix_input = matrix_train
    if embeded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embeded_matrix.T))

    m, n = matrix_input.shape
    model = WRMF(num_users=m, num_items=n, embed_dim=rank, lam=lam)

    model.train_model(matrix_input, alpha, iteration)

    RQ = model.get_RQ()
    Y = model.get_Y().T
    model.sess.close()
    tf.reset_default_graph()

    return RQ, Y, None










