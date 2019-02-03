from utils.progress import WorkSplitter, inhour
from models.ifvae import IFVAE
from tqdm import tqdm
from models.alpredictor import sampling_predict
from utils.regularizers import Regularizer

import math
import numpy as np


def create_one_hot_vector(num_classes, nth_item):
    return np.eye(num_classes)[[nth_item]]

def entropy_sampling(item_mu, user_mu, user_sigma):
    log_pdf = calculate_gaussian_log_pdf(item_mu, user_mu, user_sigma)
    entropy = np.negative(np.multiply(np.exp(log_pdf), (log_pdf / np.log(2))) + np.multiply(1-np.exp(log_pdf), np.log2(1-np.exp(log_pdf))))
    return entropy

def calculate_gaussian_log_pdf(item_mu, user_mu, user_sigma):
    result = []
    for user_index in range(len(user_mu)):
       result.append(multivariate_normal_log_pdf(x=item_mu, mean=user_mu[user_index], cov=np.square(user_sigma[user_index])))
#    return np.negative(np.sum(np.divide(np.square(item_gaussian_mu-user_gaussian_mu[0]), 2 * np.square(user_gaussian_sigma[0])) + 0.5 * np.log(2 * math.pi * np.square(user_gaussian_sigma[0])), axis=1))
#    np.log(multivariate_normal.pdf(x=item_gaussian_mu[0], mean=user_gaussian_mu[0], cov=np.square(user_gaussian_sigma[0])))
    return result

# log_p(I_Mu|U_Mu, U_Sigma)
def multivariate_normal_log_pdf(x, mean, cov):
    return np.negative(np.sum(np.divide(np.square(x-mean), 2 * cov) + 0.5 * np.log(2 * math.pi * cov), axis=1))

def entropy(matrix_train, matrix_valid, topk, total_steps,
            retrain_interval, validation, embedded_matrix=np.empty((0)),
            iteration=100, rank=200, corruption=0.2, gpu_on=True, lam=80,
            optimizer="RMSProp", beta=1.0, **unused):

    progress = WorkSplitter()
    matrix_input = matrix_train
    if embedded_matrix.shape[0] > 0:
        matrix_input = vstack((matrix_input, embedded_matrix.T))

    matrix_input_with_negative = matrix_input.copy()

    m, n = matrix_input.shape

    metrics_result = []

    model = IFVAE(n, rank, 100, lamb=lam, beta=beta,
                  observation_distribution="Gaussian", optimizer=Regularizer[optimizer])

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
        # Get all user distribution by feedforward passing user vector through
        # encoder
        user_gaussian_mu, user_gaussian_sigma = [], []
        for nth_user in tqdm(range(m)):
            user_vector = matrix_input_with_negative[nth_user, :]
            Gaussian_Params = model.uncertainty(user_vector.todense())
            user_gaussian_mu.append(Gaussian_Params[0][0])
            user_gaussian_sigma.append(Gaussian_Params[1][0])
        user_gaussian_mu, user_gaussian_sigma = np.array(user_gaussian_mu), np.array(user_gaussian_sigma)

        progress.section("Sampling")
        prediction_scores = entropy_sampling(item_gaussian_mu, user_gaussian_mu, user_gaussian_sigma)

        prediction = sampling_predict(prediction_scores=prediction_scores,
                                      topK=topk,
                                      matrix_train=matrix_input,
                                      gpu=gpu_on)

        if len(prediction) == 0:
            return metrics_result

        # TODO: Use the trained model with test set, get performance measures
        if validation:
            progress.section("Create Metrics")
            import time
            start_time = time.time()

            from evaluation.metrics import evaluate
            metric_names = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision', 'MAP']

            evaluation_items = sampling_predict(prediction_scores=-prediction_scores,
                                                topK=topk,
                                                matrix_train=matrix_input,
                                                gpu=gpu_on)
            result = evaluate(evaluation_items, matrix_valid, metric_names, [topk])

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

    model.sess.close()
    tf.reset_default_graph()

    return metrics_result

