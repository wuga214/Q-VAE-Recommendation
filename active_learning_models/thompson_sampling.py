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


def update_matrix(history_items, matrix_train, matrix_active, observation, train_index, iterative, sample_all, num_item_per_iter, al_iteration, gpu):
    if iterative:
        topk = num_item_per_iter
    else:
        topk = num_item_per_iter * al_iteration

    predict_items, history_items = sampling_predict(prediction_scores=observation,
                                     topK=topk,
                                     matrix_train=matrix_train[train_index:],
                                     matrix_active=matrix_active[train_index:],
                                     sample_all=sample_all,
                                     iterative=iterative,
                                     history_items=history_items,
                                     gpu=gpu)

    # import ipdb; ipdb.set_trace()


    mask_row = train_index + np.repeat(np.arange(len(predict_items)), topk)
    mask_col = predict_items.ravel()
    mask_data = np.full(len(predict_items)*topk, True)
    mask = csr_matrix((mask_data, (mask_row, mask_col)), shape=matrix_train.shape)
    matrix_train[mask] = matrix_active[mask]

    return matrix_train, history_items


# Remove users who have less than 2*num_item_per_iter*al_iteration positive
# ratings in active set from active and test set.
# Remove users who have less than 2*topk positive ratings in active set
# from active and test set.
def filter_users(matrix_train, matrix_active, matrix_test, train_index, active_threshold, test_threshold):
    active_user_num_nonzero = np.array(matrix_active.sum(axis=1)).ravel()
    active_users = np.where(active_user_num_nonzero >= active_threshold)[0]

    users = np.concatenate([np.arange(train_index), active_users])

    return matrix_train[users,:], matrix_active[users,:], matrix_test[users,:], users

#    test_user_num_nonzero = np.array(matrix_test.sum(axis=1)).ravel()
#    test_users = np.where(test_user_num_nonzero >= test_threshold)[0]

#    test_users = np.intersect1d(active_users, test_users)
#    users = np.concatenate([np.arange(train_index), test_users])
#    return matrix_train[users,:], matrix_active[users,:], matrix_test[users,:], users


def thompson_sampling(matrix_train, matrix_active, matrix_test, rec_model, topk,
                      train_index, al_iteration, latent, iterative, sample_all,
                      iteration=100, rank=200, corruption=0.2, gpu=True, lam=80,
                      optimizer="RMSProp", num_item_per_iter=1, **unused):

    progress = WorkSplitter()

    matrix_train, matrix_active, matrix_test, _ = filter_users(matrix_train,
                                                               matrix_active,
                                                               matrix_test,
                                                               train_index,
                                                               active_threshold=2*num_item_per_iter*al_iteration,
                                                               test_threshold=2*topk)

    m, n = matrix_train.shape

    metrics_result = []
    history_items = np.array([])

    model = vaes[rec_model](n, rank, 100, lamb=lam, optimizer=Regularizer[optimizer])

    progress.section("Training")
    model.train_model(matrix_train[:train_index], corruption, iteration)


    for i in range(al_iteration):
        print('This is step {} \n'.format(i))
        print('The number of ones in train set is {}'.format(len(matrix_train[train_index:].nonzero()[0])))
        print('The number of ones in active set is {}'.format(len(matrix_active[train_index:].nonzero()[0])))

#        import ipdb; ipdb.set_trace()
        progress.section("Predicting")
        observation = model.inference(matrix_train[train_index:].A, sampling=latent)

        # import ipdb; ipdb.set_trace()

        progress.section("Update Train Set")
        matrix_train, history_items = update_matrix(history_items, matrix_train, matrix_active, observation, train_index, iterative, sample_all, num_item_per_iter, al_iteration, gpu)

        if not iterative:
            break

    print('The number of ones in train set is {}'.format(len(matrix_train[train_index:].nonzero()[0])))

    progress.section("Re-Training")
    model.train_model(matrix_train, corruption, iteration)

    progress.section("Re-Predicting")
    observation = model.inference(matrix_train.A, sampling=latent)

    result = {}
    for topk in [1, 5, 10, 50]:
        predict_items, _ = sampling_predict(prediction_scores=observation[train_index:],
                                        topK=topk,
                                        matrix_train=matrix_train[train_index:],
                                        matrix_active=matrix_active[train_index:],
                                        sample_all=True,
                                        iterative=False,
                                        history_items=np.array([]),
                                        gpu=gpu)

        progress.section("Create Metrics")
        result.update(eval(matrix_test[train_index:], topk, predict_items))

    metrics_result.append(result)

    model.sess.close()
    tf.reset_default_graph()

    return metrics_result

