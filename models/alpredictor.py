from tqdm import tqdm

import numpy as np


def sampling_predict(prediction_scores, topK, matrix_train, gpu=False):
    prediction = []

    for user_index in tqdm(range(len(prediction_scores))):
        vector_train = matrix_train[user_index]
        if len(vector_train.nonzero()[0]) > 0:
            vector_predict = sub_routine(prediction_scores[user_index], vector_train, topK=topK, gpu=gpu)
        else:
            vector_predict = np.zeros(topK, dtype=np.float32)

        # Return empty list when there is a user has less than topK items to
        # recommend. The main program will stop.
        if len(vector_predict) != topK:
            raise ValueError('user {} has less than top {} items to recommend. Return empty list in this case.'.format(user_index, topK))
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

