from tqdm import tqdm

import numpy as np


def sampling_predict(prediction_scores, topK, matrix_train, gpu=False):
    prediction = []

    for user_index in tqdm(range(len(prediction_scores))):
        vector_train = matrix_train[user_index]
        vector_predict = sub_routine(prediction_scores[user_index], vector_train, topK=topK, gpu=gpu)
        '''
        if len(vector_train.nonzero()[0]) > 0:
            vector_predict = sub_routine(prediction_scores[user_index], vector_train, topK=topK, gpu=gpu)
        else:
            vector_predict = np.zeros(topK, dtype=np.float32)
        '''
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


def get_latent_gaussian_params(model, is_item, size=None, matrix=None):
    if is_item:
        matrix = np.diag(np.ones(size))
    return model.uncertainty(matrix)

def predict_gaussian_prob(item_mu, user_mu, user_sigma, model, matrix, latent=True):
    if latent:
        return logsumexp_pdf(item_mu, user_mu, user_sigma)
    else:
        return model.inference(matrix.A)

def logsumexp_pdf(item_mu, user_mu, user_sigma):
    log_pdf = calculate_gaussian_log_pdf(item_mu.astype(np.float64), user_mu.astype(np.float64), user_sigma.astype(np.float64))
    # from scipy.stats import multivariate_normal
    # scipy_scipy = [multivariate_normal.pdf(x=item, mean=user_mu[0], cov=np.square(user_sigma[0])) for item in item_mu]
    # import ipdb; ipdb.set_trace()
    A = np.amax(log_pdf, axis=1)
    return np.exp(log_pdf-np.vstack(A))

def calculate_gaussian_log_pdf(item_mu, user_mu, user_sigma):
    result = []
    for user_index in range(len(user_mu)):
        result.append(multivariate_normal_log_pdf(x=item_mu, mean=user_mu[user_index], cov=np.square(user_sigma[user_index])))
        # return np.negative(np.sum(np.divide(np.square(item_gaussian_mu-user_gaussian_mu[0]), 2 * np.square(user_gaussian_sigma[0])) + 0.5 * np.log(2 * math.pi * np.square(user_gaussian_sigma[0])), axis=1))
        # np.log(multivariate_normal.pdf(x=item_gaussian_mu[0], mean=user_gaussian_mu[0], cov=np.square(user_gaussian_sigma[0])))
    return result

# log_p(I_Mu|U_Mu, U_Sigma)
def multivariate_normal_log_pdf(x, mean, cov):
    return np.negative(np.sum(np.divide(np.square(x-mean), 2 * cov) + 0.5 * np.log(2 * np.pi * cov), axis=1))

def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = linalg.det(sigma)
        if det == 0:
            raise NameError("The covariance matrix can't be singular")

        norm_const = 1.0/ ( math.pow((2*pi),float(size)/2) * math.pow(det,1.0/2) )
        x_mu = matrix(x - mu)
        inv = sigma.I
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")

