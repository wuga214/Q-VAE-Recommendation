import numpy as np
from utils.progress import WorkSplitter, inhour
import argparse
import time
from utils.io import load_numpy, load_pandas, load_csv
from utils.argcheck import check_float_positive, check_int_positive, shape
from utils.modelnames import models
from models.predictor import predict, predict_batch
from evaluation.metrics import evaluate


# TODO: Move this to util dir eventually
from scipy.sparse import csr_matrix

def entropy_sampling(mean, sigma, num_rows):
    if mean[mean >= 1].size:
        print("THERE ARE GAUSSIAN PARAMETERS GREATER OR EQUAL TO 1. CHECK!!!")
    if mean[mean < 0].size:
        print("THERE ARE GAUSSIAN PARAMETERS GREATER OR EQUAL TO 1. CHECK!!!")
    entropy_scores = np.multiply(-mean, np.log2(mean))
    entropy_scores = np.nan_to_num(entropy_scores)
    return np.tile(entropy_scores, (num_rows, 1))

def random_sampling(mean, sigma, num_rows):
    return np.random.random((num_rows, mean.size))


def sampling_predict(prediction_scores, topK, matrix_Train, gpu=False):
    prediction = []

    from tqdm import tqdm
    for user_index in tqdm(range(prediction_scores.shape[0])):
        vector_train = matrix_Train[user_index]
        if len(vector_train.nonzero()[0]) > 0:
            vector_predict = sub_routine(prediction_scores[user_index], vector_train, topK=topK, gpu=gpu)
        else:
            vector_predict = np.zeros(topK, dtype=np.float32)

        prediction.append(vector_predict)

    return np.vstack(prediction)


def sub_routine(vector_predict, vector_train, topK=500, gpu=False):

    train_index = vector_train.nonzero()[1]
    if gpu:
        import cupy as cp
        candidate_index = cp.argpartition(-vector_predict, topK+len(train_index))[:topK+len(train_index)]
        vector_predict = candidate_index[vector_predict[candidate_index].argsort()[::-1]]
        vector_predict = cp.asnumpy(vector_predict).astype(np.float32)
    else:
        candidate_index = np.argpartition(-vector_predict, topK+len(train_index))[:topK+len(train_index)]
        vector_predict = candidate_index[vector_predict[candidate_index].argsort()[::-1]]
    vector_predict = np.delete(vector_predict, np.isin(vector_predict, train_index).nonzero()[0])

    return vector_predict[:topK]

def main(args):
    # Progress bar
    progress = WorkSplitter()

    # Show hyper parameter settings
    progress.section("Parameter Setting")
    print("Data Path: {0}".format(args.path))
    print("Train File Name: {0}".format(args.train))
    if args.validation:
        print("Valid File Name: {0}".format(args.valid))
    print("Algorithm: {0}".format(args.model))
    if args.item == True:
        mode = "Item-based"
    else:
        mode = "User-based"
    print("Mode: {0}".format(mode))
    print("Alpha: {0}".format(args.alpha))
    print("Rank: {0}".format(args.rank))
    print("Lambda: {0}".format(args.lamb))
    print("SVD/Alter Iteration: {0}".format(args.iter))
    print("Evaluation Ranking Topk: {0}".format(args.topk))
    print('Number of Steps to Evaluate: {}'.format(args.num_steps))
    print('Number of Recommendations in Each Step: {}'.format(args.num_rec))

    # Load Data
    progress.section("Loading Data")
    start_time = time.time()
    if args.shape is None:
        R_train = load_numpy(path=args.path, name=args.train)
    else:
        # R_train = load_pandas(path=args.path, name=args.train, shape=args.shape)
        R_train = load_csv(path=args.path, name=args.train, shape=args.shape)
    R_valid = load_numpy(path=args.path, name=args.valid)
    print("Elapsed: {0}".format(inhour(time.time() - start_time)))

    print("Train U-I Dimensions: {0}".format(R_train.shape))

    metrics_result = []

    # By default, only 1 step
    for i in range(args.num_steps):
        print('This is step {} \n'.format(i))

        progress.section("Train")
        # Train the model using the train set and get weights
        # TODO: Gaussian_params contains RQ. Need to be optimized here
        # TODO: Get probability per sample
        RQ, Yt, Bias, Gaussian_Params_mu, Gaussian_Params_sigma = models[args.model](R_train, embedded_matrix=np.empty((0)),
                                                                                     iteration=args.iter, rank=args.rank,
                                                                                     corruption=args.corruption, gpu_on=args.gpu,
                                                                                     lam=args.lamb, alpha=args.alpha, seed=args.seed, root=args.root)
        Y = Yt.T

        # print('U is \n {}'.format(RQ))
        # print('V is \n {}'.format(Y))
        # print('B is \n {}'.format(Bias))

        progress.section("Predict")

        # TODO: Select ‘k’ most-informative samples based on
        # per-sample-probabilities, i.e., those that the model was most
        # uncertain about regarding their labelling.
        prediction_scores = entropy_sampling(Gaussian_Params_mu, Gaussian_Params_sigma, R_train.shape[0])
        #prediction_scores = random_sampling(Gaussian_Params_mu, Gaussian_Params_sigma, R_train.shape[0])

        print(prediction_scores)
        prediction = sampling_predict(prediction_scores=prediction_scores,
                                      topK=args.topk,
                                      matrix_Train=R_train,
                                      gpu=args.gpu)

        # TODO: Use the trained model with test set, get performance measures
        if args.validation:
            progress.section("Create Metrics")
            start_time = time.time()

            metric_names = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision', 'MAP']
            result = evaluate(prediction, R_valid, metric_names, [args.topk])
            print("-")
            for metric in result.keys():
                print("{0}:{1}".format(metric, result[metric]))
            print("Elapsed: {0}".format(inhour(time.time() - start_time)))

        metrics_result.append(result)

        # TODO: Move these ‘k’ samples from the validation set to the train-set
        # and query their labels.
        index = np.tile(np.arange(prediction.shape[0]),(prediction.shape[1],1)).T
        index_prediction = np.dstack((index, prediction)).reshape((prediction.shape[0]*prediction.shape[1]), 2)
        index_valid = np.dstack((R_valid.nonzero()[0], R_valid.nonzero()[1]))[0]

        start_time = time.time()
        index_prediction_set = set([tuple(x) for x in index_prediction])
        index_valid_set = set([tuple(x) for x in index_valid])
        prediction_valid_intersect = np.array([x for x in index_prediction_set & index_valid_set])
        mask_row = np.array(prediction_valid_intersect)[:, 0]
        mask_col = np.array(prediction_valid_intersect)[:, 1]
        mask_data = np.full(len(prediction_valid_intersect), True)
        mask = csr_matrix((mask_data, (mask_row, mask_col)), shape=R_train.shape)
        R_train = R_train.tolil()
        R_valid = R_valid.tolil()
        R_train[mask] = 1
        R_valid[mask] = 0
        R_train = R_train.tocsr()
        R_valid = R_valid.tocsr()
        print("Elapsed for ravel: {0}".format(inhour(time.time() - start_time)))

#    import ipdb; ipdb.set_trace()



if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="LRec")

    parser.add_argument('--disable-item-item', dest='item', action='store_false')
    parser.add_argument('--disable-validation', dest='validation', action='store_false')
    parser.add_argument('-i', dest='iter', type=check_int_positive, default=1)
    parser.add_argument('-a', dest='alpha', type=check_float_positive, default=100.0)
    parser.add_argument('-l', dest='lamb', type=check_float_positive, default=100)
    parser.add_argument('-r', dest='rank', type=check_int_positive, default=100)
    parser.add_argument('-f', dest='root', type=check_float_positive, default=1)
    parser.add_argument('-c', dest='corruption', type=check_float_positive, default=0.5)
    parser.add_argument('-s', dest='seed', type=check_int_positive, default=1)
    parser.add_argument('-ns', dest='num_steps', type=check_int_positive, default=1)
    parser.add_argument('-nr', dest='num_rec', type=check_int_positive, default=2)
    parser.add_argument('-m', dest='model', default="WRMF")
    parser.add_argument('-d', dest='path', default="datax/")
    parser.add_argument('-t', dest='train', default='Rtrain.npz')
    parser.add_argument('-v', dest='valid', default='Rvalid.npz')
    parser.add_argument('-k', dest='topk', type=check_int_positive, default=50)
    parser.add_argument('-gpu', dest='gpu', action='store_true')
    parser.add_argument('--similarity', dest='sim_measure', default='Cosine')
    parser.add_argument('--shape', help="CSR Shape", dest="shape", type=shape, nargs=2)
    args = parser.parse_args()

    main(args)

# Will this actually hurt the performance? Maybe just uncover the rating. No
# need to penalize low rating. Just to be consistent with the dataset
# TODO: Pick Top 1 and 2 for each user from prediction. Convert those two picks
# to (1, 2 -> some hyperparameter such as -0.1 & 3, 4 -> 0 & 5 -> 1) and
# feedforward through encoder to get latent mu and sigma. Use that particular
# sampling method to predict. And continue


