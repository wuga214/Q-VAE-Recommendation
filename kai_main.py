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
# Normalize input matrix by column
from scipy.sparse import csr_matrix
def normalize_matrix_by_column(matrix):
    matrix = matrix.tobsr()
    sum_over_column = matrix.sum(axis=0)
    sum_over_column[0][sum_over_column[0] == 0] = 1
    return csr_matrix(matrix/sum_over_column)

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
    print("Elapsed: {0}".format(inhour(time.time() - start_time)))

    print("Train U-I Dimensions: {0}".format(R_train.shape))

    '''
    # Item-Item or User-User
    if args.item == True:
        RQ, Yt, Bias = models[args.model](R_train, embedded_matrix=np.empty((0)),
                                          iteration=args.iter, rank=args.rank,
                                          corruption=args.corruption, gpu_on=args.gpu,
                                          lam=args.lamb, alpha=args.alpha, seed=args.seed, root=args.root)
        Y = Yt.T
    else:
        Y, RQt, Bias = models[args.model](R_train.T, embedded_matrix=np.empty((0)),
                                          iteration=args.iter, rank=args.rank,
                                          corruption=args.corruption, gpu_on=args.gpu,
                                          lam=args.lamb, alpha=args.alpha, seed=args.seed, root=args.root)
        RQ = RQt.T

    # np.save('latent/U_{0}_{1}'.format(args.model, args.rank), RQ)
    # np.save('latent/V_{0}_{1}'.format(args.model, args.rank), Y)
    # if Bias is not None:
    #     np.save('latent/B_{0}_{1}'.format(args.model, args.rank), Bias)

    progress.section("Predict")
    prediction = predict(matrix_U=RQ,
                         matrix_V=Y,
                         bias=Bias,
                         topK=args.topk,
                         matrix_Train=R_train,
                         measure=args.sim_measure,
                         gpu=args.gpu)
    '''

    # By default, only 1 step
    for i in range(args.num_steps):
        print('This is step {} \n'.format(i))

        # Normalize train set
        R_train_normalized = normalize_matrix_by_column(R_train)

        # Train the model using the train set and get weights
        # TODO: Gaussian_params contains RQ. Need to be optimized here
        # TODO: Get probability per sample
        RQ, Yt, Bias, Gaussian_Params_mu, Gaussian_Params_sigma = models[args.model](R_train_normalized, embedded_matrix=np.empty((0)),
                                                                                     iteration=args.iter, rank=args.rank,
                                                                                     corruption=args.corruption, gpu_on=args.gpu,
                                                                                     lam=args.lamb, alpha=args.alpha, seed=args.seed, root=args.root)
        Y = Yt.T

        # TODO: Use the trained model with test set, get performance measures

        # TODO: Select ‘k’ most-informative samples based on
        # per-sample-probabilities, i.e., those that the model was most
        # uncertain about regarding their labelling.
        # TODO: Move these ‘k’ samples from the validation set to the train-set
        # and query their labels.
        # TODO: Inverse normalization for all the data-sets
        # TODO: Stop according to the stop criterion, otherwise normalize train
        # set.


        progress.section("Predict")
        prediction = predict(matrix_U=RQ,
                             matrix_V=Y,
                             bias=Bias,
                             topK=args.topk,
                             matrix_Train=R_train_normalized,
                             measure=args.sim_measure,
                             gpu=args.gpu)
        print('U is \n {}'.format(RQ))
        print('V is \n {}'.format(Y))
        print('B is \n {}'.format(Bias))
        print('Prediction is \n {}'.format(prediction))

    if args.validation:
        progress.section("Create Metrics")
        start_time = time.time()

        metric_names = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision', 'MAP']
        R_valid = load_numpy(path=args.path, name=args.valid)
        result = evaluate(prediction, R_valid, metric_names, [args.topk])
        print("-")
        for metric in result.keys():
            print("{0}:{1}".format(metric, result[metric]))
        print("Elapsed: {0}".format(inhour(time.time() - start_time)))


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


