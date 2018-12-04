import numpy as np
from utils.progress import WorkSplitter, inhour
import argparse
import time
from utils.io import load_numpy, load_pandas, load_csv
from utils.argcheck import check_float_positive, check_int_positive, shape
from utils.modelnames import models
from models.predictor import predict,predict_batch
from evaluation.metrics import evaluate


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

    # Load Data
    progress.section("Loading Data")
    start_time = time.time()
    if args.shape is None:
        R_train = load_numpy(path=args.path, name=args.train)
    else:
        # R_train = load_pandas(path=args.path, name=args.train, shape=args.shape)
        R_train = load_csv(path=args.path, name=args.train, shape=args.shape)
    print "Elapsed: {0}".format(inhour(time.time() - start_time))

    print("Train U-I Dimensions: {0}".format(R_train.shape))

    # Item-Item or User-User
    if args.item == True:
        RQ, Yt, Bias = models[args.model](R_train, embeded_matrix=np.empty((0)),
                                          iteration=args.iter, rank=args.rank,
                                          corruption=args.corruption, gpu_on=args.gpu,
                                          lam=args.lamb, alpha=args.alpha, seed=args.seed, root=args.root)
        Y = Yt.T
    else:
        Y, RQt, Bias = models[args.model](R_train.T, embeded_matrix=np.empty((0)),
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
    if args.validation:
        progress.section("Create Metrics")
        start_time = time.time()

        metric_names = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision']
        R_valid = load_numpy(path=args.path, name=args.valid)
        result = evaluate(prediction, R_valid, metric_names, [args.topk])
        print("-")
        for metric in result.keys():
            print("{0}:{1}".format(metric, result[metric]))
        print "Elapsed: {0}".format(inhour(time.time() - start_time))


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