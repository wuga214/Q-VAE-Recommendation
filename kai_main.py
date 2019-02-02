import numpy as np
from utils.progress import WorkSplitter, inhour
import argparse
import time
from utils.io import load_numpy, load_pandas, load_csv
from utils.argcheck import check_float_positive, check_int_positive, shape
from utils.modelnames import models
from models.predictor import predict, predict_batch
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
    print("Active Learning Algorithm: {0}".format(args.al_model))
    print("Retrain Interval: {0}".format(args.retrain_interval))
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
    print("GPU: {}".format(args.gpu))
    print('Number of Steps to Evaluate: {}'.format(args.total_steps))

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

    metrics_result = models[args.model](R_train, R_valid, topk=args.topk,
                                        al_model=args.al_model,
                                        total_steps=args.total_steps,
                                        retrain_interval=args.retrain_interval,
                                        validation=args.validation,
                                        embedded_matrix=np.empty((0)),
                                        iteration=args.iter, rank=args.rank,
                                        corruption=args.corruption, gpu_on=args.gpu,
                                        lam=args.lamb, alpha=args.alpha,
                                        seed=args.seed, root=args.root)

    # import ipdb; ipdb.set_trace()



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
    parser.add_argument('-ts', dest='total_steps', type=check_int_positive, default=1)
    parser.add_argument('-m', dest='model', default="WRMF")
    parser.add_argument('-alm', dest='al_model', default="Random")
    parser.add_argument('-ri', dest='retrain_interval', type=check_int_positive, default=0)
    parser.add_argument('-d', dest='path', default="datax/")
    parser.add_argument('-t', dest='train', default='Rtrain.npz')
    parser.add_argument('-v', dest='valid', default='Rvalid.npz')
    parser.add_argument('-k', dest='topk', type=check_int_positive, default=50)
    parser.add_argument('-gpu', dest='gpu', action='store_false')
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


