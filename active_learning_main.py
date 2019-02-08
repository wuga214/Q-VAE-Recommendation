import numpy as np
import pandas as pd
from utils.progress import WorkSplitter, inhour
import argparse
import time
from utils.io import load_numpy, load_pandas, load_csv
from utils.argcheck import check_float_positive, check_int_positive, shape
from utils.modelnames import models
from utils.active_learning_model_names import active_learning_models
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
    print("Test File Name: {0}".format(args.test))
    print("Algorithm: {0}".format(args.model))
    print("Active Learning Algorithm: {0}".format(args.active_learning_model))
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
        print("Train U-I Dimensions: {0}".format(R_train.shape))

    if args.validation:
        R_valid = load_numpy(path=args.path, name=args.valid)
        print("Valid U-I Dimensions: {0}".format(R_valid.shape))

    R_test = load_numpy(path=args.path, name=args.test)
    print("Test U-I Dimensions: {0}".format(R_test.shape))

    print("Elapsed: {0}".format(inhour(time.time() - start_time)))

    metrics_result = active_learning_models[args.active_learning_model](R_train, R_valid, topk=args.topk,
                                              total_steps=args.total_steps,
                                              validation=args.validation,
                                              embedded_matrix=np.empty((0)),
                                              iteration=args.iter, rank=args.rank,
                                              corruption=args.corruption, gpu_on=args.gpu,
                                              lam=args.lamb, alpha=args.alpha,
                                              seed=args.seed, root=args.root)

    import ipdb; ipdb.set_trace()

    export_metrics_df_name = args.active_learning_model + "_" + str(args.total_steps) + "steps_" + str(args.topk) + "items_per_step_per_user_retrain_every_" + str(args.retrain_interval) + "steps"
    pd.DataFrame(metrics_result).to_pickle(export_metrics_df_name)


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Deep_Preference_Elicitation")

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
    parser.add_argument('-m', dest='model', default="IFVAE")
    parser.add_argument('-alm', dest='active_learning_model', default="Random")
    parser.add_argument('-d', dest='path', default="datax/")
    parser.add_argument('-tr', dest='train', default='Rtrain.npz')
    parser.add_argument('-v', dest='valid', default='Rvalid.npz')
    parser.add_argument('-te', dest='test', default='Rtest.npz')
    parser.add_argument('-k', dest='topk', type=check_int_positive, default=50)
    parser.add_argument('-gpu', dest='gpu', action='store_false')
    parser.add_argument('--similarity', dest='sim_measure', default='Cosine')
    parser.add_argument('--shape', help="CSR Shape", dest="shape", type=shape, nargs=2)
    args = parser.parse_args()

    main(args)

