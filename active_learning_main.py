import numpy as np
import pandas as pd
from utils.progress import WorkSplitter, inhour
import argparse
import time
from utils.io import load_numpy, load_pandas, load_csv
from utils.argcheck import check_float_positive, check_int_positive, shape, ratio
from utils.modelnames import models
from utils.active_learning_model_names import active_learning_models
from predict.predictor import predict, predict_batch
from evaluation.metrics import evaluate


def main(args):
    # Progress bar
    progress = WorkSplitter()

    # Show hyper parameter settings
    progress.section("Parameter Setting")
    print("Data Path: {}".format(args.path))
    print("Train File Name: {}".format(args.train))
    print("Active File Name: {}".format(args.active))
    print("Test File Name: {}".format(args.test))
    print("Recommendation Model: {}".format(args.rec_model))
    print("Active Learning Algorithm: {}".format(args.active_learning_model))
    if args.item == True:
        mode = "Item-based"
    else:
        mode = "User-based"
    print("Mode: {}".format(mode))
    print("Alpha: {}".format(args.alpha))
    print("Rank: {}".format(args.rank))
    print("Lambda: {}".format(args.lamb))
    print("SVD/Alter Iteration: {}".format(args.iter))
    print("Evaluation Ranking Topk: {}".format(args.topk))
    print("GPU: {}".format(args.gpu))
    print("Latent: {}".format(args.latent))
    print("Iterative: {}".format(args.iterative))
    print("Sample From All: {}".format(args.sample_all))
    print("Number of Active Learning Iteration: {}".format(args.al_iter))
    print("Train Valid Test Split Ratio: {}".format(args.ratio))

    # Load Data
    progress.section("Loading Data")
    start_time = time.time()
    if args.shape is None:
        R_train = load_numpy(path=args.path, name=args.train)
    else:
        # R_train = load_pandas(path=args.path, name=args.train, shape=args.shape)
        R_train = load_csv(path=args.path, name=args.train, shape=args.shape)
    print("Train U-I Dimensions: {}".format(R_train.shape))

    R_active = load_numpy(path=args.path, name=args.active)
    print("Active U-I Dimensions: {}".format(R_active.shape))

    R_test = load_numpy(path=args.path, name=args.test)
    print("Test U-I Dimensions: {}".format(R_test.shape))

    print("Elapsed: {}".format(inhour(time.time() - start_time)))

    for i in range(10):
        metrics_result = active_learning_models[args.active_learning_model](R_train, R_active, R_test, rec_model=args.rec_model, topk=args.topk,
                                                                            train_index=int(R_test.shape[0]*args.ratio[0]),
                                                                            al_iteration=args.al_iter, latent=args.latent,
                                                                            iterative=args.iterative, sample_all=args.sample_all,
                                                                            iteration=args.iter, rank=args.rank,
                                                                            corruption=args.corruption, gpu=args.gpu,
                                                                            lam=args.lamb, alpha=args.alpha,
                                                                            seed=args.seed, root=args.root)

        #import ipdb; ipdb.set_trace()

    #    user_study_df_name = "RecsysPlots/user_study_sampling_" + str(args.sampling) + ".csv"
    #    user_study_df = pd.DataFrame(user_case)
    #    user_study_df.to_csv(user_study_df_name, sep='\t', encoding='utf-8', index=False)


        export_metrics_df_name = "RecsysPlots/" + args.rec_model + ".csv"

        result_df = pd.DataFrame(metrics_result)
        result_df['active_learning_model'] = args.active_learning_model
        result_df['rec_model'] = args.rec_model
        result_df['rank'] = args.rank
        result_df['alpha'] = args.alpha
        result_df['lambda'] = args.lamb
        result_df['iter'] = args.iter
        result_df['corruption'] = args.corruption
        result_df['root'] = args.root
        result_df['alpha'] = args.alpha
        result_df['lambda'] = args.lamb
        result_df['latent'] = args.latent
        result_df['active_learning_iteration'] = args.al_iter
        result_df['iterative'] = args.iterative
        result_df['latent'] = args.latent
        result_df['sample_all'] = args.sample_all

    #    result_df.to_csv(export_metrics_df_name, sep='\t', encoding='utf-8', index=False)

    #    import ipdb; ipdb.set_trace()

        previous_df = pd.read_csv(export_metrics_df_name, sep='\t', encoding='utf-8')
        result_df = pd.concat([previous_df, result_df])
        result_df.to_csv(export_metrics_df_name, sep='\t', encoding='utf-8', index=False)

    """
    result_df.to_pickle(export_metrics_df_name)

    export_metrics_df_name = args.active_learning_model + "_" + \
        args.rec_model + "_Latent_" + str(args.latent) + "_" + \
        str(args.al_iter) + "steps_" + \
        str(args.topk) + "items_per_step_per_user"
    """


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Deep_Preference_Elicitation")

    parser.add_argument('--disable-item-item', dest='item', action='store_false')
    parser.add_argument('--disable-gpu', dest='gpu', action='store_false')
    parser.add_argument('--disable-latent', dest='latent', action='store_false')
    parser.add_argument('--disable-iterative', dest='iterative', action='store_false')
    parser.add_argument('--disable-sample-all', dest='sample_all', action='store_false')
    parser.add_argument('-i', dest='iter', type=check_int_positive, default=1)
    parser.add_argument('-a', dest='alpha', type=check_float_positive, default=100.0)
    parser.add_argument('-l', dest='lamb', type=check_float_positive, default=100)
    parser.add_argument('-r', dest='rank', type=check_int_positive, default=100)
    parser.add_argument('-f', dest='root', type=check_float_positive, default=1)
    parser.add_argument('-c', dest='corruption', type=check_float_positive, default=0.5)
    parser.add_argument('-s', dest='seed', type=check_int_positive, default=1)
    parser.add_argument('-ali', dest='al_iter', type=check_int_positive, default=1)
    parser.add_argument('-m', dest='rec_model', default="IFVAE")
    parser.add_argument('-alm', dest='active_learning_model', default="Random")
    parser.add_argument('-d', dest='path', default="datax/")
    parser.add_argument('-tr', dest='train', default='Rtrain.npz')
    parser.add_argument('-al', dest='active', default='Ractive.npz')
    parser.add_argument('-te', dest='test', default='Rtest.npz')
    parser.add_argument('-ratio', dest='ratio', type=ratio, default='0.5, 0.0, 0.5')
    parser.add_argument('-k', dest='topk', type=check_int_positive, default=50)
    parser.add_argument('--similarity', dest='sim_measure', default='Cosine')
    parser.add_argument('--shape', help="CSR Shape", dest="shape", type=shape, nargs=2)
    args = parser.parse_args()

    main(args)

