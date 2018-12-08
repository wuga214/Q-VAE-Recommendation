import numpy as np
import argparse
import json
import pandas as pd
from experiment.personalization import personalization
from utils.io import load_numpy, save_dataframe_latex, save_dataframe_csv, find_best_hyperparameters
from utils.modelnames import models


def main(args):

    df = find_best_hyperparameters('tables/'+args.problem, 'NDCG')

    R_train = load_numpy(path=args.path, name=args.train)
    R_valid = load_numpy(path=args.path, name=args.valid)
    topK = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

    personalization(R_train, R_valid, df, topK, args.problem, args.model_folder, gpu_on=args.gpu)

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Personalization")
    parser.add_argument('-d', dest='path', default="datax/")
    parser.add_argument('-t', dest='train', default='Rtrain.npz')
    parser.add_argument('-v', dest='valid', default='Rtest.npz')
    parser.add_argument('-p', dest='problem', default='movielens1m')
    parser.add_argument('-s', dest='model_folder', default='latent')  # Model saving folder
    parser.add_argument('-gpu', dest='gpu', action='store_true')
    args = parser.parse_args()

    main(args)