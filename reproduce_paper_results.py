import numpy as np
import argparse
import json
import pandas as pd
from experiment.execute import execute
from utils.io import load_numpy, save_dataframe_latex, save_dataframe_csv, find_best_hyperparameters
from utils.modelnames import models


""" Example Params.csv File

model,alpha,corruption,rank,iter,lam
VAE-CF,1,0.2,100,300,0.0001
AutoRec,1,0.2,50,300,0.000001
CDAE,1,0.2,50,300,0.000001
IFVAE,1,0.2,50,300,0.0001

"""


def main(args):

    df = find_best_hyperparameters('tables/'+args.problem, 'NDCG')

    R_train = load_numpy(path=args.path, name=args.train)
    R_valid = load_numpy(path=args.path, name=args.valid)

    frame = []
    for idx, row in df.iterrows():
        row = row.to_dict()
        row['metric'] = ['R-Precision', 'NDCG', 'Precision', 'Recall']
        row['topK'] = [5, 10, 15, 20, 50]
        result = execute(R_train, R_valid, row, models[row['model']],
                         measure=row['similarity'], gpu_on=args.gpu, folder=args.model_folder)
        frame.append(result)

    results = pd.concat(frame)
    save_dataframe_csv(results, 'tables/', args.name)

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Reproduce")
    parser.add_argument('-n', dest='name', default="final_result.csv")
    parser.add_argument('-d', dest='path', default="datax/")
    parser.add_argument('-t', dest='train', default='Rtrain.npz')
    parser.add_argument('-v', dest='valid', default='Rtest.npz')
    parser.add_argument('-p', dest='problem', default='movielens1m')
    parser.add_argument('-s', dest='model_folder', default='latent') # Model saving folder
    parser.add_argument('-gpu', dest='gpu', action='store_true')
    args = parser.parse_args()

    main(args)