import numpy as np
import argparse
import json
import pandas as pd
from models.cdae import cdae
from models.vae import vae_cf
from models.ifvae import ifvae
from models.autorec import autorec
from experiment.converge import converge
from utils.io import load_numpy, save_dataframe_latex, save_dataframe_csv
from utils.progress import WorkSplitter
from plots.rec_plots import show_training_progress


""" Example Params.csv File

model,alpha,corruption,rank,iter,lam
VAE-CF,1,0.2,100,300,0.0001
AutoRec,1,0.2,50,300,0.000001
CDAE,1,0.2,50,300,0.000001
IFVAE,1,0.2,50,300,0.0001

"""


models = {
    "AutoRec": autorec,
    "CDAE": cdae,
    "VAE-CF": vae_cf,
    "IFVAE": ifvae
}

def main(args):
    progress = WorkSplitter()

    df = pd.read_csv(args.path + args.param)

    R_train = load_numpy(path=args.path, name=args.train)
    R_valid = load_numpy(path=args.path, name=args.valid)

    results = converge(R_train, R_valid, df, epochs=300, gpu_on=args.gpu)

    save_dataframe_latex(results, 'tables/', args.name)
    save_dataframe_csv(results, 'tables/', args.name)

    show_training_progress(results, metric='NDCG', name="epoch_vs_ndcg")

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="LRec")
    parser.add_argument('-n', dest='name', default="convergence_analysis.csv")
    parser.add_argument('-d', dest='path', default="datax/")
    parser.add_argument('-t', dest='train', default='Rtrain.npz')
    parser.add_argument('-v', dest='valid', default='Rtest.npz')
    parser.add_argument('-p', dest='param', default='Params.csv')
    parser.add_argument('-gpu', dest='gpu', action='store_true')
    args = parser.parse_args()

    main(args)