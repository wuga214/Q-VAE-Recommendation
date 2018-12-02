import numpy as np
import argparse
from models.cdae import cdae
from models.vae import vae_cf
from models.ifvae import ifvae
from models.autorec import autorec
from experiment.tuning import hyper_parameter_tuning
from utils.io import load_numpy, save_dataframe_latex, save_dataframe_csv, load_yaml


models = {
    "AutoRec": autorec,
    "CDAE": cdae,
    "VAE-CF": vae_cf,
    "IFVAE": ifvae
}


def main(args):
    params = load_yaml(args.grid)
    params['models'] = models[params['models']]

    R_train = load_numpy(path=args.path, name=args.train)
    R_valid = load_numpy(path=args.path, name=args.valid)
    df = hyper_parameter_tuning(R_train, R_valid, params, measure="Cosine", gpu_on=args.gpu)
    save_dataframe_latex(df, 'tables/', args.name)
    save_dataframe_csv(df, 'tables/', args.name)

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="LRec")
    parser.add_argument('-n', dest='name', default="autorecs_tuning")
    parser.add_argument('-d', dest='path', default="datax/")
    parser.add_argument('-t', dest='train', default='Rtrain.npz')
    parser.add_argument('-v', dest='valid', default='Rvalid.npz')
    parser.add_argument('-y', dest='grid', default='yaml/default.yml')
    parser.add_argument('-gpu', dest='gpu', action='store_true')
    args = parser.parse_args()

    main(args)