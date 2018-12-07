import argparse
from experiment.usercategory import usercategory
from utils.io import load_numpy, find_best_hyperparameters


def main(args):

    df = find_best_hyperparameters(args.param, 'NDCG')

    R_train = load_numpy(path=args.path, name=args.train)
    R_valid = load_numpy(path=args.path, name=args.valid)
    topK = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    metric = ['R-Precision', 'NDCG', 'Clicks', 'Recall', 'Precision']

    usercategory(R_train, R_valid, df, topK, metric, gpu_on=args.gpu)

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Personalization")
    parser.add_argument('-n', dest='name', default="final_result.csv")
    parser.add_argument('-d', dest='path', default="datax/")
    parser.add_argument('-t', dest='train', default='Rtrain.npz')
    parser.add_argument('-v', dest='valid', default='Rtest.npz')
    parser.add_argument('-p', dest='param', default='tables/movielens1m')
    parser.add_argument('-gpu', dest='gpu', action='store_true')
    args = parser.parse_args()

    main(args)