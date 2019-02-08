from utils.io import save_numpy, load_pandas, save_array
from utils.argcheck import shape, ratio
from providers.split import split_user_randomly, time_ordered_split
import argparse
from utils.progress import WorkSplitter


def main(args):
    progress = WorkSplitter()

    progress.section("Load Raw Data")
    rating_matrix = load_pandas(path=args.path, name=args.name, shape=args.shape)
    timestamp_matrix = load_pandas(path=args.path, value_name='timestamp', name=args.name, shape=args.shape)
    progress.section("Split CSR Matrices")
    rtrain, rvalid, rtest, train_index, valid_index, \
        rtime = split_user_randomly(rating_matrix=rating_matrix,
                                    timestamp_matrix=timestamp_matrix,
                                    ratio=args.ratio,
                                    implicit=args.implicit)

    rtrain, rvalid, _, nonzero_index, rtime = time_ordered_split(rating_matrix=rtrain,
                                                                 timestamp_matrix=rtime,
                                                                 ratio=[0.5, 0.5, 0.0],
                                                                 implicit=False,
                                                                 remove_empty=False)


    progress.section("Save NPZ")
    save_numpy(rtrain, args.path, "Rtrain")
    save_numpy(rvalid, args.path, "Rvalid")
    save_numpy(rtest, args.path, "Rtest")
    save_numpy(rtime, args.path, "Rtime")
    save_array(nonzero_index, args.path, "Index")

if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="Data Preprocess")
    parser.add_argument('--implicit', dest='implicit', action='store_true')
    parser.add_argument('-r', dest='ratio', type=ratio, default='0.5, 0.0, 0.5')
    parser.add_argument('-d', dest='path', default="data/")
    parser.add_argument('-n', dest='name', default='movielens/ml-20m/ratings.csv')
    parser.add_argument('--shape', help="CSR Shape", dest="shape", type=shape, nargs=2)
    args = parser.parse_args()

    main(args)
