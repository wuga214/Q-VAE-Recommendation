import os
from scipy.sparse import save_npz, load_npz
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import yaml
import stat
from os import listdir
from os.path import isfile, join
from ast import literal_eval


def save_dataframe_csv(df, path, name):
    df.to_csv(path+name)


def save_dataframe_latex(df, path, model):
    with open('{0}{1}_parameter_tuning.tex'.format(path, model), 'w') as handle:
        handle.write(df.to_latex(index=False))


def save_numpy(matrix, path, model):
    save_npz('{0}{1}'.format(path, model), matrix)


def save_array(array, path, model):
    np.save('{0}{1}'.format(path, model), array)


def load_numpy(path, name):
    return load_npz(path+name).tocsr()


def load_pandas(path, name, row_name='userId', col_name='movieId',
                value_name='rating', shape=(138494, 131263), sep=','):
    df = pd.read_csv(path + name, sep=sep)
    rows = df[row_name]
    cols = df[col_name]
    values = df[value_name]
    return csr_matrix((values,(rows, cols)), shape=shape)


def load_csv(path, name, shape=(1010000, 2262292)):
    data = np.genfromtxt(path + name, delimiter=',')
    matrix = csr_matrix((data[:, 2], (data[:, 0], data[:, 1])), shape=shape)
    save_npz(path + "rating.npz", matrix)
    return matrix


def load_yahoo(path, name, shape, sep='\t'):
    '''
    Load yahoo dataset from WebScope. Only tested on R1 so far
    '''
    df = pd.read_csv(path + name, sep=sep, header=None, names=['userId', 'trackId', 'rating'])

    # Coordinate Format
    userCOO = list()
    itemCOO = list()
    ratingCOO = list()

    # Create unique mapping
    itemIdToUniqueItemId = dict()
    userIdToUniqueUserId = dict()
    uniqueItemId = 0
    uniqueUserId = 0
    numIndex = 0
    for index, row in df.iterrows():
        numIndex += 1
        itemId = row['trackId']
        userId = row['userId']
        rating = row['rating']
        # Ensure it is in map
        if itemId not in itemIdToUniqueItemId.keys():
            itemIdToUniqueItemId[itemId] = uniqueItemId
            uniqueItemId += 1
        if userId not in userIdToUniqueUserId.keys():
            userIdToUniqueUserId[userId] = uniqueUserId
            uniqueUserId += 1
        userCOO.append(userIdToUniqueUserId[userId])
        itemCOO.append(itemIdToUniqueItemId[itemId])
        ratingCOO.append(rating)
    rows = np.array(userCOO)
    cols = np.array(itemCOO)
    values = np.array(ratingCOO)
    print("Number of UniqueItems", uniqueItemId)
    print("Number of UniqueUsers", uniqueUserId)
    print("Number of ratings", numIndex)
    #rows = df['userId']
    #cols = df['trackId']
    #values = df['rating']
    csrMat = csr_matrix((values,(rows, cols)), shape=shape)
    return csrMat


def load_netflix(path, shape=(2649430, 17771)):
    # Cautious: This function will reindex the user-item IDs, only for experiment usage
    frames = []
    print("Load Files")
    for file in tqdm(os.listdir(path)):
        if file.endswith(".txt"):
            movie_path = os.path.join(path, file)
            with open(movie_path) as f:
                movie_index = f.readline().split(':')[0]
            df = pd.read_csv(movie_path, skiprows=1, header=None, names=['userID', 'rating', 'timestamp'])
            df['movieId'] = int(movie_index)
            frames.append(df)

    df = pd.concat(frames)
    ratings = df['rating']
    rows = df['userID']
    cols = df['movieId']
    timestamps = df['timestamp']
    tqdm.pandas()
    print("Transform timestamps")
    timestamps = timestamps.str.replace('-', '').progress_apply(int)
    print("Create Sparse Matrices")
    return csr_matrix((ratings, (rows, cols)), shape=shape), csr_matrix((timestamps, (rows, cols)), shape=shape)


def save_pickle(path, name, data):
    with open('{0}/{1}.pickle'.format(path, name), 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path, name):
    with open('{0}/{1}.pickle'.format(path, name), 'rb') as handle:
        data = pickle.load(handle)

    return data


def load_yaml(path, key='parameters'):
    with open(path, 'r') as stream:
        try:
            return yaml.load(stream)[key]
        except yaml.YAMLError as exc:
            print(exc)


def find_best_hyperparameters(folder_path, meatric):
    csv_files = [join(folder_path, f) for f in listdir(folder_path)
                 if isfile(join(folder_path, f)) and f.endswith('.csv')]
    best_settings = []
    for record in csv_files:
        df = pd.read_csv(record)
        df[meatric+'_Score'] = df[meatric].map(lambda x: literal_eval(x)[0])
        best_settings.append(df.loc[df[meatric+'_Score'].idxmax()].to_dict())

    df = pd.DataFrame(best_settings).drop(meatric+'_Score', axis=1)

    return df


def get_file_names(folder_path, extension='.yml'):
    return [f for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.endswith(extension)]


def write_file(folder_path, file_name, content, exe=False):
    full_path = folder_path+'/'+file_name
    with open(full_path, 'w') as the_file:
        the_file.write(content)

    if exe:
        st = os.stat(full_path)
        os.chmod(full_path, st.st_mode | stat.S_IEXEC)