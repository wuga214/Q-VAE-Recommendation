import scipy.sparse as sparse
import numpy as np
from tqdm import tqdm


def split_seed_randomly(rating_matrix, ratio=[0.5, 0.2, 0.3], threshold=70,
                        implicit=True, remove_empty=True, split_seed=8292, sampling=True, percentage=0.1):
    '''
    Split based on a deterministic seed randomly
    '''

    if sampling:
        m, n = rating_matrix.shape
        index = np.random.choice(m, int(m * percentage))
        rating_matrix = rating_matrix[index]


    if implicit:
        '''
        If only implicit (clicks, views, binary) feedback, convert to implicit feedback
        '''
        temp_rating_matrix = sparse.csr_matrix(rating_matrix.shape)
        temp_rating_matrix[(rating_matrix >= threshold).nonzero()] = 1
        rating_matrix = temp_rating_matrix
    if remove_empty:
        # Remove empty columns. record original item index
        nonzero_index = np.unique(rating_matrix.nonzero()[1])
        rating_matrix = rating_matrix[:, nonzero_index]

        # Remove empty rows. record original user index
        nonzero_rows = np.unique(rating_matrix.nonzero()[0])
        rating_matrix = rating_matrix[nonzero_rows]

    # Note: This just gets the highest userId and doesn't account for non-contiguous users.
    user_num, item_num = rating_matrix.shape

    rtrain = []
    rvalid = []
    rtest = []

    # Set the random seed for splitting
    np.random.seed(split_seed)
    for i in tqdm(range(user_num)):
        item_indexes = rating_matrix[i].nonzero()[1]
        rating_data = rating_matrix[i].data
        num_nonzeros = len(item_indexes)
        if num_nonzeros >= 1:
            num_test = int(num_nonzeros * ratio[2])
            num_valid = int(num_nonzeros * (ratio[1] + ratio[2]))

            valid_offset = num_nonzeros - num_valid
            test_offset = num_nonzeros - num_test

            # Randomly shuffle the data
            permuteIndices = np.random.permutation(rating_data.size)
            rating_data = rating_data[permuteIndices]
            item_indexes = item_indexes[permuteIndices]

            # Append data to train, valid, test
            rtrain.append([rating_data[:valid_offset], np.full(valid_offset, i), item_indexes[:valid_offset]])
            rvalid.append([rating_data[valid_offset:test_offset], np.full(test_offset - valid_offset, i),
                           item_indexes[valid_offset:test_offset]])
            rtest.append([rating_data[test_offset:], np.full(num_test, i), item_indexes[test_offset:]])

    # Convert to csr matrix
    rtrain = np.array(rtrain)
    rvalid = np.array(rvalid)
    rtest = np.array(rtest)

    rtrain = sparse.csr_matrix((np.hstack(rtrain[:, 0]), (np.hstack(rtrain[:, 1]), np.hstack(rtrain[:, 2]))),
                               shape=rating_matrix.shape, dtype=np.float32)
    rvalid = sparse.csr_matrix((np.hstack(rvalid[:, 0]), (np.hstack(rvalid[:, 1]), np.hstack(rvalid[:, 2]))),
                               shape=rating_matrix.shape, dtype=np.float32)
    rtest = sparse.csr_matrix((np.hstack(rtest[:, 0]), (np.hstack(rtest[:, 1]), np.hstack(rtest[:, 2]))),
                              shape=rating_matrix.shape, dtype=np.float32)

    return rtrain, rvalid, rtest, nonzero_index


def time_ordered_split(rating_matrix, timestamp_matrix, ratio=[0.5, 0.2, 0.3],
                       implicit=True, remove_empty=True, sampling=False, sampling_ratio=0.1):

    if sampling:
        m, n = rating_matrix.shape
        index = np.random.choice(m, size=m*sampling_ratio, replace=False)
        rating_matrix = rating_matrix[index]

    if implicit:
        # rating_matrix[rating_matrix.nonzero()] = 1

        temp_rating_matrix = sparse.csr_matrix(rating_matrix.shape)
        temp_rating_matrix[(rating_matrix > 3).nonzero()] = 1
        rating_matrix = temp_rating_matrix
        timestamp_matrix = timestamp_matrix.multiply(rating_matrix)

    nonzero_index = None

    if remove_empty:
        # Remove empty columns. record original item index
        nonzero_index = np.unique(rating_matrix.nonzero()[1])
        rating_matrix = rating_matrix[:, nonzero_index]
        timestamp_matrix = timestamp_matrix[:, nonzero_index]

        # Remove empty rows. record original user index
        nonzero_rows = np.unique(rating_matrix.nonzero()[0])
        rating_matrix = rating_matrix[nonzero_rows]
        timestamp_matrix = timestamp_matrix[nonzero_rows]

    user_num, item_num = rating_matrix.shape

    rtrain = []
    rtime = []
    rvalid = []
    rtest = []

    for i in tqdm(range(user_num)):
        item_indexes = rating_matrix[i].nonzero()[1]
        data = rating_matrix[i].data
        timestamp = timestamp_matrix[i].data
        num_nonzeros = len(item_indexes)
        if num_nonzeros >= 1:
            num_test = int(num_nonzeros * ratio[2])
            num_valid = int(num_nonzeros * (ratio[1] + ratio[2]))

            valid_offset = num_nonzeros - num_valid
            test_offset = num_nonzeros - num_test

            argsort = np.argsort(timestamp)
            data = data[argsort]
            item_indexes = item_indexes[argsort]

            rtrain.append([data[:valid_offset], np.full(valid_offset, i), item_indexes[:valid_offset]])
            rvalid.append([data[valid_offset:test_offset], np.full(test_offset - valid_offset, i),
                           item_indexes[valid_offset:test_offset]])
            rtest.append([data[test_offset:], np.full(num_test, i), item_indexes[test_offset:]])

    rtrain = np.array(rtrain)
    rvalid = np.array(rvalid)
    rtest = np.array(rtest)

    rtrain = sparse.csr_matrix((np.hstack(rtrain[:, 0]), (np.hstack(rtrain[:, 1]), np.hstack(rtrain[:, 2]))),
                               shape=rating_matrix.shape, dtype=np.float32)
    rvalid = sparse.csr_matrix((np.hstack(rvalid[:, 0]), (np.hstack(rvalid[:, 1]), np.hstack(rvalid[:, 2]))),
                               shape=rating_matrix.shape, dtype=np.float32)
    rtest = sparse.csr_matrix((np.hstack(rtest[:, 0]), (np.hstack(rtest[:, 1]), np.hstack(rtest[:, 2]))),
                              shape=rating_matrix.shape, dtype=np.float32)


    return rtrain, rvalid, rtest, nonzero_index, timestamp_matrix


def split_user_randomly(rating_matrix, timestamp_matrix, ratio=[0.5, 0.0, 0.5],
                        implicit=True, remove_empty=True, threshold=3,
                        sampling=False, sampling_ratio=0.1, split_seed=8292):

    if sampling:
        m, n = rating_matrix.shape
        index = np.random.choice(m, size=m*sampling_ratio, replace=False)
        rating_matrix = rating_matrix[index]

    if implicit:
        temp_rating_matrix = sparse.lil_matrix(rating_matrix.shape)
        temp_rating_matrix[(rating_matrix > threshold).nonzero()] = 1
        rating_matrix = temp_rating_matrix.tocsr()
        timestamp_matrix = timestamp_matrix.multiply(rating_matrix)

    nonzero_index = None

    if remove_empty:
        # Remove empty columns. record original item index
        nonzero_index = np.unique(rating_matrix.nonzero()[1])
        rating_matrix = rating_matrix[:, nonzero_index]
        timestamp_matrix = timestamp_matrix[:, nonzero_index]

        # Remove empty rows. record original user index
        nonzero_rows = np.unique(rating_matrix.nonzero()[0])
        rating_matrix = rating_matrix[nonzero_rows]
        timestamp_matrix = timestamp_matrix[nonzero_rows]

    user_num, item_num = rating_matrix.shape

    # Set the random seed for splitting
    np.random.seed(split_seed)
    user_index = np.arange(user_num)
    np.random.shuffle(user_index)
    rating_matrix = rating_matrix[user_index, :]
    timestamp_matrix = timestamp_matrix[user_index, :]

    nonzero_row, nonzero_col = rating_matrix.nonzero()
    nonzero_data = rating_matrix.data

    user_train_index = int(user_num * ratio[0])
    train_index = np.where(nonzero_row==user_train_index)[0][0]

    user_valid_index = int(user_num * (ratio[0] + ratio[1]))
    valid_index = np.where(nonzero_row==user_valid_index)[0][0]

    rtrain = sparse.csr_matrix((nonzero_data[:train_index],
                                (nonzero_row[:train_index],
                                 nonzero_col[:train_index])),
                               shape=(user_num, item_num), dtype=np.float32)

    rvalid = sparse.csr_matrix((nonzero_data[train_index:valid_index],
                                (nonzero_row[train_index:valid_index],
                                 nonzero_col[train_index:valid_index])),
                               shape=(user_num, item_num), dtype=np.float32)

    rtest = sparse.csr_matrix((nonzero_data[valid_index:],
                               (nonzero_row[valid_index:],
                                nonzero_col[valid_index:])),
                              shape=(user_num, item_num), dtype=np.float32)

    return rtrain, rvalid, rtest, train_index, valid_index, timestamp_matrix

