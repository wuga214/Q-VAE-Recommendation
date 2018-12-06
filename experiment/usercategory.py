import numpy as np


def personalization(Rtrain, Rvalid, df):
    item_popularity = np.array(np.sum(Rtrain, axis=0)).flatten()

    for idx, row in df.iterrows():
        row = row.to_dict()