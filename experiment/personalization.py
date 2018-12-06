import os
import numpy as np
import pandas as pd
import scipy.stats as ss
from models.predictor import predict
from evaluation.metrics import evaluate
from plots.rec_plots import pandas_ridge_plot


def personalization(Rtrain, Rvalid, df_input, topK, gpu_on=True):
    item_popularity = np.array(np.sum(Rtrain, axis=0)).flatten()

    index = None

    for k in topK:
        medians = []
        giant_dataframes = []

        for idx, row in df_input.iterrows():
            row = row.to_dict()

            RQ = np.load('latent/U_{0}_{1}.npy'.format(row['model'], row['rank']))
            Y = np.load('latent/V_{0}_{1}.npy'.format(row['model'], row['rank']))

            if os.path.isfile('latent/B_{0}_{1}.npy'.format(row['model'], row['rank'])):
                Bias = np.load('latent/B_{0}_{1}.npy'.format(row['model'], row['rank']))
            else:
                Bias = None

            prediction = predict(matrix_U=RQ,
                                 matrix_V=Y,
                                 bias=Bias,
                                 topK=k,
                                 matrix_Train=Rtrain,
                                 measure=row['similarity'],
                                 gpu=gpu_on)

            result = dict()
            top_popularity = item_popularity[prediction.astype(np.int32)]
            result['pop'] = top_popularity[np.array(np.sum(Rvalid, axis=1)).flatten() != 0].flatten()

            df = pd.DataFrame(result)
            df['model'] = row['model']
            giant_dataframes.append(df)
            medians.append(np.median(result['pop']))

        if index is None:
            index = np.argsort(medians).tolist()

        giant_dataframes = [giant_dataframes[i] for i in index]

        df = pd.concat(giant_dataframes)

        pandas_ridge_plot(df, 'model', 'pop', k, folder='analysis/personalization',
                          name="personalization_at_{0}".format(k))

