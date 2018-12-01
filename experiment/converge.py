import json
import pandas as pd
import tensorflow as tf
from models.cdae import CDAE
from models.vae import VAE
from models.ifvae import IFVAE
from models.autorec import AutoRec
from models.predictor import predict
from evaluation.metrics import evaluate
from utils.progress import WorkSplitter
from utils.regularizers import Regularizer

models = {
    "AutoRec": AutoRec,
    "CDAE": CDAE,
    "VAE-CF": VAE,
    "IFVAE": IFVAE
}


def converge(Rtrain, Rtest, df, epochs=10, gpu_on=True):
    progress = WorkSplitter()
    m, n = Rtrain.shape

    results = pd.DataFrame(columns=['model', 'rank', 'lambda', 'epoch', 'optimizer'])

    for run in range(3):

        for idx, row in df.iterrows():
            row = row.to_dict()

            progress.section(json.dumps(row))

            row['metric'] = ['NDCG', 'R-Precision']
            row['topK'] = [50]
            try:
                model = models[row['model']](n, row['rank'],
                                             batch_size=100,
                                             lamb=row['lam'],
                                             optimizer=Regularizer[row['optimizer']])
            except:
                model = models[row['model']](m, n, row['rank'],
                                             batch_size=100,
                                             lamb=row['lam'],
                                             optimizer=Regularizer[row['optimizer']])

            for i in range(epochs):

                model.train_model(Rtrain, corruption=row['corruption'], epoch=1)

                if (i + 1) % 10 == 0:

                    RQ = model.get_RQ(Rtrain)
                    Y = model.get_Y()
                    Bias = model.get_Bias()

                    Y = Y.T

                    prediction = predict(matrix_U=RQ,
                                         matrix_V=Y,
                                         bias=Bias,
                                         topK=row['topK'][0],
                                         matrix_Train=Rtrain,
                                         measure='Cosine',
                                         gpu=gpu_on)

                    result = evaluate(prediction, Rtest, row['metric'], row['topK'])
                    # Note Finished yet
                    result_dict = {'model': row['model'],
                                   'rank': row['rank'],
                                   'lambda': row['lam'],
                                   'optimizer': row['optimizer'],
                                   'epoch': i}

                    for name in result.keys():
                        result_dict[name] = round(result[name][0], 4)
                    results = results.append(result_dict, ignore_index=True)

            model.sess.close()
            tf.reset_default_graph()

    return results
