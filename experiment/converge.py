# Import model class..

from models.cdae import CDAE
from models.vae import VAE
from models.ifvae import IFVAE
from models.autorec import AutoRec
from models.predictor import predict
from evaluation.metrics import evaluate

models = {
    "AutoRec": AutoRec,
    "CDAE": CDAE,
    "VAE-CF": VAE,
    "IFVAE": IFVAE
}


def converge(Rtrain, Rtest, df, gpu_on=True):
    m, n = Rtrain.shape

    for idx, row in df.iterrows():
        row = row.to_dict()
        row['metric'] = ['NDCG', 'R-Precision']
        row['topK'] = [50]

        model = models[row['model']](n, row['rank'], batch_size=100, lamb=row['lam'])
        model.train_model(Rtrain, corruption=row['corruption'], iteration=1)
        for i in range(row['iter']):
            RQ = model.get_RQ(Rtrain)
            Y = model.get_Y()
            Bias = model.get_Bias()

            prediction = predict(matrix_U=RQ,
                                 matrix_V=Y,
                                 bias=Bias,
                                 topK=row['topk'][0],
                                 matrix_Train=Rtrain,
                                 measure='Cosine',
                                 gpu=gpu_on)
            result = evaluate(prediction, Rtest, row['metric'], row['topK'])
            # Note Finished yet
