import numpy as np
import pandas as pd
from tqdm import tqdm
from evaluation.metrics import evaluate
from utils.progress import WorkSplitter
import inspect
from models.predictor import predict


def execute(train, test, params, model, measure='Cosine', gpu_on=True):
    progress = WorkSplitter()

    df = pd.DataFrame(columns=['model', 'rank', 'alpha', 'lambda'])

    RQ, Yt, Bias = model(train,
                         embeded_matrix=np.empty((0)),
                         iteration=params['iter'],
                         rank=params['rank'],
                         lam=params['lam'],
                         alpha=params['alpha'],
                         corruption=params['corruption'],
                         gpu_on=True)
    Y = Yt.T

    progress.subsection("Prediction")

    prediction = predict(matrix_U=RQ, matrix_V=Y, measure=measure, bias=Bias,
                         topK=params['topK'][-1], matrix_Train=train, gpu=gpu_on)

    progress.subsection("Evaluation")

    result = evaluate(prediction, test, params['metric'], params['topK'])

    result_dict = {'model': params['model'],
                   'rank': params['rank'],
                   'alpha': params['alpha'],
                   'lambda': params['lam']}

    for name in result.keys():
        result_dict[name] = [round(result[name][0], 4), round(result[name][1], 4)]
    df = df.append(result_dict, ignore_index=True)

    return df