import numpy as np
import pandas as pd
from tqdm import tqdm
from evaluation.metrics import evaluate
from utils.progress import WorkSplitter
import inspect
from models.predictor import predict


def hyper_parameter_tuning(train, validation, params, measure='Cosine', gpu_on=True):
    progress = WorkSplitter()
    df = pd.DataFrame(columns=['model', 'rank', 'alpha', 'lambda', 'iter', 'similarity', 'corruption', 'root', 'topK'])

    num_user = train.shape[0]

    for algorithm in params['models']:

        for rank in params['rank']:

            for alpha in params['alpha']:

                for lam in params['lambda']:

                    for corruption in params['corruption']:

                        for root in params['root']:

                            format = "model: {0}, rank: {1}, alpha: {2}, lambda: {3}, corruption: {4}, root: {5}"
                            progress.section(format.format(algorithm, rank, alpha, lam, corruption, root))
                            RQ, Yt, Bias = params['models'][algorithm](train,
                                                                       embeded_matrix=np.empty((0)),
                                                                       iteration=params['iter'],
                                                                       rank=rank,
                                                                       lam=lam,
                                                                       alpha=alpha,
                                                                       corruption=corruption,
                                                                       root=root,
                                                                       gpu_on=gpu_on)
                            Y = Yt.T

                            progress.subsection("Prediction")

                            prediction = predict(matrix_U=RQ, matrix_V=Y, measure=measure, bias=Bias,
                                                 topK=params['topK'][-1], matrix_Train=train, gpu=gpu_on)

                            progress.subsection("Evaluation")

                            result = evaluate(prediction, validation, params['metric'], params['topK'])

                            result_dict = {'model': algorithm, 'rank': rank, 'alpha': alpha, 'lambda': lam,
                                           'iter': params['iter'], 'similarity': params['similarity'],
                                           'corruption': corruption, 'root': root}

                            for name in result.keys():
                                result_dict[name] = [round(result[name][0], 4), round(result[name][1], 4)]

                            df = df.append(result_dict, ignore_index=True)
    return df