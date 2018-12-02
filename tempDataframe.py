import pandas as pd
from os import listdir
from os.path import isfile, join
from ast import literal_eval

def find_best_hyperparameters(folder_path, meatric):
    csv_files = [join(folder_path, f) for f in listdir(folder_path) if isfile(join(folder_path, f)) and f.endswith('.csv')]
    best_settings = []
    for record in csv_files:
        df = pd.read_csv(record)
        df[meatric+'_Score'] = df[meatric].map(lambda x: literal_eval(x)[0])
        best_settings.append(df.loc[df[meatric+'_Score'].idxmax()].to_dict())

    df = pd.DataFrame(best_settings)



find_best_hyperparameters("tables/movielens1m", "NDCG")