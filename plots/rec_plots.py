import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

import seaborn as sns
sns.axes_style("white")


def show_training_progress(df, hue='model', metric='NDCG', name="epoch_vs_ndcg", save=True):
    fig, ax = plt.subplots(figsize=(8, 4))
    #plt.axhline(y=0.165, color='r', linestyle='-')
    ax = sns.lineplot(x='epoch', y=metric, hue=hue, style=hue, data=df)
    plt.tight_layout()
    if save:
        fig.savefig('figs/train/progress/'+name+'.png', bbox_inches="tight", pad_inches=0, format='png')
    else:
        plt.show()
