import numpy as np
import itertools
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


def pandas_ridge_plot(df, model, pop, k, folder='figures', name='personalization', save=True):
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    num_models = len(df.model.unique())


    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(num_models, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row=model, hue=model, aspect=15, height=.5, palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, pop, clip_on=False, shade=True, alpha=1, lw=1.5, bw=.1)
    g.map(sns.kdeplot, pop, clip_on=False, color="w", lw=2, bw=.1)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(-0.1, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)

    g.map(label, pop)

    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap

    g.set_xlabels("Popularity Distribution of The Top-{0} Recommended Items".format(k))
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    if save:
        plt.savefig("figs/{0}/{1}.pdf".format(folder, name), format="pdf")
        plt.savefig("figs/{0}/{1}.png".format(folder, name), format="png")
    else:
        plt.show()
    plt.close()


def pandas_bar_plot(df, x, y, hue, x_name, y_name, folder='figures', name='unknown', save=True):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(ax=ax, x=x, y=y, hue=hue, data=df, errwidth=1)

    num_category = len(df[x].unique())
    hatch = None
    hatches = itertools.cycle(['//', '+++', '///', '---', 'xxx', '\\\\\\', '+/+/', '+\\+\\', '...', '+-+-'])
    for i, bar in enumerate(ax.patches):
        if i % num_category == 0:
            hatch = next(hatches)
        bar.set_hatch(hatch)

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.xticks(rotation=15)
    plt.legend(loc='upper left')
    if 'Precision' not in y:
        ax.legend_.remove()
    plt.tight_layout()
    if save:
        plt.savefig("figs/{0}/{1}_bar.pdf".format(folder, name), format="pdf")
        plt.savefig("figs/{0}/{1}_bar.png".format(folder, name), format="png")
    else:
        plt.show()
    plt.close()
