import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

import seaborn as sns
sns.axes_style("white")

def show_samples(images, row, col, image_shape, name="Unknown", save=True, shift=False):
    num_images = row*col
    if shift:
        images = (images+1.)/2.
    fig = plt.figure(figsize=(col, row))
    grid = ImageGrid(fig, 111,
                     nrows_ncols=(row, col),
                     axes_pad=0.)
    for i in xrange(num_images):
        im = images[i].reshape(image_shape)
        axis = grid[i]
        axis.axis('off')
        axis.imshow(im)
    plt.axis('off')
    plt.tight_layout()
    if save:
        fig.savefig('figs/train/grid/'+name+'.png', bbox_inches="tight", pad_inches=0, format='png')
    else:
        plt.show()


def latent_distribution_ellipse(means, stds, name="Unknown", save=True, shift=False):
    fig, ax = plt.subplots(figsize=(8, 8))
    patches = []
    m, _ = means.shape
    for i in range(m):
        ellipse = mpatches.Ellipse(means[i], stds[i][0], stds[i][1], facecolor=None, fill=False)
        patches.append(ellipse)

    collection = PatchCollection(patches, cmap=plt.cm.hsv, alpha=0.3)
    #colors = np.linspace(0, 1, len(patches))
    #collection.set_array(np.array(colors))
    ax.add_collection(collection)

    plt.axis('equal')
    #plt.axis('off')
    plt.tight_layout()
    if save:
        fig.savefig('figs/train/grid/'+name+'.png', bbox_inches="tight", pad_inches=0, format='png')
    else:
        plt.show()


