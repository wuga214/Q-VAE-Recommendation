import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
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