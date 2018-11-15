from layers.coders import fc_mnist_encoder, fc_mnist_decoder
import tensorflow as tf
import numpy as np
from plots.grid_plots import show_samples, latent_distribution_ellipse
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm
from models.ifvae import IFVAE
import random

def main():
    flags = tf.flags
    flags.DEFINE_integer("latent_dim", 8, "Dimension of latent space.")
    flags.DEFINE_integer("batch_size", 128, "Batch size.")
    flags.DEFINE_integer("epochs", 200, "As it said")
    flags.DEFINE_float("dropout", 0.9, "Dropout rate")
    flags.DEFINE_integer("updates_per_epoch", 100, "Really just can set to 1 if you don't like mini-batch.")
    flags.DEFINE_string("data_dir", 'mnist', "Tensorflow demo data download position.")
    FLAGS = flags.FLAGS

    kwargs = {
        'latent_dim': FLAGS.latent_dim,
        'batch_size': FLAGS.batch_size,
        'encoder': fc_mnist_encoder,
        'decoder': fc_mnist_decoder
    }
    vae = IFVAE(**kwargs)
    mnist = input_data.read_data_sets(train_dir=FLAGS.data_dir)
    tbar = tqdm(range(FLAGS.epochs))
    for epoch in tbar:
        training_loss = 0.

        for _ in range(FLAGS.updates_per_epoch):
            x, _ = mnist.train.next_batch(FLAGS.batch_size)
            # Random Corruption
            corrupt_rate = random.uniform(0.1, 0.5)
            loss = vae.update(x, corruption=corrupt_rate)
            training_loss += loss

        training_loss /= FLAGS.updates_per_epoch
        s = "Loss: {:.4f}".format(training_loss)
        tbar.set_description(s)

    # # Original
    x, _ = mnist.train.next_batch(FLAGS.batch_size)
    show_samples(x, 10, 10, [28, 28], name='original')

    # Random Dropout
    corrupted_x = x*np.random.binomial([np.ones((FLAGS.batch_size, 784))], 1-FLAGS.dropout)[0]
    show_samples(corrupted_x, 10, 10, [28, 28], name='corrupted')
    # Rescale
    corrupted_x = corrupted_x * (1.0/(1-FLAGS.dropout))

    # Reconstructed
    samples = vae.inference(corrupted_x)
    show_samples(samples, 10, 10, [28, 28], name='predicted')
    # vae.save_generator('weights/vae_mnist/generator')

    # Recurrent Corruption Evaluation

    import ipdb; ipdb.set_trace()
    latent(mnist, vae, lim=4)

def latent(mnist, vae, lim=6):
    corruptions = []
    rescales = []
    corruption = np.ones((1, 784))
    for i in range(9):
        corruption = corruption * np.random.binomial([np.ones((1, 784))], 0.75)[0]
        corruptions.append(corruption)
        rescales.append(1.0/(0.75**(i+1)))
    x, _ = mnist.train.next_batch(1)
    x = np.tile(x, (9, 1))

    corrupted_x = x * np.array(corruptions).reshape(9, 784)
    show_samples(corrupted_x, 3, 3, [28, 28], name='recurrent_corrupt')
    corrupted_x = corrupted_x * np.array(rescales).reshape(-1, 1)
    uncertainty = vae.uncertainty(corrupted_x)
    np.savetxt('mean.csv', uncertainty[0], delimiter=',')
    np.savetxt('std.csv', uncertainty[1], delimiter=',')
    latent_distribution_ellipse(uncertainty[0][:, 0:2], 6*uncertainty[1][:, 0:2], 0.74,
                                lim=lim, name="recurrent_latent")
    samples = vae.inference(corrupted_x)
    show_samples(samples, 3, 3, [28, 28], name='recurrent_predict')


if __name__ == '__main__':
    main()