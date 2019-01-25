# TODO: Import dataset and split dataset into two as pre-train and test in
# implicit feedback (1, 2, 3, 4 -> 0, 5->1)

# TODO: Train (Question: the sampling method of the value of z from pre-train
# dataset should also be consistent with the following steps, right?) the
# network using the pre-train dataset and extract encode, latent mu,
# latent sigma, decode and prediction part

# TODO: Pick Top 1 and 2 for each user from prediction. Convert those two picks
# to (1, 2 -> some hyperparameter such as -0.1 & 3, 4 -> 0 & 5 -> 1) and
# feedforward through encoder to get latent mu and sigma. Use that particular
# sampling method to predict. And continue


