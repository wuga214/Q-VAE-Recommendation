from active_learning_models.random import random
from active_learning_models.entropy import entropy
from active_learning_models.ucb1 import ucb1
from active_learning_models.expected_best import expected_best
from active_learning_models.thompson_sampling import thompson_sampling

active_learning_models = {
    "Random": random,
    "Entropy": entropy,
    "UCB1": ucb1,
    "ExpectedBest": expected_best,
    "ThompsonSampling": thompson_sampling,
}

