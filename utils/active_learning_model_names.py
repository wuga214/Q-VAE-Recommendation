from active_learning_models.random import random
from active_learning_models.entropy import entropy
from active_learning_models.non_lin_ucb import non_lin_ucb
from active_learning_models.expected_best import expected_best
from active_learning_models.thompson_sampling import thompson_sampling
from active_learning_models.ucb import ucb

active_learning_models = {
    "Random": random,
    "Entropy": entropy,
    "NonLinUCB": non_lin_ucb,
    "ExpectedBest": expected_best,
    "ThompsonSampling": thompson_sampling,
}

