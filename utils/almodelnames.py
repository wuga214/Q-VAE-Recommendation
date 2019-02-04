from al_models.random import random
from al_models.entropy import entropy
from al_models.ucb1 import ucb1

al_models = {
    "Random": random,
    "Entropy": entropy,
    "UCB1": ucb1
}

