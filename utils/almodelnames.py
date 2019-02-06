from al_models.random import random
from al_models.entropy import entropy
from al_models.ucb1 import ucb1
from al_models.best_item import best_item

al_models = {
    "Random": random,
    "Entropy": entropy,
    "UCB1": ucb1,
    "BestItem": best_item,
}

