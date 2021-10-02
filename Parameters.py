import numpy as np

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

EPSILON = 0.1
GAMMA = 0.90
LEARNING_RATE = 0.1
MAX_STEPS = np.inf
NUMBER_OF_EPISODES = 1000

MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "10x10": []
}

GRID_SIZE = 4