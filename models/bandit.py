import numpy as np


class EpsilonGreedyBandit:
    def __init__(self, n_movies, epsilon=0.1):
        self.epsilon = epsilon
        self.n_movies = n_movies
        self.count = np.zeros(n_movies + 1)


# What this should contain :
#   1. Knowledge base - way to track the average reward for every movie
#   2. Predict - exploration and exploitation
#   3. Update - Update the average score of the movie based on the reward
