import numpy as np


class EpsilonGreedyBandit:
    def __init__(self, n_movies, epsilon=0.1):
        self.epsilon = epsilon
        self.n_movies = n_movies
        self.counts = np.zeros(n_movies + 1)  # 1 based indexing
        self.values = np.zeros(n_movies + 1)
        # to track total reward and counts for each movie

    def predict(self, n_to_recommend=5):
        if np.random.rand() < self.epsilon:
            return np.random.choice(
                range(1, self.n_movies + 1), n_to_recommend, replace=False
            )
            # explore case : randomly sample movie ids,in future sample from valid movie_id
        else:
            return np.argsort(self.values)[-n_to_recommend:][::-1]
            # exploit case : pick movies with the highest estimate value

    def update(self, movie_id, reward):
        """Updates the estimated value of a movie based on feedback"""
        self.counts[movie_id] += 1
        n = self.counts[movie_id]
        value = self.values[movie_id]

        self.values[movie_id] = value + (1 / n) * (reward - value)


# What this should contain :
#   1. Knowledge base - way to track the average reward for every movie
#   2. Predict - exploration and exploitation
#   3. Update - Update the average score of the movie based on the reward
