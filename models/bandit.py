import os
import pickle
from typing import List, Optional, Union

import numpy as np


class EpsilonGreedyBandit:
    def __init__(self, n_movies: int, epsilon: float = 0.1):
        self.epsilon = epsilon
        self.n_movies = n_movies
        self.counts = np.zeros(n_movies + 1, dtype=np.int64)  # 1-based indexing
        self.values = np.zeros(n_movies + 1, dtype=np.float64)

    def predict(self, n_to_recommend: int = 5) -> np.ndarray:
        n_to_recommend = min(n_to_recommend, self.n_movies)
        if np.random.rand() < self.epsilon:
            # Explore case: sample from valid 1-based movie IDs
            return np.random.choice(
                np.arange(1, self.n_movies + 1), n_to_recommend, replace=False
            )
        else:
            # Exploit case: sort only valid movies 1..n_movies (avoid index 0)
            valid_values = self.values[1:]
            top_relative = np.argsort(valid_values)[-n_to_recommend:][::-1]
            return top_relative + 1

    def update(self, movie_id: int, reward: float):
        """Updates the estimated value of a movie based on feedback."""
        if movie_id < 1 or movie_id > self.n_movies:
            return
        self.counts[movie_id] += 1
        n = self.counts[movie_id]
        value = self.values[movie_id]
        self.values[movie_id] = value + (1.0 / n) * (reward - value)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "EpsilonGreedyBandit":
        with open(filepath, "rb") as f:
            return pickle.load(f)


class UCBBandit:
    """Upper Confidence Bound (UCB1) Multi-Armed Bandit."""
    def __init__(self, n_movies: int, c: float = 1.0):
        self.n_movies = n_movies
        self.c = c
        self.counts = np.zeros(n_movies + 1, dtype=np.int64)
        self.values = np.zeros(n_movies + 1, dtype=np.float64)
        self.total_pulls = 0

    def predict(self, n_to_recommend: int = 5) -> np.ndarray:
        n_to_recommend = min(n_to_recommend, self.n_movies)
        
        # If any movie hasn't been pulled yet, prioritize unpulled movies
        unvisited = np.where(self.counts[1:] == 0)[0] + 1
        if len(unvisited) >= n_to_recommend:
            return np.random.choice(unvisited, n_to_recommend, replace=False)

        # UCB1 score: Q(a) + c * sqrt(ln(t) / N(a))
        t = max(self.total_pulls, 1)
        valid_counts = np.maximum(self.counts[1:], 1)
        exploration = self.c * np.sqrt(np.log(t + 1) / valid_counts)
        ucb_scores = self.values[1:] + exploration

        # If there are still some unvisited, set their score to inf
        if len(unvisited) > 0:
            ucb_scores[unvisited - 1] = 1e9

        top_indices = np.argsort(ucb_scores)[-n_to_recommend:][::-1] + 1
        return top_indices

    def update(self, movie_id: int, reward: float):
        if movie_id < 1 or movie_id > self.n_movies:
            return
        self.counts[movie_id] += 1
        self.total_pulls += 1
        n = self.counts[movie_id]
        value = self.values[movie_id]
        self.values[movie_id] = value + (1.0 / n) * (reward - value)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "UCBBandit":
        with open(filepath, "rb") as f:
            return pickle.load(f)


class LinUCBBandit:
    """
    Contextual Linear Upper Confidence Bound Bandit (LinUCB with disjoint linear models).
    Leverages user context/state vector x (e.g. 42-dim demographic + genre affinity).
    """
    def __init__(self, n_movies: int, state_dim: int = 42, alpha: float = 0.5):
        self.n_movies = n_movies
        self.state_dim = state_dim
        self.alpha = alpha

        # A_a = d x d ridge regression matrix for each movie (initialized to identity)
        # b_a = d x 1 reward vector for each movie (initialized to 0)
        self.A = np.zeros((n_movies + 1, state_dim, state_dim), dtype=np.float32)
        for i in range(1, n_movies + 1):
            self.A[i] = np.eye(state_dim, dtype=np.float32)
        self.b = np.zeros((n_movies + 1, state_dim), dtype=np.float32)
        self.counts = np.zeros(n_movies + 1, dtype=np.int64)

    def predict(self, state: np.ndarray, n_to_recommend: int = 5) -> np.ndarray:
        n_to_recommend = min(n_to_recommend, self.n_movies)
        x = state.reshape(-1).astype(np.float32)

        p = np.zeros(self.n_movies + 1, dtype=np.float32)
        p[0] = -np.inf  # Never recommend index 0

        # Vectorized/loop LinUCB calculation
        for a in range(1, self.n_movies + 1):
            A_inv = np.linalg.pinv(self.A[a])
            theta_a = A_inv @ self.b[a]
            mean_est = np.dot(theta_a, x)
            var_est = self.alpha * np.sqrt(np.dot(x, A_inv @ x))
            p[a] = mean_est + var_est

        top_indices = np.argsort(p[1:])[-n_to_recommend:][::-1] + 1
        return top_indices

    def update(self, movie_id: int, reward: float, state: np.ndarray):
        if movie_id < 1 or movie_id > self.n_movies:
            return
        x = state.reshape(-1).astype(np.float32)
        self.A[movie_id] += np.outer(x, x)
        self.b[movie_id] += reward * x
        self.counts[movie_id] += 1

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filepath: str) -> "LinUCBBandit":
        with open(filepath, "rb") as f:
            return pickle.load(f)
