import numpy as np

from env.simulator import MovieRecommendEnv
from models.bandit import EpsilonGreedyBandit


class RecommendationService:
    def __init__(self):
        self.env = MovieRecommendEnv(top_n=5)
        n_movies = self.env.loader.movies_df["movie_id"].max()
        self.agent = EpsilonGreedyBandit(n_movies=n_movies, epsilon=0.1)

    def get_recommendation(self, user_id: int, n: int):
        self.env.reset(user_id=user_id)
        movie_ids = self.agent.predict(n_to_recommend=n)
        return list(movie_ids)

    def process_feedback(self, user_id: int, movie_id: int, rating: float):
        reward = 1.0 if rating >= 4 else 0.0
        self.agent.update(movie_id, reward)
        return {"status": "success", "reward_applied": reward}


recommendation_service = RecommendationService()
