import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from core.config import (
    BANDIT_WEIGHTS_PATH,
    DEFAULT_DIVERSITY_WEIGHT,
    DEFAULT_TOP_N,
    DQN_WEIGHTS_PATH,
    NUM_GENRES,
    REWARD_MAPPING,
    STATE_DIM,
)
from env.simulator import MovieRecommendEnv
from models.bandit import EpsilonGreedyBandit, LinUCBBandit, UCBBandit
from models.dqn import DQNAgent


class RecommendationService:
    _instance: Optional["RecommendationService"] = None

    def __init__(self):
        self.env = MovieRecommendEnv(top_n=DEFAULT_TOP_N)
        self.n_movies = int(self.env.loader.movies_df["movie_id"].max())

        # Build genre matrix for diversity reranking
        genre_df = self.env.loader.get_movie_genres()
        self.genre_matrix = np.zeros((self.n_movies, NUM_GENRES), dtype=np.float32)
        for mid, row in genre_df.iterrows():
            if mid <= self.n_movies:
                self.genre_matrix[mid - 1] = row.values

        # Initialize DQN Agent
        self.dqn_agent = DQNAgent(
            state_dim=STATE_DIM,
            action_dim=self.n_movies,
            epsilon=0.0,  # Pure inference by default
            genre_matrix=self.genre_matrix,
        )
        self.dqn_weights_loaded = False
        if os.path.exists(DQN_WEIGHTS_PATH):
            try:
                self.dqn_agent.load(str(DQN_WEIGHTS_PATH))
                self.dqn_weights_loaded = True
                print(f"✅ Successfully loaded DQN weights from {DQN_WEIGHTS_PATH}")
            except Exception as e:
                print(f"⚠️ Could not load DQN weights: {e}")
        else:
            print(f"ℹ️ Pretrained DQN weights not found at {DQN_WEIGHTS_PATH}. Running with initialized weights.")

        # Initialize Bandits
        self.bandit_agent = EpsilonGreedyBandit(n_movies=self.n_movies, epsilon=0.05)
        self.bandit_weights_loaded = False
        if os.path.exists(BANDIT_WEIGHTS_PATH):
            try:
                self.bandit_agent = EpsilonGreedyBandit.load(str(BANDIT_WEIGHTS_PATH))
                self.bandit_weights_loaded = True
                print(f"✅ Successfully loaded Bandit weights from {BANDIT_WEIGHTS_PATH}")
            except Exception as e:
                print(f"⚠️ Could not load Bandit weights: {e}")

        self.linucb_agent = LinUCBBandit(n_movies=self.n_movies, state_dim=STATE_DIM, alpha=0.5)

    @classmethod
    def get_instance(cls) -> "RecommendationService":
        if cls._instance is None:
            cls._instance = RecommendationService()
        return cls._instance

    def get_recommendation(
        self,
        user_id: int,
        n: int = DEFAULT_TOP_N,
        model: str = "dqn",
        diversity_weight: float = DEFAULT_DIVERSITY_WEIGHT,
    ) -> Tuple[str, List[int], List[Dict[str, Any]]]:
        model = (model or "dqn").lower()
        state = self.env.reset(user_id=user_id)

        if model == "dqn":
            movie_ids = self.dqn_agent.predict(
                state,
                n_to_recommend=n,
                diversity_weight=diversity_weight,
                deterministic=True,
            )
            model_used = "dqn"
        elif model in ("bandit", "epsilon_greedy"):
            movie_ids = self.bandit_agent.predict(n_to_recommend=n)
            model_used = "bandit"
        elif model == "linucb":
            movie_ids = self.linucb_agent.predict(state, n_to_recommend=n)
            model_used = "linucb"
        elif model == "popular":
            movie_ids = np.array(self.env.loader.get_popular_movies(limit=n))
            model_used = "popular"
        elif model == "random":
            movie_ids = np.random.choice(
                np.arange(1, self.n_movies + 1), min(n, self.n_movies), replace=False
            )
            model_used = "random"
        else:
            # Fallback to DQN
            movie_ids = self.dqn_agent.predict(
                state,
                n_to_recommend=n,
                diversity_weight=diversity_weight,
                deterministic=True,
            )
            model_used = f"dqn (fallback from '{model}')"

        clean_ids = [int(mid) for mid in movie_ids]
        items = []
        for mid in clean_ids:
            info = self.env.loader.get_movie_info(mid)
            if info:
                items.append(info)
            else:
                items.append({"movie_id": mid, "title": f"Movie {mid}", "genres": []})

        return model_used, clean_ids, items

    def process_feedback(
        self, user_id: int, movie_id: int, rating: float
    ) -> Dict[str, Any]:
        int_rating = int(round(rating))
        reward = REWARD_MAPPING.get(int_rating, 0.0)

        # Update bandits online
        self.bandit_agent.update(movie_id, reward)
        state = self.env.reset(user_id=user_id)
        self.linucb_agent.update(movie_id, reward, state)

        # Update DQN replay memory and perform an online step
        next_state = self.env.loader.get_user_state_vector(
            user_history=[movie_id] + self.env.user_history[:9], user_id=user_id
        )
        self.dqn_agent.memory.push(state, movie_id, reward, next_state, done=False)

        updated_loss = None
        if len(self.dqn_agent.memory) >= 32:
            updated_loss = self.dqn_agent.update(batch_size=32)

        return {
            "status": "success",
            "user_id": user_id,
            "movie_id": movie_id,
            "rating": rating,
            "reward_applied": reward,
            "online_training_loss": updated_loss,
            "message": "Feedback recorded and online models updated.",
        }

    def get_movie(self, movie_id: int) -> Optional[Dict[str, Any]]:
        return self.env.loader.get_movie_info(movie_id)

    def search_movies(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        if self.env.loader.movies_df is None:
            return []
        mask = self.env.loader.movies_df["title"].str.contains(query, case=False, na=False)
        matches = self.env.loader.movies_df[mask].head(limit)
        results = []
        for _, row in matches.iterrows():
            results.append(self.env.loader.get_movie_info(int(row["movie_id"])))
        return results

    def get_health(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "models_available": ["dqn", "bandit", "linucb", "popular", "random"],
            "dqn_weights_loaded": self.dqn_weights_loaded,
            "bandit_weights_loaded": self.bandit_weights_loaded,
            "device": str(self.dqn_agent.device),
        }


def get_recommendation_service() -> RecommendationService:
    return RecommendationService.get_instance()
