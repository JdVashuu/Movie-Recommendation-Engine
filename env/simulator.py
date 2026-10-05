from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.config import REWARD_MAPPING
from data.loader import MovieLensLoader


class MovieRecommendEnv:
    def __init__(
        self,
        loader: Optional[MovieLensLoader] = None,
        top_n: int = 5,
        max_steps: int = 10,
    ):
        self.top_n = top_n
        self.max_steps = max_steps
        if loader is None:
            self.loader = MovieLensLoader()
            self.loader.load_data()
        else:
            self.loader = loader

        self.current_user: Optional[int] = None
        self.user_history: List[int] = []
        self.step_count: int = 0
        self.cumulative_reward: float = 0.0

    def reset(self, user_id: Optional[int] = None) -> np.ndarray:
        if user_id is None:
            if self.loader.ratings_df is not None:
                user_id = int(np.random.choice(self.loader.ratings_df["user_id"].unique()))
            else:
                user_id = 1

        self.current_user = user_id
        self.user_history = self.loader.get_user_history(user_id, limit=10)
        self.step_count = 0
        self.cumulative_reward = 0.0
        return self._get_state()

    def step(
        self, action_movie_id: List[int]
    ) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Action : A list of movies recommended by the agent
        Returns : next_state, total_reward, done, info
        """
        self.step_count += 1

        if self.loader.ratings_df is not None and self.current_user is not None:
            user_rating = self.loader.ratings_df[
                (self.loader.ratings_df["user_id"] == self.current_user)
                & (self.loader.ratings_df["movie_id"].isin(action_movie_id))
            ]
        else:
            user_rating = None

        # Granular reward lookup
        individual_rewards: Dict[int, float] = {}
        if user_rating is not None and not user_rating.empty:
            for _, row in user_rating.iterrows():
                rating = int(row["rating"])
                r = REWARD_MAPPING.get(rating, 0.0)
                individual_rewards[int(row["movie_id"])] = r

        total_reward = float(sum(individual_rewards.values()))
        self.cumulative_reward += total_reward

        # Update history only for positive interactions (rating >= 4 -> reward >= 0.6)
        liked_movies = [mid for mid, r in individual_rewards.items() if r >= 0.6]
        self.user_history = (liked_movies + self.user_history)[:10]

        done = self.step_count >= self.max_steps

        info = {
            "individual_rewards": individual_rewards,
            "step_count": self.step_count,
            "cumulative_reward": self.cumulative_reward,
            "liked_movies": liked_movies,
            "hit_count": len(individual_rewards),
        }

        return self._get_state(), total_reward, done, info

    def _get_state(self) -> np.ndarray:
        return self.loader.get_user_state_vector(self.user_history, self.current_user)
