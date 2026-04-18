import numpy as np

from data.loader import MovieLensLoader


class MovieRecommendEnv:
    def __init__(self, top_n=5):
        self.top_n = top_n
        self.loader = MovieLensLoader()
        self.loader.load_data()
        self.current_user = None
        self.user_history = []

    def reset(self, user_id=None):
        if user_id is None:
            user_id = np.random.choice(self.loader.ratings_df["user_id"].unique())

        self.current_user = user_id
        self.user_history = self.loader.get_user_history(user_id, limit=10)
        return self._get_state()

    def step(self, action_movie_id):
        """
        Action : A list of movies recommended by the agent
        Returns : next_state, total_reward, done, info
        """

        user_rating = self.loader.ratings_df[
            (self.loader.ratings_df["user_id"] == self.current_user)
            & (self.loader.ratings_df["movie_id"].isin(action_movie_id))
        ]

        # Calculate individual rewards for the bandit to learn
        individual_rewards = {
            row["movie_id"]: 1 if row["rating"] >= 4 else 0
            for _, row in user_rating.iterrows()
        }

        reward = sum(individual_rewards.values())

        liked_movies = [mid for mid, r in individual_rewards.items() if r == 1]
        self.user_history = (liked_movies + self.user_history)[:10]

        done = False

        return (
            self._get_state(),
            reward,
            done,
            {"individual_rewards": individual_rewards},
        )

    def _get_state(self):
        return {"user_id": self.current_user, "history": self.user_history}
