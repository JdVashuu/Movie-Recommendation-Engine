import os
from posixpath import sep

import numpy as np
import pandas as pd

DATA_DIR = "data/raw/ml-100k"


class MovieLensLoader:
    def __init__(self):
        self.ratings_df = None
        self.movies_df  = None
        self.user_df = None

    def load_data(self):
        ratings_col = ["user_id", "movie_id", "rating", "timestamp"]
        self.ratings_df = pd.read_csv(
            os.path.join(DATA_DIR, "u.data"), sep="\t", names=ratings_col
        )

        self.movies_df = pd.read_csv(
            os.path.join(DATA_DIR, "u.item"), sep="|", encoding="latin-1", header=None
        )

        self.movies_df = self.movies_df.iloc[:, [0, 1] + list(range(5, 24))]
        self.movies_df.rename(columns={0: "movie_id", 1: "title"}, inplace=True)

        users_col = ["user_id", "age", "gender", "occupation", "zip"]
        self.user_df = pd.read_csv(
            os.path.join(DATA_DIR, "u.user"), sep="|", names=users_col
        )

        self.user_df["gender"] = (self.user_df["gender"] == "F").astype(int)           # M = 0, F = 1
        self.user_df["age"] = self.user_df["age"] / self.user_df["age"].max()          # age normalisation
        self.user_df = pd.get_dummies(self.user_df, columns=["occupation"])


    def get_implicit_feedback(self, threshold=4):
        df = self.ratings_df.copy()
        df["reward"] = (df["rating"] >= threshold).astype(int)
        return df

    def get_user_history(self, user_id, limit=10):
        user_data = self.ratings_df[self.ratings_df["user_id"] == user_id]
        sorted_data = user_data.sort_values(by="timestamp", ascending=False)
        return sorted_data.head(limit)["movie_id"].tolist()

    def get_movie_genres(self):
        # Drop title and return only the genre flags
        return self.movies_df.drop(columns=["title"]).set_index("movie_id")

    def get_user_state_vector(self, user_history, user_id):
        """
        combines user demographic and gender affinity
        """
        
        genre_vec = np.zeros(19)                                                # genre affinity
        if user_history:
            genre_df = self.get_movie_genres()
            history_genres = genre_df.loc[genre_df.index.isin(user_history)]

            if not history_genres.empty:
                genre_vec = history_genres.sum(axis=0).values
                norm = np.linalg.norm(genre_vec)
                if norm > 0:
                    genre_vec = genre_vec / norm

        
        user_info = self.user_df[self.user_df["user_id"] == user_id].iloc(0)
        demo_vec = user_info.drop(['user_id', 'zip']).values.astype(np.float32)

        full_state = np.concatenate([genre_vec, demo_vec]).astype(np.float32)
        return full_state
