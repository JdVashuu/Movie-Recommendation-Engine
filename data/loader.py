import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.config import GENRE_NAMES, ML_100K_DIR, NUM_GENRES


class MovieLensLoader:
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = str(data_dir or ML_100K_DIR)
        self.ratings_df: Optional[pd.DataFrame] = None
        self.movies_df: Optional[pd.DataFrame] = None
        self.user_df: Optional[pd.DataFrame] = None
        self.default_demo_vec: np.ndarray = np.zeros(23, dtype=np.float32)

    def load_data(self):
        ratings_col = ["user_id", "movie_id", "rating", "timestamp"]
        self.ratings_df = pd.read_csv(
            os.path.join(self.data_dir, "u.data"), sep="\t", names=ratings_col
        )

        raw_movies = pd.read_csv(
            os.path.join(self.data_dir, "u.item"),
            sep="|",
            encoding="latin-1",
            header=None,
        )

        # Columns: 0 is movie_id, 1 is title, 5-23 are the 19 genre flags
        selected_cols = [0, 1] + list(range(5, 24))
        self.movies_df = raw_movies.iloc[:, selected_cols].copy()
        
        rename_dict = {0: "movie_id", 1: "title"}
        for idx, genre_name in enumerate(GENRE_NAMES, start=5):
            rename_dict[idx] = genre_name
        self.movies_df.rename(columns=rename_dict, inplace=True)

        users_col = ["user_id", "age", "gender", "occupation", "zip"]
        self.user_df = pd.read_csv(
            os.path.join(self.data_dir, "u.user"), sep="|", names=users_col
        )

        self.user_df["gender"] = (self.user_df["gender"] == "F").astype(int)  # M = 0, F = 1
        self.user_df["age"] = self.user_df["age"] / self.user_df["age"].max()  # age normalization
        self.user_df = pd.get_dummies(self.user_df, columns=["occupation"])

        # Calculate default demographic vector for unseen users (cold-start)
        demo_cols = [c for c in self.user_df.columns if c not in ["user_id", "zip"]]
        self.default_demo_vec = self.user_df[demo_cols].mean(axis=0).values.astype(np.float32)

    def get_implicit_feedback(self, threshold=4):
        df = self.ratings_df.copy()
        df["reward"] = (df["rating"] >= threshold).astype(int)
        return df

    def get_user_history(self, user_id, limit=10):
        if self.ratings_df is None:
            return []
        user_data = self.ratings_df[self.ratings_df["user_id"] == user_id]
        if user_data.empty:
            return []
        sorted_data = user_data.sort_values(by="timestamp", ascending=False)
        return sorted_data.head(limit)["movie_id"].tolist()

    def get_movie_genres(self):
        # Drop title and return only the genre flags indexed by movie_id
        return self.movies_df.drop(columns=["title"]).set_index("movie_id")

    def get_user_state_vector(self, user_history, user_id=None):
        """
        Combines user demographic and genre affinity.
        Gracefully handles cold-start / unseen users.
        """
        genre_vec = np.zeros(NUM_GENRES, dtype=np.float32)
        if user_history and self.movies_df is not None:
            genre_df = self.get_movie_genres()
            history_genres = genre_df.loc[genre_df.index.isin(user_history)]

            if not history_genres.empty:
                genre_vec = history_genres.sum(axis=0).values.astype(np.float32)
                norm = np.linalg.norm(genre_vec)
                if norm > 0:
                    genre_vec = genre_vec / norm

        # Demographic features
        if user_id is not None and self.user_df is not None and not self.user_df.empty:
            matching_user = self.user_df[self.user_df["user_id"] == user_id]
            if not matching_user.empty:
                user_info = matching_user.iloc[0]
                demo_vec = user_info.drop(["user_id", "zip"]).values.astype(np.float32)
            else:
                demo_vec = self.default_demo_vec.copy()
        else:
            demo_vec = self.default_demo_vec.copy()

        full_state = np.concatenate([genre_vec, demo_vec]).astype(np.float32)
        return full_state

    def get_movie_info(self, movie_id: int) -> Optional[Dict]:
        """Fetch title and genre names for a given movie_id."""
        if self.movies_df is None:
            return None
        match = self.movies_df[self.movies_df["movie_id"] == movie_id]
        if match.empty:
            return None
        row = match.iloc[0]
        genres = [g for g in GENRE_NAMES if row.get(g, 0) == 1]
        return {
            "movie_id": int(movie_id),
            "title": str(row["title"]),
            "genres": genres,
        }

    def get_popular_movies(self, limit: int = 10) -> List[int]:
        """Return most popular highly-rated movies."""
        if self.ratings_df is None:
            return list(range(1, limit + 1))
        popular = (
            self.ratings_df[self.ratings_df["rating"] >= 4]
            .groupby("movie_id")
            .size()
            .sort_values(ascending=False)
            .head(limit)
            .index.tolist()
        )
        return [int(mid) for mid in popular]

