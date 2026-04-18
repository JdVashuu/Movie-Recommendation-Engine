import os
from posixpath import sep

import numpy as np
import pandas as pd

DATA_DIR = "data/raw/ml-100k"


class MovieLensLoader:
    def __init__(self):
        self.ratings_df = None
        self.movies_df = None

    def load_data(self):
        ratings_col = ["user_id", "timestamp", "rating", "movie_id"]
        self.ratings_df = pd.read_csv(
            os.path.join(DATA_DIR, "u.data"), sep="\t", names=ratings_col
        )

        self.movies_df = pd.read_csv(
            os.path.join(DATA_DIR, "u.item"), sep="|", encoding="latin-1", header=None
        )

        self.movies_df = self.movies_df.iloc[:, [0, 1] + list(range(5, 24))]
        self.movies_df.rename(columns={0: "movie_id", 1: "title"}, inplace=True)

    def get_implicit_feedback(self, threshold=4):
        df = self.ratings_df.copy()
        df["reward"] = (df["rating"] >= threshold).astype(int)
        return df

    def get_user_history(self, user_id, limit=10):
        user_data = self.ratings_df[self.ratings_df["user_id"] == user_id]
        sorted_data = user_data.sort_values(by="timestamp", ascending=False)
        return sorted_data.head(limit)["movie_id"].tolist()

    def get_movie_genres(self):
        return self.movies_df.drop(columns=["title"]).set_index("movie_id")
