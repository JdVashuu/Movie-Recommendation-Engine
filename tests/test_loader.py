import numpy as np
import pytest

from core.config import GENRE_NAMES, NUM_GENRES, STATE_DIM
from data.loader import MovieLensLoader


@pytest.fixture(scope="module")
def loader():
    l = MovieLensLoader()
    l.load_data()
    return l


def test_loader_load_data(loader):
    assert loader.ratings_df is not None
    assert len(loader.ratings_df) == 100000

    assert loader.movies_df is not None
    assert len(loader.movies_df) == 1682

    assert loader.user_df is not None
    assert len(loader.user_df) == 943


def test_genre_columns(loader):
    for genre in GENRE_NAMES:
        assert genre in loader.movies_df.columns


def test_implicit_feedback(loader):
    fb = loader.get_implicit_feedback(threshold=4)
    assert "reward" in fb.columns
    # Check that ratings >= 4 are mapped to 1, others to 0
    assert (fb.loc[fb["rating"] >= 4, "reward"] == 1).all()
    assert (fb.loc[fb["rating"] < 4, "reward"] == 0).all()


def test_user_history(loader):
    history = loader.get_user_history(user_id=1, limit=5)
    assert isinstance(history, list)
    assert len(history) <= 5
    for mid in history:
        assert 1 <= mid <= 1682


def test_user_state_vector_existing_user(loader):
    history = loader.get_user_history(user_id=1, limit=10)
    state = loader.get_user_state_vector(user_history=history, user_id=1)
    assert isinstance(state, np.ndarray)
    assert state.shape == (STATE_DIM,)
    assert not np.isnan(state).any()


def test_user_state_vector_cold_start(loader):
    # Cold-start / unknown user should return valid default vector without raising IndexError
    state = loader.get_user_state_vector(user_history=[], user_id=999999)
    assert isinstance(state, np.ndarray)
    assert state.shape == (STATE_DIM,)
    assert not np.isnan(state).any()


def test_get_movie_info(loader):
    info = loader.get_movie_info(1)
    assert info is not None
    assert info["movie_id"] == 1
    assert "Toy Story" in info["title"]
    assert "Animation" in info["genres"]


def test_get_popular_movies(loader):
    popular = loader.get_popular_movies(limit=10)
    assert len(popular) == 10
    for mid in popular:
        assert 1 <= mid <= 1682
