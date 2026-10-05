import numpy as np
import pytest

from models.bandit import EpsilonGreedyBandit, LinUCBBandit, UCBBandit


def test_epsilon_greedy_no_zero_index():
    n_movies = 20
    bandit = EpsilonGreedyBandit(n_movies=n_movies, epsilon=0.0)

    # Set all valid movies to negative reward
    for i in range(1, n_movies + 1):
        bandit.update(i, -1.0)

    # Predict top 5: ensure 0 is NEVER recommended
    preds = bandit.predict(n_to_recommend=5)
    assert len(preds) == 5
    assert 0 not in preds
    for mid in preds:
        assert 1 <= mid <= n_movies


def test_epsilon_greedy_update():
    bandit = EpsilonGreedyBandit(n_movies=10, epsilon=0.0)
    bandit.update(movie_id=3, reward=1.0)
    bandit.update(movie_id=3, reward=0.5)

    assert bandit.counts[3] == 2
    assert np.isclose(bandit.values[3], 0.75)

    # Exploitation should pick movie 3 first
    preds = bandit.predict(n_to_recommend=1)
    assert preds[0] == 3


def test_bandit_serialization(tmp_path):
    bandit = EpsilonGreedyBandit(n_movies=10, epsilon=0.0)
    bandit.update(movie_id=5, reward=1.0)

    save_file = str(tmp_path / "bandit.pkl")
    bandit.save(save_file)

    loaded = EpsilonGreedyBandit.load(save_file)
    assert loaded.n_movies == 10
    assert loaded.counts[5] == 1
    assert loaded.values[5] == 1.0
    assert loaded.predict(1)[0] == 5


def test_ucb_bandit():
    ucb = UCBBandit(n_movies=15, c=1.0)
    preds = ucb.predict(5)
    assert len(preds) == 5
    for mid in preds:
        assert 1 <= mid <= 15
        assert mid != 0

    # Updating arm 2 multiple times with high reward
    for _ in range(10):
        ucb.update(movie_id=2, reward=1.0)
    assert ucb.counts[2] == 10
    assert np.isclose(ucb.values[2], 1.0)


def test_linucb_bandit():
    state_dim = 10
    n_movies = 20
    agent = LinUCBBandit(n_movies=n_movies, state_dim=state_dim, alpha=0.5)

    dummy_state = np.ones(state_dim, dtype=np.float32)
    preds = agent.predict(dummy_state, n_to_recommend=5)
    assert len(preds) == 5
    assert 0 not in preds
    for mid in preds:
        assert 1 <= mid <= n_movies

    # Update movie 7 with positive reward on dummy_state
    agent.update(movie_id=7, reward=2.0, state=dummy_state)
    assert agent.counts[7] == 1

    # Should give high score to movie 7
    preds_after = agent.predict(dummy_state, n_to_recommend=5)
    assert 7 in preds_after
