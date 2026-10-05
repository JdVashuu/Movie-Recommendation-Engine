import numpy as np
import pytest

from core.config import STATE_DIM
from data.loader import MovieLensLoader
from env.simulator import MovieRecommendEnv


@pytest.fixture(scope="module")
def env():
    loader = MovieLensLoader()
    loader.load_data()
    return MovieRecommendEnv(loader=loader, top_n=5, max_steps=3)


def test_env_reset(env):
    state = env.reset(user_id=1)
    assert isinstance(state, np.ndarray)
    assert state.shape == (STATE_DIM,)
    assert env.current_user == 1
    assert env.step_count == 0


def test_env_step_and_done(env):
    state = env.reset(user_id=1)
    action = [1, 2, 3, 4, 5]

    next_state, reward, done, info = env.step(action)
    assert next_state.shape == (STATE_DIM,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "individual_rewards" in info
    assert info["step_count"] == 1
    assert not done

    # Step up to max_steps (max_steps=3)
    _, _, done, _ = env.step(action)
    assert not done
    _, _, done, info = env.step(action)
    assert done
    assert info["step_count"] == 3


def test_env_cold_start_user(env):
    state = env.reset(user_id=888888)
    assert state.shape == (STATE_DIM,)
    next_state, reward, done, info = env.step([10, 20])
    assert next_state.shape == (STATE_DIM,)
    assert not np.isnan(next_state).any()
