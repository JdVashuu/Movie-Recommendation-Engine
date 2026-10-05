import numpy as np
import pytest
import torch

from models.dqn import DQNAgent, DuelingQNetwork, ReplayBuffer


def test_dueling_q_network_forward():
    state_dim = 42
    action_dim = 100
    model = DuelingQNetwork(state_dim, action_dim)

    batch_size = 8
    dummy_input = torch.randn(batch_size, state_dim)
    output = model(dummy_input)

    assert output.shape == (batch_size, action_dim)


def test_replay_buffer():
    buf = ReplayBuffer(capacity=10)
    assert len(buf) == 0

    s = np.zeros(5)
    buf.push(s, 1, 1.0, s, False)
    assert len(buf) == 1

    sample = buf.sample(1)
    assert len(sample) == 1
    assert sample[0][1] == 1  # action


def test_dqn_agent_predict():
    state_dim = 42
    action_dim = 50
    genre_matrix = np.zeros((action_dim, 19), dtype=np.float32)
    # First 5 movies are Action (genre 1), next 5 are Comedy (genre 5)
    genre_matrix[:5, 1] = 1.0
    genre_matrix[5:10, 5] = 1.0

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        genre_matrix=genre_matrix,
        epsilon=0.0,
    )

    dummy_state = np.zeros(state_dim, dtype=np.float32)
    preds = agent.predict(dummy_state, n_to_recommend=5, diversity_weight=0.5, deterministic=True)

    assert len(preds) == 5
    assert 0 not in preds
    for mid in preds:
        assert 1 <= mid <= action_dim


def test_dqn_agent_update():
    state_dim = 42
    action_dim = 50
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim, lr=1e-3)

    # Populate replay buffer with synthetic experiences
    for i in range(70):
        s = np.random.randn(state_dim).astype(np.float32)
        ns = np.random.randn(state_dim).astype(np.float32)
        agent.memory.push(s, np.random.randint(1, action_dim + 1), float(np.random.choice([0.0, 1.0])), ns, False)

    loss = agent.update(batch_size=32)
    assert loss is not None
    assert isinstance(loss, float)
    assert loss >= 0.0


def test_dqn_save_and_load(tmp_path):
    state_dim = 42
    action_dim = 50
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)

    weight_path = str(tmp_path / "dqn_test.pth")
    agent.save(weight_path)

    agent_loaded = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    agent_loaded.load(weight_path)

    # Weights in policy_net should match
    for p1, p2 in zip(agent.policy_net.parameters(), agent_loaded.policy_net.parameters()):
        assert torch.equal(p1, p2)
