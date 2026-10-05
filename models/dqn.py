import os
import random
from collections import deque
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super(DuelingQNetwork, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        # Value Stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # Advantage Stream A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        return value + (advantages - advantages.mean(dim=1, keepdim=True))


class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.997,
        epsilon_min: float = 0.01,
        genre_matrix: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.genre_matrix = genre_matrix

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.policy_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimiser = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer()

    def predict(
        self,
        state: np.ndarray,
        n_to_recommend: int = 5,
        diversity_weight: float = 0.2,
        deterministic: bool = False,
    ) -> np.ndarray:
        n_to_recommend = min(n_to_recommend, self.action_dim)

        # Exploration branch (if not deterministic)
        if not deterministic and np.random.rand() < self.epsilon:
            return np.random.choice(
                np.arange(1, self.action_dim + 1), n_to_recommend, replace=False
            )

        self.policy_net.eval()
        state_tensor = (
            torch.FloatTensor(state).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze().cpu().numpy()

        if self.genre_matrix is None or diversity_weight <= 0:
            # Standard top-K selection
            indices = np.argsort(q_values)[-n_to_recommend:][::-1]
            return indices + 1

        # Diversity Reranking with linear slate penalty
        selected_indices: List[int] = []
        num_genres = self.genre_matrix.shape[1]
        selected_genres = np.zeros(num_genres, dtype=np.float32)
        base_q = q_values.copy()
        current_q = base_q.copy()

        for _ in range(n_to_recommend):
            idx = int(np.argmax(current_q))
            selected_indices.append(idx)
            current_q[idx] = -1e9  # Mask selected movie

            # Accumulate genre representation in recommended slate
            movie_genres = self.genre_matrix[idx]
            selected_genres += movie_genres
            genre_penalty = self.genre_matrix @ selected_genres

            # Re-penalize unselected candidates linearly against baseline Q
            unselected = current_q > -1e8
            current_q[unselected] = (
                base_q[unselected] - diversity_weight * genre_penalty[unselected]
            )

        return np.array(selected_indices) + 1

    def update(self, batch_size: int = 64) -> Optional[float]:
        if len(self.memory) < batch_size:
            return None

        self.policy_net.train()
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        # Convert 1-based action to 0-based index
        actions_t = torch.LongTensor(np.array(actions) - 1).to(self.device)
        rewards_t = torch.FloatTensor(np.array(rewards)).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(np.array(dones).astype(np.float32)).to(self.device)

        # Current Q-values
        current_q = self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Double DQN target: policy net picks action, target net evaluates
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(1, keepdim=True)
            next_q = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            expected_q = rewards_t + (1 - dones_t) * self.gamma * next_q

        loss = nn.MSELoss()(current_q, expected_q)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return float(loss.item())

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        torch.save(self.policy_net.state_dict(), filepath)

    def load(self, filepath: str, map_location: Optional[Union[str, torch.device]] = None):
        if map_location is None:
            map_location = self.device
        state_dict = torch.load(filepath, map_location=map_location, weights_only=True)
        self.policy_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        self.policy_net.eval()
