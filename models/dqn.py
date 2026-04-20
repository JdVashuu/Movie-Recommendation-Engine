import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingQNetwork, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Value Stream V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Advantage Stream A(S, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Q(S, a) = V(S) + (A(S, a) - mean(A(S, a)))
        return value + (advantages - advantages.mean(dim=1, keepdim=True))

class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.997, genre_matrix=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        self.genre_matrix = genre_matrix

        self.policy_net = DuelingQNetwork(state_dim, action_dim)
        self.target_net = DuelingQNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimiser = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer()

    def predict(self, state, n_to_recommend=5, diversity_weight=0.2):
        if np.random.rand() < self.epsilon:
            # explore cond - pick random movie id
            return np.random.choice(range(1, self.action_dim + 1), n_to_recommend, replace=False)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze().cpu().numpy()
        
        if self.genre_matrix is None:
            # Normal top K selection
            _, indices = torch.topk(torch.tensor(q_values), n_to_recommend)
            return (indices.numpy() + 1)
        
        # Diversity Reranking
        selected_indices = []
        selected_genres = np.zeros(19)
        current_q = q_values.copy()

        for _ in range(n_to_recommend):
            if len(selected_indices) > 0:
                # Penalty = diversity_weight * (count of genres already in slate)
                genre_penalty = self.genre_matrix @ selected_genres 
                current_q -= diversity_weight * genre_penalty
            
            idx = np.argmax(current_q)
            selected_indices.append(idx)

            # Update selected genre counts
            movie_genres = self.genre_matrix[idx]
            selected_genres += movie_genres

            current_q[idx] = -1e9       # mask selected movie

        return np.array(selected_indices) + 1
        
    def update(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
    
        batch = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions)) - 1
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))

        # current q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # next q values - use policy net to select action and target net to evaluate it
        with torch.no_grad():
            next_action = self.policy_net(next_states).argmax(1, keepdim=True)      # select action using policy network
            next_q = self.target_net(next_states).gather(1, next_action).squeeze()  # evaluate action using target network
            expected_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.MSELoss()(current_q, expected_q)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        # decay epsilon
        if self.epsilon > self.epsilon_min : 
            self.epsilon *= self.epsilon_decay

    def update_target_network(self): 
        self.target_net.load_state_dict(self.policy_net.state_dict())



