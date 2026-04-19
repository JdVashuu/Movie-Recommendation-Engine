import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        )
    
    def forward(self, x):
        self.fc(x)

class ReplayBuffer:
    def __init__(self, capacity= 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state , action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)
    
class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.1

        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimiser = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer()

    def predict(self, state, n_to_recommend=5):
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(1, self.action_dim + 1), n_to_recommend, replace=False)
            # explore cond
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
            # picking the top K elements and converting them into one based indexes
            _, indices = torch.topk(q_values, n_to_recommend)
            return (indices.squeeze() + 1).numpy()
        
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

        #current q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # next q values
        next_q = self.target_net(next_states).max(1)[0]
        expected_q = rewards + (1 - dones) * self.gamma + next_q

        loss = nn.MSELoss()(current_q, expected_q)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        #decay epsilon
        if self.epsilon > self.epsilon_min : 
            self.epsilon *= self.epsilon_decay

    def update_target_network(self) : 
        self.target_net.load_state_dict(self.policy_net.state_dict())



