import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple

class MastermindAgent(nn.Module):
    def __init__(self, code_length: int = 4, num_colors: int = 6, max_guesses: int = 10, hidden_size: int = 128):
        super().__init__()
        self.code_length = code_length
        self.num_colors = num_colors
        self.observation_size = max_guesses * (code_length + 2)
        self.action_size = num_colors ** code_length
        
        self.network = nn.Sequential(
            nn.Linear(self.observation_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, self.action_size)
        )
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)
        self.saved_log_probs = []
        self.rewards = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def select_action(self, observation: np.ndarray) -> int:
        state = torch.FloatTensor(observation).unsqueeze(0)
        logits = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        self.saved_log_probs.append(dist.log_prob(action))
        return action.item()
    
    def update(self, gamma: float = 0.99) -> float:
        if not self.rewards:
            return 0.0
        R = 0
        returns = []
        for r in reversed(self.rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
        self.saved_log_probs = []
        self.rewards = []
        return loss.item()
