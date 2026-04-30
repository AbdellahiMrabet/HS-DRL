# models/ppo_network.py - PPO Actor-Critic Network for PPO-LRT

import torch
import torch.nn as nn
import numpy as np
from config import DEVICE


class PPONetwork(nn.Module):
    """PPO Actor-Critic Network for PPO-LRT."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PPONetwork, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, state):
        features = self.shared(state)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value
    
    def get_action(self, state, deterministic=False):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        
        logits, value = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        
        if deterministic:
            action = probs.argmax(dim=-1).item()
            return action, 0.0, value.item()
        
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def evaluate_actions(self, state, action):
        logits, value = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(action)
        entropy = dist.entropy().mean()
        return log_probs, value.squeeze(), entropy
