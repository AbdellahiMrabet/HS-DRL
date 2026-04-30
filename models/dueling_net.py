# models/dueling_net.py - Dueling DQN Network

import torch
import torch.nn as nn
from config import HIDDEN_DIM


class DuelingNet(nn.Module):
    """Dueling DQN Network for EPRS agent"""
    
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim)
        )
    
    def forward(self, x):
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + a - a.mean(dim=-1, keepdim=True)
DuelingNetwork = DuelingNet
