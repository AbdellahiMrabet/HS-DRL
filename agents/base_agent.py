# agents/base_agent.py - Base Agent Class

from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple


class BaseAgent(ABC):
    """Abstract base class for all RL agents"""
    
    def __init__(self, state_dim: int, action_dim: int, name: str = "BaseAgent"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.name = name
        self.training_step = 0
    
    @abstractmethod
    def act(self, state: np.ndarray, explore: bool = True) -> int:
        """Select an action based on current state"""
        pass
    
    @abstractmethod
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> None:
        """Update agent's policy based on experience"""
        pass
    
    @abstractmethod
    def get_stats(self) -> dict:
        """Return agent statistics"""
        pass
    
    def save(self, path: str):
        """Save agent model"""
        pass
    
    def load(self, path: str):
        """Load agent model"""
        pass