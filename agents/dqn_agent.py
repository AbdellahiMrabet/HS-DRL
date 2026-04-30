# agents/dqn_agent.py - DRS (DQN) Agent with fixed state dimension

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from config import (BUFFER_SIZE, BATCH_SIZE, LEARNING_RATE_DQN, GAMMA_DQN,
                    EPSILON_START, EPSILON_MIN, EPSILON_DECAY, DEVICE,
                    DQN_HIDDEN_LAYERS, NUM_NODES)
from agents.base_agent import BaseAgent


class DQNAgent(BaseAgent):
    """DQN Agent with fixed state dimension (NUM_NODES * 2 + 2)"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__(state_dim, action_dim, name="DRS")
        
        # Expected state dimension
        self.expected_state_dim = NUM_NODES * 2 + 2
        self.current_state_dim = state_dim
        
        print(f"[DQN] Initializing: expected_dim={self.expected_state_dim}, actual_dim={state_dim}")
        
        self.q_network = self._build_network(self.expected_state_dim, action_dim)
        self.target_network = self._build_network(self.expected_state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE_DQN)
        self.memory = deque(maxlen=BUFFER_SIZE)
        
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.gamma = GAMMA_DQN
        self.update_target_every = 100
        self.training_step = 0
    
    def _build_network(self, state_dim: int, action_dim: int) -> nn.Module:
        """Build network with fixed state dimension"""
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in DQN_HIDDEN_LAYERS:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        return nn.Sequential(*layers)
    
    def _ensure_state_dim(self, state: np.ndarray) -> np.ndarray:
        """Ensure state has the expected dimension (pad if necessary)"""
        if len(state) < self.expected_state_dim:
            # Pad with zeros
            padding = np.zeros(self.expected_state_dim - len(state))
            state = np.concatenate([state, padding])
        elif len(state) > self.expected_state_dim:
            # Truncate (should not happen)
            state = state[:self.expected_state_dim]
        return state
    
    def act(self, state: np.ndarray, explore: bool = True) -> int:
        # Ensure state has correct dimension
        state = self._ensure_state_dim(state)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> None:
        
        # Ensure states have correct dimension
        state = self._ensure_state_dim(state)
        next_state = self._ensure_state_dim(next_state)
        
        self.memory.append((state, action, reward, next_state, done))
        
        if len(self.memory) < BATCH_SIZE:
            return
        
        batch = np.random.choice(len(self.memory), BATCH_SIZE, replace=False)
        
        states = torch.FloatTensor([self.memory[i][0] for i in batch]).to(DEVICE)
        actions = torch.LongTensor([self.memory[i][1] for i in batch]).to(DEVICE)
        rewards = torch.FloatTensor([self.memory[i][2] for i in batch]).to(DEVICE)
        next_states = torch.FloatTensor([self.memory[i][3] for i in batch]).to(DEVICE)
        dones = torch.BoolTensor([self.memory[i][4] for i in batch]).to(DEVICE)
        
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            next_q[dones] = 0
            targets = rewards + self.gamma * next_q
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        loss = nn.MSELoss()(current_q, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 0.5)
        self.optimizer.step()
        
        self.training_step += 1
        if self.training_step % self.update_target_every == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_stats(self) -> dict:
        return {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'training_step': self.training_step
        }