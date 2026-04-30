# agents/rlsk_agent.py - RLSK Agent

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from config import (BUFFER_SIZE, BATCH_SIZE, LEARNING_RATE_DQN, GAMMA_DQN,
                    EPSILON_START, EPSILON_MIN, EPSILON_DECAY, DEVICE, NUM_NODES)
from models.tiny_net import TinyNet
from agents.base_agent import BaseAgent


class RLSKAgent(BaseAgent):
    """RLSK Agent with load balancing bias"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__(state_dim, action_dim, name="RLSK")
        
        self.q_network = TinyNet(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE_DQN)
        self.memory = deque(maxlen=BUFFER_SIZE)
        
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.gamma = GAMMA_DQN
        self.num_nodes = NUM_NODES
    
    def act(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            q_values = self.q_network(torch.FloatTensor(state).to(DEVICE))
            # RLSK bias: prefer less loaded nodes
            for i in range(min(self.num_nodes, self.action_dim)):
                q_values[i] -= state[i * 2] * 0.15
            return q_values.argmax().item()
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> None:
        
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
            next_q = self.q_network(next_states).max(1)[0]
            next_q[dones] = 0
            targets = rewards + self.gamma * next_q
        
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        loss = nn.MSELoss()(current_q, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def get_stats(self) -> dict:
        return {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory)
        }
