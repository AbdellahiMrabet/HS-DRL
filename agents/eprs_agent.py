# agents/eprs_agent.py - EPRS Agent (D3QN + PER)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from config import (BUFFER_SIZE, BATCH_SIZE, LEARNING_RATE_EPRS, GAMMA_EPRS,
                    EPSILON_START, EPSILON_MIN, EPSILON_DECAY, DEVICE,
                    PER_ALPHA, PER_BETA_START, PER_BETA_INCREMENT, PER_EPSILON)
from models.dueling_net import DuelingNet
from utils.sum_tree import SumTree
from agents.base_agent import BaseAgent


class EPRSAgent(BaseAgent):
    """EPRS Agent with D3QN and Prioritized Experience Replay"""
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__(state_dim, action_dim, name="EPRS")
        
        self.online_net = DuelingNet(state_dim, action_dim).to(DEVICE)
        self.target_net = DuelingNet(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.online_net.state_dict())
        
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=LEARNING_RATE_EPRS)
        self.memory = SumTree(BUFFER_SIZE)
        
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        self.gamma = GAMMA_EPRS
        self.alpha = PER_ALPHA
        self.beta = PER_BETA_START
        self.beta_increment = PER_BETA_INCREMENT
        self.epsilon_td = PER_EPSILON
        self.steps = 0
        self.update_target_every = 100
    
    def act(self, state: np.ndarray, explore: bool = True) -> int:
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            q_values = self.online_net(torch.FloatTensor(state).unsqueeze(0).to(DEVICE))
            return q_values.argmax().item()
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> None:
        
        # Compute TD error for priority
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            next_t = torch.FloatTensor(next_state).unsqueeze(0).to(DEVICE)
            
            next_q = self.online_net(next_t)
            next_action = next_q.argmax(dim=-1, keepdim=True)
            target_q = self.target_net(next_t).gather(1, next_action).squeeze()
            
            if done:
                target = reward
            else:
                target = reward + self.gamma * target_q
            
            current_q = self.online_net(state_t)[0][action]
            td_error = abs(target - current_q).item()
        
        priority = (td_error + self.epsilon_td) ** self.alpha
        self.memory.add(priority, (state, action, reward, next_state, done))
        
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        
        if self.memory.size < BATCH_SIZE:
            return
        
        # Sample with priorities
        batch = []
        segment = self.memory.total / BATCH_SIZE  # FIXED: Changed from total() to total (property)
        for i in range(BATCH_SIZE):
            s = np.random.uniform(segment * i, segment * (i + 1))
            batch.append(self.memory.get(s))
        
        states = torch.FloatTensor([b[0] for b in batch]).to(DEVICE)
        actions = torch.LongTensor([b[1] for b in batch]).to(DEVICE)
        rewards = torch.FloatTensor([b[2] for b in batch]).to(DEVICE)
        next_states = torch.FloatTensor([b[3] for b in batch]).to(DEVICE)
        dones = torch.BoolTensor([b[4] for b in batch]).to(DEVICE)
        
        with torch.no_grad():
            next_q = self.online_net(next_states)
            next_actions = next_q.argmax(dim=-1, keepdim=True)
            target_q = self.target_net(next_states).gather(1, next_actions).squeeze()
            target_q[dones] = 0
            targets = rewards + self.gamma * target_q
        
        current_q = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        loss = nn.MSELoss()(current_q, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.beta = min(1.0, self.beta + self.beta_increment)
    
    def get_stats(self) -> dict:
        return {
            'epsilon': self.epsilon,
            'beta': self.beta,
            'memory_size': self.memory.size,
            'steps': self.steps
        }