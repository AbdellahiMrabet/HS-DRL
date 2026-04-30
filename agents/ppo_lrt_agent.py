# agents/ppo_lrt_agent.py - PPO-LRT Agent
# Based on: "Optimization of Task-Scheduling Strategy in Edge Kubernetes 
#           Clusters Based on Deep Reinforcement Learning"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from config import (DEVICE, PPO_LRT_CLIP_EPSILON, PPO_LRT_GAMMA, PPO_LRT_LR_ACTOR,
                    PPO_LRT_LR_CRITIC, PPO_LRT_GAE_LAMBDA, PPO_LRT_UPDATE_STEPS,
                    PPO_LRT_EPOCHS, PPO_LRT_HIDDEN_DIM, PPO_LRT_ENTROPY_COEFF,
                    PPO_LRT_ALPHA, PPO_LRT_BETA, PPO_LRT_GAMMA_RT,
                    PPO_LRT_CPU_UPPER, PPO_LRT_MEM_UPPER)
from models.ppo_network import PPONetwork
from agents.base_agent import BaseAgent


class PPOLRTAgent(BaseAgent):
    """
    PPO-LRT: Proximal Policy Optimization with Least Response Time.
    
    From paper Section 3:
    - PPO with clip ε=0.2 for stable policy updates
    - LRT integrated into reward: alpha*(1/sigma_c + 1/sigma_m) - β*p - gamma*R_i
    - Update every 30 steps
    - Actor LR: 3e-4, Critic LR: 1e-3
    - Discount gamma: 0.99, GAE λ: 0.95
    """
    
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__(state_dim, action_dim, name="PPO-LRT")
        
        self.network = PPONetwork(state_dim, action_dim, PPO_LRT_HIDDEN_DIM).to(DEVICE)
        self.optimizer = optim.Adam([
            {'params': self.network.actor.parameters(), 'lr': PPO_LRT_LR_ACTOR},
            {'params': self.network.critic.parameters(), 'lr': PPO_LRT_LR_CRITIC},
            {'params': self.network.shared.parameters(), 'lr': PPO_LRT_LR_ACTOR}
        ])
        
        self.gamma = PPO_LRT_GAMMA
        self.clip_epsilon = PPO_LRT_CLIP_EPSILON
        self.gae_lambda = PPO_LRT_GAE_LAMBDA
        self.update_steps = PPO_LRT_UPDATE_STEPS
        self.ppo_epochs = PPO_LRT_EPOCHS
        self.entropy_coeff = PPO_LRT_ENTROPY_COEFF
        
        # Trajectory buffer
        self.states, self.actions, self.rewards = [], [], []
        self.values, self.log_probs, self.dones = [], [], []
        self.response_times = []
        self.steps_since_update = 0
        
        self.losses, self.avg_rt = [], deque(maxlen=100)
    
    def act(self, state: np.ndarray, explore: bool = True) -> int:
        if explore:
            action, log_prob, value = self.network.get_action(state, deterministic=False)
        else:
            action, log_prob, value = self.network.get_action(state, deterministic=True)
        self.last_log_prob = log_prob
        self.last_value = value
        return action
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> None:
        
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(self.last_value)
        self.log_probs.append(self.last_log_prob)
        self.dones.append(done)
        self.steps_since_update += 1
        
        if self.steps_since_update >= self.update_steps:
            self._ppo_update()
            self.steps_since_update = 0
    
    def update_response_time(self, node_idx: int, response_time_ms: float):
        self.response_times.append(response_time_ms)
        self.avg_rt.append(response_time_ms)
    
    def _compute_gae(self):
        advantages, gae = [], 0
        
        if self.states:
            state_t = torch.FloatTensor(self.states[-1]).unsqueeze(0).to(DEVICE)
            _, next_value = self.network.forward(state_t)
            next_value = next_value.item()
        else:
            next_value = 0.0
        
        values = self.values + [next_value]
        
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + self.gamma * values[t + 1] * (1 - self.dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        
        advantages = np.array(advantages)
        returns = advantages + np.array(self.values)
        
        return advantages, returns
    
    def _ppo_update(self):
        if len(self.states) < 2:
            return
        
        states = torch.FloatTensor(np.array(self.states)).to(DEVICE)
        actions = torch.LongTensor(np.array(self.actions)).to(DEVICE)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(DEVICE)
        
        advantages, returns = self._compute_gae()
        advantages = torch.FloatTensor(advantages).to(DEVICE)
        returns = torch.FloatTensor(returns).to(DEVICE)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(len(states))
            for start in range(0, len(states), 64):
                batch_idx = indices[start:start + 64]
                
                new_log_probs, values, entropy = self.network.evaluate_actions(
                    states[batch_idx], actions[batch_idx]
                )
                
                ratio = torch.exp(new_log_probs - old_log_probs[batch_idx])
                surr1 = ratio * advantages[batch_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[batch_idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.MSELoss()(values, returns[batch_idx])
                
                loss = policy_loss + 0.5 * value_loss - self.entropy_coeff * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()
                
                self.losses.append(loss.item())
        
        self.states, self.actions, self.rewards = [], [], []
        self.values, self.log_probs, self.dones = [], [], []
    
    def _adapt_to_new_state_dim(self, new_state_dim: int):
        if new_state_dim == self.state_dim:
            return
        self.state_dim = new_state_dim
        old_hidden = self.network.shared[0].out_features
        self.network = PPONetwork(new_state_dim, self.action_dim, old_hidden).to(DEVICE)
        self.optimizer = optim.Adam([
            {'params': self.network.actor.parameters(), 'lr': PPO_LRT_LR_ACTOR},
            {'params': self.network.critic.parameters(), 'lr': PPO_LRT_LR_CRITIC},
            {'params': self.network.shared.parameters(), 'lr': PPO_LRT_LR_ACTOR}
        ])
    
    def get_stats(self) -> dict:
        return {
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0.0,
            'avg_response_time': np.mean(self.avg_rt) if self.avg_rt else 0.0,
            'buffer_size': len(self.states)
        }
