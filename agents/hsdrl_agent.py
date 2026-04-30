# agents/hsdrl_agent.py - HS-DRL Agent with PPO and Safety Awareness

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Dict, List, Tuple, Optional
from config import (BUFFER_SIZE, BATCH_SIZE, LEARNING_RATE_PPO, GAMMA_PPO, 
                    GAE_LAMBDA, PPO_CLIP_EPSILON, PPO_ENTROPY_COEFF, 
                    PPO_VALUE_COEFF, DEVICE, NUM_NODES,
                    EPSILON_START, EPSILON_MIN, EPSILON_DECAY)
from agents.base_agent import BaseAgent


class ActorNetwork(nn.Module):
    """Policy Network for HS-DRL"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [128, 128]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Actor outputs: mean and log_std for Gaussian policy
        self.mean_layer = nn.Linear(prev_dim, action_dim)
        self.log_std_layer = nn.Linear(prev_dim, action_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        features = self.feature_extractor(state)
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, -20, 2)
        return mean, log_std
    
    def get_action(self, state, deterministic=False):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        if deterministic:
            action = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()
        
        # Compute log probability
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Apply tanh squashing
        action = torch.tanh(action)
        
        return action, log_prob
    
    def evaluate(self, state, action):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        
        dist = torch.distributions.Normal(mean, std)
        
        # Inverse tanh to get pre-squashed action
        action = torch.clamp(action, -0.999, 0.999)
        raw_action = 0.5 * torch.log((1 + action) / (1 - action))
        
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        
        # Add tanh correction
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy


class CriticNetwork(nn.Module):
    """Value Network for HS-DRL"""
    
    def __init__(self, state_dim: int, hidden_dims: List[int] = [128, 128]):
        super().__init__()
        
        layers = []
        prev_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state):
        return self.network(state).squeeze(-1)


class HSDRLAgent(BaseAgent):
    """
    HS-DRL Agent with PPO and Safety-Aware Learning
    
    Features:
    - Actor-Critic architecture with PPO
    - GAE for advantage estimation
    - Entropy bonus for exploration
    - Clipped surrogate objective for stable updates
    """
    
    def __init__(self, state_dim: int, action_dim: int, name: str = "HS-DRL"):
        super().__init__(state_dim, action_dim, name)
        
        # Networks
        self.actor = ActorNetwork(state_dim, action_dim).to(DEVICE)
        self.critic = CriticNetwork(state_dim).to(DEVICE)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE_PPO)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE_PPO)
        
        # Hyperparameters
        self.gamma = GAMMA_PPO
        self.gae_lambda = GAE_LAMBDA
        self.clip_epsilon = PPO_CLIP_EPSILON
        self.entropy_coeff = PPO_ENTROPY_COEFF
        self.value_coeff = PPO_VALUE_COEFF
        
        # Exploration (epsilon-greedy fallback for discrete action selection)
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_MIN
        self.epsilon_decay = EPSILON_DECAY
        
        # Experience buffer
        self.buffer = deque(maxlen=BUFFER_SIZE)
        self.batch_size = BATCH_SIZE
        
        # Training stats
        self.training_step = 0
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        
        # PPO update parameters
        self.ppo_epochs = 10
        
        # Response time tracking (per-node)
        self.node_response_times = {i: 100.0 for i in range(NUM_NODES)}
        
        # For state dimension adaptation
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def _adapt_to_new_state_dim(self, new_state_dim: int):
        """Adapt networks to new state dimension (for dynamic environments)"""
        if new_state_dim == self.state_dim:
            return
        
        print(f"  🔄 Adapting HS-DRL from state_dim {self.state_dim} to {new_state_dim}")
        
        # Create new networks
        old_actor = self.actor
        old_critic = self.critic
        
        self.actor = ActorNetwork(new_state_dim, self.action_dim).to(DEVICE)
        self.critic = CriticNetwork(new_state_dim).to(DEVICE)
        
        # Copy weights where possible (for overlapping dimensions)
        with torch.no_grad():
            min_dim = min(self.state_dim, new_state_dim)
            
            # Copy first layer weights for overlapping features
            self.actor.feature_extractor[0].weight[:, :min_dim] = old_actor.feature_extractor[0].weight[:, :min_dim]
            self.critic.network[0].weight[:, :min_dim] = old_critic.network[0].weight[:, :min_dim]
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE_PPO)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LEARNING_RATE_PPO)
        
        self.state_dim = new_state_dim
    
    def act(self, state: np.ndarray, explore: bool = True) -> int:
        """Select action based on current state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        
        # Epsilon-greedy exploration
        if explore and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            action, _ = self.actor.get_action(state_tensor, deterministic=not explore)
            action = action.squeeze(0).cpu().numpy()
        
        # Convert continuous action to discrete (for node selection)
        # Action is in [-1, 1], map to [0, action_dim-1]
        discrete_action = int((action[0] + 1) / 2 * (self.action_dim - 1))
        discrete_action = np.clip(discrete_action, 0, self.action_dim - 1)
        
        return discrete_action
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> None:
        """Store experience and update policy"""
        self.buffer.append((state, action, reward, next_state, done))
        self.training_step += 1
        
        if len(self.buffer) >= self.batch_size:
            self._update_policy()
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def _compute_gae(self, rewards: List[float], values: List[float], 
                     dones: List[bool]) -> Tuple[List[float], List[float]]:
        """Compute Generalized Advantage Estimation"""
        advantages = []
        returns = []
        gae = 0
        
        values = values + [0]
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        
        return advantages, returns
    
    def _update_policy(self):
        """Update policy using PPO"""
        if len(self.buffer) < self.batch_size:
            return
        
        # Sample batch
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        
        states = torch.FloatTensor([b[0] for b in batch]).to(DEVICE)
        actions = torch.LongTensor([b[1] for b in batch]).to(DEVICE)
        rewards = [b[2] for b in batch]
        next_states = torch.FloatTensor([b[3] for b in batch]).to(DEVICE)
        dones = [b[4] for b in batch]
        
        # Convert discrete actions to continuous for PPO
        continuous_actions = torch.FloatTensor(
            [(a / (self.action_dim - 1)) * 2 - 1 for a in actions.numpy()]
        ).unsqueeze(1).to(DEVICE)
        
        # Get values
        with torch.no_grad():
            values = self.critic(states).cpu().numpy().tolist()
        
        # Compute advantages and returns
        advantages, returns = self._compute_gae(rewards, values, dones)
        advantages = torch.FloatTensor(advantages).to(DEVICE)
        returns = torch.FloatTensor(returns).to(DEVICE)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(self.ppo_epochs):
            # Actor loss
            log_probs, entropy = self.actor.evaluate(states, continuous_actions)
            
            with torch.no_grad():
                old_log_probs, _ = self.actor.evaluate(states, continuous_actions)
            
            ratio = torch.exp(log_probs - old_log_probs)
            
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Entropy bonus
            entropy_loss = -self.entropy_coeff * entropy.mean()
            
            # Total actor loss
            total_actor_loss = actor_loss + entropy_loss
            
            # Critic loss
            values_pred = self.critic(states)
            critic_loss = self.value_coeff * nn.MSELoss()(values_pred, returns)
            
            # Update actor
            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
            self.actor_optimizer.step()
            
            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
            
            self.actor_losses.append(total_actor_loss.item())
            self.critic_losses.append(critic_loss.item())
    
    def update_response_time(self, node_idx: int, response_time_ms: float):
        """Update per-node response time tracking"""
        if node_idx < len(self.node_response_times):
            self.node_response_times[node_idx] = response_time_ms
    
    def get_stats(self) -> dict:
        """Return agent statistics"""
        return {
            'name': self.name,
            'epsilon': self.epsilon,
            'buffer_size': len(self.buffer),
            'training_step': self.training_step,
            'avg_actor_loss': np.mean(self.actor_losses[-100:]) if self.actor_losses else 0,
            'avg_critic_loss': np.mean(self.critic_losses[-100:]) if self.critic_losses else 0,
            'node_response_times': self.node_response_times.copy()
        }
    
    def save(self, path: str):
        """Save agent model"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'node_response_times': self.node_response_times
        }, path)
        print(f"  ✓ Saved HS-DRL model to {path}")
    
    def load(self, path: str):
        """Load agent model"""
        checkpoint = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.node_response_times = checkpoint.get('node_response_times', {i: 100.0 for i in range(NUM_NODES)})
        print(f"  ✓ Loaded HS-DRL model from {path}")