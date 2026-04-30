# models/hierarchical_net.py - Hierarchical Neural Network

import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_NODES, HIDDEN_DIM


class HierarchicalPolicyNetwork(nn.Module):
    """Hierarchical Policy Network for HS-DRL agent"""
    
    def __init__(self, state_dim: int, action_dim: int, num_nodes: int = NUM_NODES):
        super().__init__()
        
        self.num_nodes = num_nodes
        
        # Node feature extractor (CPU, Memory per node)
        node_feature_dim = num_nodes * 2
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
        )
        
        # Pod feature extractor
        pod_feature_dim = 2
        self.pod_encoder = nn.Sequential(
            nn.Linear(pod_feature_dim, HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM // 2, HIDDEN_DIM // 2),
            nn.ReLU(),
        )
        
        # Combined feature processing
        combined_dim = HIDDEN_DIM + (HIDDEN_DIM // 2)
        self.combined_encoder = nn.Sequential(
            nn.Linear(combined_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, action_dim)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch_size = state.shape[0]
        
        # Split state into node features and pod features
        node_features = state[:, :self.num_nodes * 2]
        pod_features = state[:, self.num_nodes * 2:]
        
        # Encode node features
        node_encoded = self.node_encoder(node_features)
        
        # Encode pod features
        pod_encoded = self.pod_encoder(pod_features)
        
        # Combine features
        combined = torch.cat([node_encoded, pod_encoded], dim=-1)
        logits = self.combined_encoder(combined)
        
        return F.softmax(logits, dim=-1)


class HierarchicalValueNetwork(nn.Module):
    """Hierarchical Value Network for HS-DRL agent"""
    
    def __init__(self, state_dim: int, num_nodes: int = NUM_NODES):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class LowerLevelController(nn.Module):
    """Lower-level controller for per-node resource allocation (ADD THIS CLASS)"""
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64):
        """
        Args:
            input_dim: [target_cpu, target_mem, current_cpu, current_mem]
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # [allocated_cpu, allocated_mem]
            nn.Sigmoid()  # Output in [0,1]
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, target_cpu: torch.Tensor, target_mem: torch.Tensor,
                current_cpu: torch.Tensor, current_mem: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target_cpu: Target CPU allocation (from upper policy)
            target_mem: Target memory allocation (from upper policy)
            current_cpu: Current CPU usage of the node
            current_mem: Current memory usage of the node
        
        Returns:
            allocated_cpu: Adjusted CPU allocation after safety constraints
            allocated_mem: Adjusted memory allocation after safety constraints
        """
        x = torch.cat([target_cpu, target_mem, current_cpu, current_mem], dim=-1)
        return self.net(x)


# Alias for backward compatibility (optional)
LowerController = LowerLevelController