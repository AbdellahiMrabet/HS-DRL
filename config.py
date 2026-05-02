# config.py - Centralized Configuration

import torch
import numpy as np
import os

# ============================================================
# HARDWARE CONFIGURATION
# ============================================================

# Set number of threads for stability
torch.set_num_threads(1)
torch.set_default_dtype(torch.float32)

# Robust device detection
DEVICE = torch.device("cpu")

# ============================================================
# KUBERNETES AVAILABILITY
# ============================================================

K8S_AVAILABLE = True

# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 16
BUFFER_SIZE = 1000
HIDDEN_DIM = 64
NUM_EPISODES = 900
MAX_STEPS = 20
NUM_NODES = 4

# ============================================================
# KUBERNETES CONFIGURATION
# ============================================================

POD_NAMESPACE = "default"
POD_IMAGE = [
    "nginx:alpine", 
    "abdellahimrabet/code:firstChart-client",
    "abdellahimrabet/code:firstChart-server"
]

# Constants for TTL range. No random generation here.
POD_TTL_MIN = 90
POD_TTL_MAX = 120

def generate_pod_ttl():
    """
    Generates a random TTL for a single pod instance.
    Returns a float representing seconds.
    """
    base_ttl = np.random.randint(POD_TTL_MIN, POD_TTL_MAX)
    # Add ±20% variation for realism
    variation = np.random.uniform(0.8, 1.2)
    return float(base_ttl * variation)

# ============================================================
# RESOURCE LIMITS
# ============================================================

CPU_LIMIT = 0.85
MEM_LIMIT = 0.85
DISK_IO_LIMIT = 100
DISK_IO_IDLE = 5

# Response Time Thresholds (ms)
RESPONSE_TIME_THRESHOLD_API = 750      # API call only
RESPONSE_TIME_THRESHOLD_FULL = 950     # Full decision process

# ============================================================
# POISSON ARRIVAL CONFIGURATION
# ============================================================

POD_ARRIVAL_RATE = 11.0
PATTERNS = [
                    {'cpu': 0.5, 'mem': 384, 'desc': 'tiny-pod'},
                    {'cpu': 0.60, 'mem': 512, 'desc': 'small-pod'},
                    {'cpu': 0.7, 'mem': 768, 'desc': 'medium-pod'},
                    {'cpu': 0.80, 'mem': 1280, 'desc': 'large-pod'},
                    {'cpu': 0.90, 'mem': 1792, 'desc': 'large-pod'},
            ]

# ============================================================
# REWARD WEIGHTS
# ============================================================

REWARD_BASE = 2.0
REWARD_EFFICIENCY_FACTOR = 5.0
REWARD_LOAD_PENALTY_FACTOR = 3.0
REWARD_FAILURE_PENALTY = -4.0
REWARD_DELAY_PENALTY = -0.4
REWARD_NODE_NOT_READY_PENALTY = -5.0
REWARD_CONSTRAINT_PENALTY = -3.0

#load bonus thresholds
LD_EXCELLENT = 0.1      #+2
LD_VERY_GOOD = 0.2      #+1      
LD_GOOD = 0.3           #+0.5
LD_VEAVY = 0.6          #-1
LD_VERY_HEAVY = 0.7     #-2
LD_OVERLOADED = 0.8     #-3

# Response time bonus thresholds (ms)
RT_EXCELLENT = 50      # +2.0 bonus
RT_GOOD = 100          # +1.0 bonus
RT_SLOW = 200          # -1.0 penalty
RT_VERY_SLOW = 300     # -3.0 penalty

# ============================================================
# EXPLORATION CONFIGURATION
# ============================================================

EPSILON_START = 1
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.9997

# ============================================================
# ENHANCED NEURAL NETWORK CONFIGURATION
# ============================================================

DQN_HIDDEN_LAYERS = [128, 128, 64]

DUELING_FEATURE_DIM = 128
DUELING_VALUE_DIM = 64
DUELING_ADVANTAGE_DIM = 64

HIERARCHICAL_POLICY_HIDDEN = [128, 128]
HIERARCHICAL_VALUE_HIDDEN = [128, 128]
HIERARCHICAL_LOWER_HIDDEN = [64, 64]

# ============================================================
# OPTIMIZER CONFIGURATION
# ============================================================

LEARNING_RATE_DQN = 1e-3
LEARNING_RATE_PPO = 3e-4
LEARNING_RATE_EPRS = 1e-3
GAMMA_DQN = 0.95
GAMMA_PPO = 0.99
GAMMA_EPRS = 0.96
GAE_LAMBDA = 0.95
PPO_CLIP_EPSILON = 0.05
PPO_ENTROPY_COEFF = 0.01
PPO_VALUE_COEFF = 0.5

# ============================================================
# EPRS SPECIFIC
# ============================================================

PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_INCREMENT = 0.001
PER_EPSILON = 1e-6

# Add to config.py after existing configurations

# ============================================================
# PPO-LRT SPECIFIC CONFIGURATION
# From: "Optimization of Task-Scheduling Strategy in Edge 
#       Kubernetes Clusters Based on Deep Reinforcement Learning"
# ============================================================

# PPO hyperparameters (Section 4.2)
PPO_LRT_CLIP_EPSILON = 0.2        # ε: clipping parameter
PPO_LRT_GAMMA = 0.99              # γ: discount factor
PPO_LRT_LR_ACTOR = 0.0003         # Actor learning rate
PPO_LRT_LR_CRITIC = 0.001         # Critic learning rate
PPO_LRT_GAE_LAMBDA = 0.95         # λ: GAE parameter
PPO_LRT_UPDATE_STEPS = 30         # Policy update frequency (steps)
PPO_LRT_EPOCHS = 10               # PPO epochs per update
PPO_LRT_HIDDEN_DIM = 128          # Hidden layer dimension
PPO_LRT_ENTROPY_COEFF = 0.01      # Entropy coefficient

# Reward function weights (Equation 12)
# reward = α * (1/σ_c + 1/σ_m) - β * p - γ * R_i
PPO_LRT_ALPHA = 1.0               # α: load balance weight
PPO_LRT_BETA = 0.5                # β: penalty weight
PPO_LRT_GAMMA_RT = 0.3            # γ: response time weight

# Upper limits for penalty (Equation 14)
# p_c = Σ (c_i - l_c) for c_i > l_c
PPO_LRT_CPU_UPPER = 0.85          # CPU upper threshold
PPO_LRT_MEM_UPPER = 0.85          # Memory upper threshold

# ============================================================
# PLOTTING CONFIGURATION
# ============================================================

COLORS = {
    'PPO-LRT': '#E74C3C',
    'RLSK': '#3498DB',
    'EPRS': '#1ABC9C',
    'HS-DRL': '#F39C12',
    'HS-DRL_BEFORE_AFTER': '#F39C12'
}
