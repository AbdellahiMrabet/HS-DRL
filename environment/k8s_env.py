# environment/k8s_env.py - Kubernetes Environment WITH Safety Projection for HS-DRL

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import subprocess
import copy
from collections import deque
from typing import List, Dict, Tuple, Optional
from config import (NUM_NODES, MAX_STEPS, POD_ARRIVAL_RATE, CPU_LIMIT, MEM_LIMIT,
                    RESPONSE_TIME_THRESHOLD_FULL, REWARD_BASE, REWARD_EFFICIENCY_FACTOR,
                    REWARD_LOAD_PENALTY_FACTOR, REWARD_FAILURE_PENALTY, REWARD_CONSTRAINT_PENALTY,
                    REWARD_DELAY_PENALTY, REWARD_NODE_NOT_READY_PENALTY, PATTERNS, generate_pod_ttl)
from environment.pod_manager import PodDeploymentManager
from utils.metrics_tracker import MetricsTracker


class K8sEnv(gym.Env):
    """
    Kubernetes Environment WITH Safety Projection for HS-DRL.
    
    Features:
    - Full response time measurement (decision + API)
    - Available nodes tracking for scalability
    - Per-node LATEST POD DEPLOYMENT RESPONSE TIME in observation space
    - Safety constraints enforced BEFORE action execution (Hierarchical Safety Projection)
    - Multi-level fallback: Validate -> Project -> Strategic Delay
    - Unique TTL for every pod
    
    Node Metrics Structure (from 'kubectl describe node' Allocated resources):
        - 'cpu': current CPU requests in cores (from Requests column)
        - 'mem': current memory requests in MiB (from Requests column)
        - 'cpu_percent': current CPU utilization ratio (extracted from parentheses)
        - 'mem_percent': current memory utilization ratio (extracted from parentheses)
    """
    
    def __init__(self, tracker: Optional[MetricsTracker] = None):
        super().__init__()
        self.pod_manager = PodDeploymentManager()
        self.num_nodes = NUM_NODES
        self.tracker = tracker
        
        # Observation space: [node0_cpu_percent, node0_mem_percent, node0_rt, ... pod_cpu, pod_mem]
        state_dim = self.num_nodes * 3 + 2 + 1
        self.observation_space = spaces.Box(0, 1, (state_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_nodes)
        
        # Poisson arrival
        self.pod_arrival_rate = POD_ARRIVAL_RATE
        self.pod_stack = deque()
        self.last_arrival_time = time.time()
        self.current_pod = None
        self.nodes = []
        
        # Per-node LATEST RESPONSE TIME (ms) of last deployed pod
        self.latest_response_time = {}
        self.latest_response_time_normalized = {}
        
        self.node_response_history = {}
        self.max_response_time_ms = 500
        
        self.available_nodes_history = []
        
        # Safety projection tracking
        self.projection_active = False
        self.projection_count = 0
        self.delay_count = 0
        
        self.reset()
        self.pod_manager.cleanup_all()
        
    
    # ========================================================================
    # Helper Methods
    # ========================================================================
    
    def _normalize_response_time(self, rt_ms: float) -> float:
        """Normalize response time to 0-1 range (1 = 500ms or more)"""
        return min(rt_ms / self.max_response_time_ms, 1.0)
    
    def _get_node_avg_response_time(self, node_name:str) ->float:
        if node_name in self.node_response_history and len(self.node_response_history[node_name]):
            return np.mean(self.node_response_history[node_name])
        return 0.0
    
    def _get_node_response_time_std(self, node_name:str) ->float:
        if node_name in self.node_response_history and len(self.node_response_history[node_name]) > 1:
            return np.std(self.node_response_history[node_name])
        return 0.0
    
    def _parse_k8s_resource(self, value: str) -> float:
        """Parse Kubernetes resource string to float value."""
        if not value:
            return 0.0
        value = value.strip()
        if value.endswith('m'):
            return float(value.replace('m', '')) / 1000.0
        if value.endswith('Ki'):
            return float(value.replace('Ki', '')) / 1024.0
        elif value.endswith('Mi'):
            return float(value.replace('Mi', ''))
        elif value.endswith('Gi'):
            return float(value.replace('Gi', '')) * 1024.0
        elif value.endswith('Ti'):
            return float(value.replace('Ti', '')) * 1024.0 * 1024.0
        try:
            return float(value)
        except ValueError:
            return 0.0
    
    def _extract_percentage(self, value: str) -> float:
        """Extract percentage value from string like '(7%)' or '7%'."""
        if not value:
            return 0.0
        cleaned = value.strip().replace('(', '').replace(')', '').replace('%', '')
        try:
            return float(cleaned) / 100.0
        except ValueError:
            return 0.0
    
    def _get_node_conditions(self, conditions_text: str) -> Dict[str, bool]:
        """Parse node conditions from 'kubectl describe node' output"""
        conditions = {
            'Ready': False,
            'MemoryPressure': False,
            'DiskPressure': False,
            'PIDPressure': False,
            'NetworkUnavailable': False
        }
        for line in conditions_text.split('\n'):
            line = line.strip()
            for cond in conditions.keys():
                if cond in line:
                    conditions[cond] = 'True' in line and 'False' not in line
                    break
        return conditions
    
    def _get_real_node_metrics(self) -> List[Dict]:
        """Get real node metrics from 'kubectl describe node' - Allocated resources only"""
        try:
            result = subprocess.run(
                ["kubectl", "get", "nodes", "--no-headers"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return self._get_fallback_nodes()
            
            node_names = [line.split()[0] for line in result.stdout.strip().split('\n') if line.strip()]
            nodes = []
            
            for node_name in node_names[:self.num_nodes]:
                desc_result = subprocess.run(
                    ["kubectl", "describe", "node", node_name],
                    capture_output=True, text=True, timeout=10
                )
                if desc_result.returncode != 0:
                    continue
                
                output = desc_result.stdout
                cpu_requests = 0.0
                mem_requests = 0.0
                cpu_percent = 0.0
                mem_percent = 0.0
                conditions_text = ""
                in_conditions = False
                in_allocated_section = False
                
                for line in output.split('\n'):
                    line_stripped = line.strip()
                    
                    if 'Conditions:' in line_stripped:
                        in_conditions = True
                        in_allocated_section = False
                        continue
                    elif in_conditions:
                        if line_stripped and not line[0].isspace():
                            in_conditions = False
                        else:
                            conditions_text += line_stripped + '\n'
                        continue
                    
                    if 'Allocated resources:' in line_stripped:
                        in_allocated_section = True
                        continue
                    
                    if in_allocated_section:
                        if line_stripped.startswith('cpu'):
                            parts = line_stripped.split()
                            if len(parts) >= 3:
                                cpu_requests = self._parse_k8s_resource(parts[1])
                                cpu_percent = self._extract_percentage(parts[2])
                        elif line_stripped.startswith('memory'):
                            parts = line_stripped.split()
                            if len(parts) >= 3:
                                mem_requests = self._parse_k8s_resource(parts[1])
                                mem_percent = self._extract_percentage(parts[2])
                        elif line_stripped.startswith('Events:'):
                            in_allocated_section = False
                
                conditions = self._get_node_conditions(conditions_text)
                
                nodes.append({
                    'name': node_name,
                    'cpu': cpu_requests,
                    'mem': mem_requests,
                    'cpu_percent': min(cpu_percent, 1.0),
                    'mem_percent': min(mem_percent, 1.0),
                    'pods': 0,
                    'ready': conditions.get('Ready', True)
                })
            
            return sorted(nodes, key=lambda x: x['name'])
            
        except Exception as e:
            print(f"  ⚠️ Error getting node metrics: {e}")
            return self._get_fallback_nodes()
    
    def _get_fallback_nodes(self) -> List[Dict]:
        """Fallback nodes with realistic values"""
        fallback = [
            {'name': 'minikube-m02', 'cpu': 0.4, 'mem': 512.0,
             'cpu_percent': 0.05, 'mem_percent': 0.13, 'pods': 0, 'ready': True},
            {'name': 'minikube-m03', 'cpu': 0.8, 'mem': 1024.0,
             'cpu_percent': 0.10, 'mem_percent': 0.26, 'pods': 0, 'ready': True},
            {'name': 'minikube-m04', 'cpu': 0.3, 'mem': 384.0,
             'cpu_percent': 0.04, 'mem_percent': 0.10, 'pods': 0, 'ready': True},
        ]
        return sorted(fallback, key=lambda x: x['name'])
    
    def _update_node_metrics(self):
        """Refresh node metrics from kubectl describe"""
        new_nodes = self._get_real_node_metrics()
        #print('len new nodes', len(new_nodes))
        while len(new_nodes) < self.num_nodes:
            new_nodes.append({
                'name': f'virtual-node-{len(new_nodes)}',
                'cpu': 0.0, 'mem': 0.0,
                'cpu_percent': 0.0, 'mem_percent': 0.0,
                'pods': 0, 'ready': False
            })
        self.nodes = new_nodes
        for node in self.nodes:
            node_name = node['name']
            if node_name not in self.latest_response_time:
                self.latest_response_time[node_name] = 15.0
                self.latest_response_time_normalized[node_name] = self._normalize_response_time(15.0)
            if node_name not in self.node_response_history:
                self.node_response_history[node_name] = []
    
    def _get_available_nodes_count(self) -> int:
        return len([n for n in self.nodes if n.get('ready', True)])
    
    def _get_available_nodes_names(self) -> List[str]:
        return [n['name'] for n in self.nodes if n.get('ready', True)]
    
    def _detect_node_changes(self, old_nodes: List[Dict], new_nodes: List[Dict]) -> List[Dict]:
        changes = []
        old_status = {n['name']: n.get('ready', True) for n in old_nodes}
        new_status = {n['name']: n.get('ready', True) for n in new_nodes}
        for name in set(old_status.keys()) | set(new_status.keys()):
            old = old_status.get(name, False)
            new = new_status.get(name, False)
            if old != new:
                changes.append({'node': name, 'old_status': old, 'new_status': new, 'timestamp': time.time()})
        return changes
    
    def _generate_poisson_arrivals(self):
        current_time = time.time()
        time_elapsed = max(0, current_time - self.last_arrival_time)
        lambda_val = self.pod_arrival_rate * time_elapsed
        if lambda_val < 0 or np.isnan(lambda_val):
            lambda_val = 0
        num_arrivals = np.random.poisson(lambda_val) if lambda_val > 0 else 0
        if num_arrivals > 0:
            self.last_arrival_time = current_time
            for _ in range(min(num_arrivals, 50)):
                
                pattern = np.random.choice(PATTERNS)
                pod_ttl = generate_pod_ttl()
                self.pod_stack.append({
                    'cpu': pattern['cpu'],
                    'mem': pattern['mem'],
                    'desc': pattern['desc'],
                    'ttl': pod_ttl
                })
    
    def _get_next_pod(self):
        if self.pod_stack:
            self.current_pod = self.pod_stack.popleft()
            return True
        return False
    
    def _calculate_imbalance(self) -> float:
        loads = [n['cpu_percent'] for n in self.nodes if n.get('ready', True)]
        return np.std(loads) if loads else 0.0
    
    # ========================================================================
    # Gymnasium Environment Methods
    # ========================================================================
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self._update_node_metrics()
        #self.pod_manager.cleanup_all()
        self.pod_manager.cleanup_old_pods()
        self.pod_stack = deque()
        self.last_arrival_time = time.time()
        self._generate_poisson_arrivals()
        
        if not self._get_next_pod():
            self.current_pod = None
        
        self.step_count = 0
        self.total_reward = 0
        self.available_nodes_history = []
        self.latest_response_time = {n['name']: 15.0 for n in self.nodes}
        self.latest_response_time_normalized = \
            {n['name']: self._normalize_response_time(15.0) for n in self.nodes}
        self.node_response_history = {n['name']:[] for n in self.nodes}
        # Safety projection tracking
        self.projection_active = False
        self.projection_count = 0
        self.delay_count = 0
        
        if self.tracker:
            self.tracker.end_episode()
        
        return self._get_obs(), {
            'available_nodes': self._get_available_nodes_count(),
            'available_nodes_list': self._get_available_nodes_names(),
            'latest_response_times': self.latest_response_time.copy(),
            'node_response_history': {k: v.copy() for k, v in self.node_response_history.items()}
        }
    
    def _get_obs(self):
        obs = []
        for node in self.nodes:
            node_name = node['name']
            obs.append(node['cpu_percent'])
            obs.append(node['mem_percent'])
            rt_normalized = self.latest_response_time_normalized.get(node_name, 0.2)
            obs.append(rt_normalized)
        
        if self.current_pod:
            obs.append(self.current_pod['cpu'])
            obs.append(self.current_pod['mem'] / 1000.0)
        else:
            obs.append(0.0)
            obs.append(0.0)
        obs.append(1.0 if self.projection_active else 0.0)  # Use the flag
        
        return np.array(obs, dtype=np.float32)
    
    def _get_response_time_bonus(self, response_time_ms: float) -> float:
        rt = (30 - response_time_ms) / 50.0
        rt = np.clip(rt, -0.4, 0.4)
        
        return rt
        
    def _get_load_bonus(self, load: float) -> float:
        # LD_EXCELLENT,      #+2
        # LD_VERY_GOOD,      #+1      
        # LD_GOOD,           #+0.5
        # LD_VEAVY,          #-1
        # LD_VERY_HEAVY,     #-2
        # LD_OVERLOADED,     #-3
        return 0.5 - load
    
    def _update_latest_response_time(self, node_name: str, new_rt_ms: float):
        self.latest_response_time[node_name] = new_rt_ms
        self.latest_response_time_normalized[node_name] = self._normalize_response_time(new_rt_ms)
        print(f"  📊 Node {node_name} latest response time updated to {new_rt_ms:.1f}ms")
        
        if node_name not in self.node_response_history:
            self.node_response_history[node_name] = []
            
        if len(self.node_response_history[node_name]) > 100:
            self.node_response_history[node_name] = \
                self.node_response_history[node_name][-100:]
    
    # ========================================================================
    # HIERARCHICAL SAFETY PROJECTION (HSP) - Key Differentiator
    # ========================================================================

    def _safe_area(self):
        """
        Level 1: Find the nearest safe node for projection.
        Returns safe node array if a configuration exists, None otherwise.
        """
        # 1. First Pass: Check all nodes for the original current_pod
        safe_nodes = []
        if self.current_pod is not None:
            for i, node in enumerate(self.nodes):
                if not node.get('ready', True):
                    continue
                
                new_cpu = node['cpu'] + self.current_pod['cpu']
                new_mem = node['mem'] + self.current_pod['mem']
                
                if node['cpu'] > 0:
                    projected_cpu = node['cpu_percent'] * (new_cpu / node['cpu'])
                else:
                    projected_cpu = 1.0 if new_cpu > 0 else 0.0
                
                if node['mem'] > 0:
                    projected_mem = node['mem_percent'] * (new_mem / node['mem'])
                else:
                    projected_mem = 1.0 if new_mem > 0 else 0.0
                
                if (max(0.0, projected_cpu - CPU_LIMIT) + max(0.0, projected_mem - MEM_LIMIT)) <= 0.01:
                    safe_nodes.append(i)

        # If the current pod found safe nodes, we exit and return them immediately
        if safe_nodes:
            return np.array(safe_nodes)

        # 2. Second Pass: If current_pod failed, check the remaining pods in the queue
        if hasattr(self, 'pod_stack') and self.pod_stack:
            # We search sequentially through the queue
            for stack_idx, alternative_pod in enumerate(self.pod_stack):
                fallback_safe_nodes = []
                
                for i, node in enumerate(self.nodes):
                    if not node.get('ready', True):
                        continue
                    
                    new_cpu = node['cpu'] + alternative_pod['cpu']
                    new_mem = node['mem'] + alternative_pod['mem']
                    
                    if node['cpu'] > 0:
                        projected_cpu = node['cpu_percent'] * (new_cpu / node['cpu'])
                    else:
                        projected_cpu = 1.0 if new_cpu > 0 else 0.0
                    
                    if node['mem'] > 0:
                        projected_mem = node['mem_percent'] * (new_mem / node['mem'])
                    else:
                        projected_mem = 1.0 if new_mem > 0 else 0.0
                    
                    if (max(0.0, projected_cpu - CPU_LIMIT) + max(0.0, projected_mem - MEM_LIMIT)) <= 0.01:
                        fallback_safe_nodes.append(i)
                
                # The moment an alternative pod matches a safe cluster node group:
                if fallback_safe_nodes:
                    # Remove it from its deep position in the queue
                    del self.pod_stack[stack_idx]
                    
                    # Push the un-schedulable old current_pod to the very front of the deque
                    if self.current_pod is not None:
                        self.pod_stack.appendleft(self.current_pod)
                    
                    # Set the promoted pod as the new active current_pod
                    self.current_pod = alternative_pod
                    return np.array(fallback_safe_nodes)

        # 3. Ultimate Fallback: No pods fit anywhere
        return None
    
    def _calcutate_projected_utilization(self, node: Dict, pod: Dict) -> Tuple[float, float]:
        new_cpu = node['cpu'] + pod['cpu']
        new_mem = node['mem'] + pod['mem']
        
        if node['cpu'] > 0:
            projected_cpu = node['cpu_percent'] * (new_cpu / node['cpu'])
        else:
            projected_cpu = 1.0 if new_cpu > 0 else 0.0
        
        if node['mem'] > 0:
            projected_mem = node['mem_percent'] * (new_mem / node['mem'])
        else:
            projected_mem = 1.0 if new_mem > 0 else 0.0
        
        return projected_cpu, projected_mem
    
    def _is_action_safe(self, action: int) -> Tuple[bool, str]:
        """
        Level 0: Validate if raw action is safe.
        
        Checks:
        1. Action within bounds
        2. Node readiness
        3. CPU limit (projected)
        4. MEM limit (projected)
        5. Response time threshold
        """
        if action > len(self.nodes):
            return False, "action_out_of_range"
        elif action == self.num_nodes:
            return True, "strategic_delay"
        
        response_time_threshold_exceeded = ""
        cpu_limit_exceeded = ""
        mem_limit_exceeded = ""
        reason = ""
            
        projection = False
        
        if action <= len(self.nodes):
            target_node = self.nodes[action]
            node_name = target_node['name']
            
            if not target_node.get('ready', True):
                return False, f"node_{node_name}_not_ready"
            
            # Calculate projected utilization
            projected_cpu_util, projected_mem_util = \
                self._calcutate_projected_utilization(target_node, \
                    self.current_pod if self.current_pod else {'cpu': 0, 'mem': 0})
                
            latest_rt = self.latest_response_time.get(node_name, 0)
            if latest_rt >= RESPONSE_TIME_THRESHOLD_FULL:
                projection = True
                response_time_threshold_exceeded = f"response_time_threshold_exceeded:{latest_rt}"
                reason = response_time_threshold_exceeded
                
            if max(0.0, projected_cpu_util - CPU_LIMIT) + max(0.0, projected_mem_util - MEM_LIMIT) > 0.01:
                
                # The current_pod fails constraints. Let's find an alternative pod in the stack.
                found_alternative = False
                
                for stack_idx, alternative_pod in enumerate(self.pod_stack):
                    alt_projected_cpu, alt_projected_mem = \
                        self._calcutate_projected_utilization(target_node, alternative_pod)
                                        
                    # Check if this alternative pod satisfies safety bounds
                    if (max(0.0, alt_projected_cpu - CPU_LIMIT) + max(0.0, alt_projected_mem - MEM_LIMIT)) <= 0.01:
                        # SUCCESS: We found a matching pod. 
                        # 1. Remove it from its position in the deque
                        del self.pod_stack[stack_idx]
                        
                        # 2. Push the blocked current_pod to the front of the queue so it is not lost
                        if self.current_pod is not None:
                            self.pod_stack.appendleft(self.current_pod)
                        
                        # 3. Promote the alternative pod to be the active current_pod
                        self.current_pod = alternative_pod
                        
                        found_alternative = True
                        break  # Exit the loop since we committed the state change
                        
                # If we looped through the entire stack and nobody fit, it's a hard safety violation
                if not found_alternative:
                    projection = True
                        
            if projected_cpu_util > CPU_LIMIT:
                cpu_limit_exceeded = f"cpu_limit_exceeded:{projected_cpu_util}"
                reason = cpu_limit_exceeded
            
            if projected_mem_util > MEM_LIMIT:
                mem_limit_exceeded = f"mem_limit_exceeded:{projected_mem_util}"
                reason = mem_limit_exceeded
                
            if projection:
                
                return False, reason
            return True, "safe"
    
    def _hierarchical_safety_projection(self, raw_action: int) -> Tuple[int, bool]:
        """
        Hierarchical Safety Projection (HSP) with Multi-Level Fallback.
        
        Level 0: Validate raw action
        Level 1: Project to nearest safe node
        Level 2: Strategic Delay (return delay action)
        
        Returns: (safe_action, projection_triggered)
        """
        # Level 0: Check if raw action is safe
        is_safe, reason = self._is_action_safe(raw_action)
        if is_safe:
            return raw_action, False
        
        print(f"  🔄 Safety violation: {reason} - attempting projection")
        
        # Level 1: Find safe projection
        safe_node = self._safe_area()
        if safe_node is not None:
            if len(safe_node) > 1:
                safe_action = safe_node[np.random.randint(len(safe_node) - 1)]
            else:
                safe_action = safe_node[0]
            self.projection_count += 1
            print(f"  🎯 Projecting action {raw_action} -> {safe_action}")
            return safe_action, True
        
        # Level 2: Strategic Delay
        self.delay_count += 1
        print(f"  ⏸️ No safe node available - Strategic Delay")
        return self.num_nodes, True
    
    def get_per_node_response_stats(self) -> Dict:
        stats = {}
        for node_name in self.node_response_history.keys():
            if self.node_response_history[node_name]:
                stats[node_name] = {
                    'latest': self.latest_response_time.get(node_name, 0),
                    'avg': np.mean(self.node_response_history[node_name]),
                    'min': np.min(self.node_response_history[node_name]),
                    'max': np.max(self.node_response_history[node_name]),
                    'std': np.std(self.node_response_history[node_name]) \
                        if len(self.node_response_history[node_name]) > 1 else 0,
                    'count': len(self.node_response_history[node_name])
                }
            else:
                stats[node_name] = {
                    'latest': 100.0,
                    'avg': 100.0,
                    'min': 100.0,
                    'max': 100.0,
                    'std': 0.0,
                    'count': 0
                }
    
    def _find_best_node(self) -> int:
        """
        Find the best node according to the HS-DRL reward function.
        
        Reward calculation matches the step() method:
            reward = REWARD_BASE 
                    + (mem_percent * REWARD_EFFICIENCY_FACTOR)
                    + get_load_bonus(mem_percent)
                    + get_response_time_bonus(rt)
                    - std(current_cpu_loads) * REWARD_LOAD_PENALTY_FACTOR
        
        Returns:
            Index of the best node (or self.num_nodes for delay if no node suitable)
        """
        if self.current_pod is None:
            return self.num_nodes  # Delay action
        
        pod_cpu = self.current_pod['cpu']
        pod_mem = self.current_pod['mem']
        
        best_action = self.num_nodes # Default to delay
        best_reward = -float('inf')
        
        # Calculate current cluster load imbalance (same for all nodes)
        loads_cpu = [n['cpu_percent'] \
                                for n in self.nodes if n.get('ready', True)]
        loads_mem = [n['mem_percent'] \
                                for n in self.nodes if n.get('ready', True)]
        current_imbalance = 0.0
        if loads_cpu:
            current_imbalance += np.std(loads_cpu)
        if loads_mem:
            current_imbalance += np.std(loads_mem)
        
        for action, node in enumerate(self.nodes):
            # Skip nodes that are not ready
            if not node.get('ready', True):
                continue
            
            node_name = node['name']
            
            # Calculate projected memory utilization after deployment
            new_mem = node['mem'] + pod_mem
            new_cpu = node['cpu'] + pod_cpu
            
            if node['mem'] > 0:
                projected_mem_util = node['mem_percent'] * (new_mem / node['mem'])
            else:
                projected_mem_util = 1.0
            
            projected_mem_util = min(projected_mem_util, 1.0)
            
            if node['cpu'] > 0:
                projected_cpu_util = node['cpu_percent'] * (new_cpu / node['cpu'])
            else:
                projected_cpu_util = 1.0
            
            # Calculate reward components
            reward = REWARD_BASE
            bonus = self._get_load_bonus(projected_mem_util) \
                    + self._get_load_bonus(projected_cpu_util) \
                    + self._get_response_time_bonus(self.latest_response_time.get(node_name, 0))
            reward += bonus * REWARD_EFFICIENCY_FACTOR
            
            # Load imbalance penalty (using CURRENT loads, NOT simulated)
            reward -= current_imbalance * REWARD_LOAD_PENALTY_FACTOR
            
            if reward > best_reward:
                best_reward = reward
                best_action = action
        
        return best_action
    
    # ========================================================================
    # Step Function
    # ========================================================================
    
    def step(self, action):
        """
        Execute action WITH Hierarchical Safety Projection.
        
        1. Apply HSP to get safe action
        2. Deploy pod (if not delay)
        3. Measure latencies
        4. Calculate reward with projection penalty
        5. Update metrics tracker
        """
        decision_start = time.perf_counter()
        
        # Get next pod if needed
        if self.current_pod is None:
            self._generate_poisson_arrivals()
            if not self._get_next_pod():
                truncated = self.step_count >= MAX_STEPS
                return self._get_obs(), 0.0, False, truncated, {
                    'success': False,
                    'available_nodes': self._get_available_nodes_count(),
                    'available_nodes_list': self._get_available_nodes_names()
                }
        
        self.step_count += 1
        reward = 0.0
        success = False
        api_response_time = 0.0
        self.pod_manager.cleanup_old_pods()
        done = False
        
        old_nodes = copy.deepcopy(self.nodes)
        
        # ===== HIERARCHICAL SAFETY PROJECTION =====
        safe_action, projection_triggered = self._hierarchical_safety_projection(action)
        self.projection_active = projection_triggered
        
        is_delay_action = (safe_action == self.num_nodes)
        
        if is_delay_action:
            queue_penalty = -0.1 * len(self.pod_stack) 
            queue_penalty = np.clip(queue_penalty, -2.0, 0.0)
            success = False
            reward = REWARD_DELAY_PENALTY + queue_penalty
            self.pod_stack.append(self.current_pod)
            print(f"  ⏸️ Delay action - penalty {reward:.2f} (Queue: {len(self.pod_stack)})")
            
        else:
            target_node = self.nodes[safe_action]
            node_name = target_node['name']
            
            if not target_node.get('ready', True):
                reward = REWARD_NODE_NOT_READY_PENALTY
                self.pod_stack.append(self.current_pod)
                print(f"  ⚠️ Node {node_name} not ready - penalty {reward}")
            else:
                api_start = time.perf_counter()
                # Check if chosen action is optimal
                is_optimal = (action == self._find_best_node() \
                    and action == safe_action)
                                    
                done = is_optimal
                print(f'expected action: {self._find_best_node()}, '
                    f'actual action: {safe_action}, done {done}')
                deployed, _, api_response_time = self.pod_manager.deploy_nginx_pod(
                    node_name=node_name,
                    cpu_request=self.current_pod['cpu'],
                    mem_request=float(self.current_pod['mem']),
                    custom_ttl=self.current_pod.get('ttl')
                )
                
                api_response_time = (time.perf_counter() - api_start) * 1000
                
                if deployed:
                    success = True
                    self._update_latest_response_time(node_name, api_response_time)
                    self._update_node_metrics()
                                        
                    updated_node = next((n for n in self.nodes if n['name'] == node_name), target_node)
                    loads_cpu = [n['cpu_percent'] \
                                for n in self.nodes if n.get('ready', True)]
                    loads_mem = [n['mem_percent'] \
                                for n in self.nodes if n.get('ready', True)]
                    if not projection_triggered:
                            # Calculate reward for the SELECTED node consistently
                            reward = REWARD_BASE
                            bonus = self._get_load_bonus(updated_node['mem_percent']) \
                                + self._get_load_bonus(updated_node['cpu_percent']) \
                                    + self._get_response_time_bonus(api_response_time)
                            reward += bonus * REWARD_EFFICIENCY_FACTOR
                            
                            # Calculate load imbalance penalty
                            
                            if loads_cpu:
                                reward -= np.std(loads_cpu) * REWARD_LOAD_PENALTY_FACTOR
                                
                            if loads_mem:
                                reward -= np.std(loads_mem) * REWARD_LOAD_PENALTY_FACTOR
                                reward = np.clip(reward, -10.0, 10.0)
                        
                    else:
                        _, reason = self._is_action_safe(action)
                        reward = REWARD_CONSTRAINT_PENALTY
                        
                        bonus = 0.0
                            
                        if "cpu_limit_exceeded" in reason:
                            bonus += self._get_load_bonus(float(reason.split(":")[1]))
                        elif "mem_limit_exceeded" in reason:
                            bonus += self._get_load_bonus(float(reason.split(":")[1]))
                        elif "response_time_threshold_exceeded" in reason:
                            bonus += self._get_response_time_bonus(float(reason.split(":")[1]))
                        else:
                            bonus += 0.0
                            
                        reward += bonus * REWARD_EFFICIENCY_FACTOR
                        reward = np.clip(reward, -10.0, 10.0)
                        
                    print(f"  ✅ Deployed to {node_name} | API: {api_response_time:.1f}ms | "
                          f"Memory Utilization: {target_node['mem_percent'] * 100:.2f} % | "
                          f"Projected: {projection_triggered} | Reward: {reward:.2f}")
                else:
                    reward = REWARD_FAILURE_PENALTY
                    self.pod_stack.append(self.current_pod)
                    print(f"  ❌ Deployment failed to {node_name} - penalty {reward}")
        
        # Get next pod
        if not self._get_next_pod():
            self.current_pod = None
        
        full_response_time = (time.perf_counter() - decision_start) * 1000
        
        if not success:
            self._update_node_metrics()
        
        node_changes = self._detect_node_changes(old_nodes, self.nodes)
        available_count = self._get_available_nodes_count()
        
        self.available_nodes_history.append({
            'step': self.step_count,
            'available_nodes': available_count,
            'timestamp': time.time()
        })
        
        for change in node_changes:
            status_str = "READY" if change['new_status'] else "NOT READY"
            print(f"  🔄 Node {change['node']} changed to {status_str}")
        
        self.total_reward += reward
        truncated = self.step_count >= MAX_STEPS
        
        # Track metrics
        if self.tracker:
            avg_util = np.mean([n['cpu_percent'] for n in self.nodes]) if self.nodes else 0.0
            imbalance = np.std([n['cpu_percent'] for n in self.nodes]) if self.nodes else 0.0
            
            self.tracker.step(
                reward=reward,
                success=success,
                util=avg_util,
                disk_io=0.0,
                imbalance=imbalance,
                full_response_time=full_response_time,
                api_response_time=api_response_time,
                nodes_data=self.nodes,
                latest_response_time=self.latest_response_time,
                projection=projection_triggered,
                arrival_count=len(self.pod_stack),
                available_nodes=available_count
            )
        
        return self._get_obs(), reward, done, truncated, {
            'success': success,
            'api_response_time': api_response_time,
            'full_response_time': full_response_time,
            'available_nodes': available_count,
            'available_nodes_list': self._get_available_nodes_names(),
            'node_changes': node_changes,
            'stack_size': len(self.pod_stack),
            'latest_response_times': self.latest_response_time.copy(),
            'projection_triggered': projection_triggered,
            'is_delay': is_delay_action,
            'update_latest_response_time': self._update_latest_response_time
        }
    
    def close(self):
        """Clean up resources"""
        self.pod_manager.cleanup_all()
        if self.tracker:
            self.tracker.end_episode()
    
    def get_safety_stats(self) -> Dict:
        """Return safety projection statistics"""
        return {
            'projection_count': self.projection_count,
            'delay_count': self.delay_count,
            'total_steps': self.step_count
        }
