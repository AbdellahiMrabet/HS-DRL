# environment/k8s_env.py - Set unhealthy nodes to maximum utilization

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import subprocess
from collections import deque
from typing import List, Dict, Tuple
from config import NUM_NODES, MAX_STEPS, POD_ARRIVAL_RATE, CPU_LIMIT, MEM_LIMIT, RESPONSE_TIME_THRESHOLD
from environment.pod_manager import PodDeploymentManager


class K8s_HS(gym.Env):
    """Kubernetes scheduling environment with unhealthy nodes at max utilization"""
    
    def __init__(self):
        super().__init__()
        self.pod_manager = PodDeploymentManager()
        self.num_nodes = NUM_NODES
        
        # State: [cpu, mem] for each node + [pod_cpu, pod_mem] + [last_node, last_rt]
        state_dim = self.num_nodes * 2 + 4
        self.observation_space = spaces.Box(0, 1, (state_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_nodes + 1)
        
        self.pod_arrival_rate = POD_ARRIVAL_RATE
        self.pod_stack = deque()
        self.last_arrival_time = time.time()
        self.current_pod = None
        self.nodes = []
        self.response_time = {}
        
        # Track last deployed pod info
        self.last_deployed_node = -1
        self.last_response_time = 0
        
        self.reset()
    
    def _get_real_node_metrics(self) -> List[Dict]:
        """Get REAL node metrics from kubectl top, mark unavailable nodes as max"""
        try:
            result = subprocess.run(
                ["kubectl", "top", "nodes", "--no-headers"],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode != 0:
                return self._get_fallback_nodes()
            
            # First, get all nodes from kubectl get nodes to check readiness
            nodes_status = subprocess.run(
                ["kubectl", "get", "nodes", "-o", "jsonpath='{range .items[*]}{.metadata.name}{\" \"}{.status.conditions[?(@.type==\"Ready\")].status}{\"\\n\"}{end}'"],
                capture_output=True, text=True, timeout=5
            )
            
            # Parse node readiness
            node_ready = {}
            for line in nodes_status.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.strip().strip("'").split()
                    if len(parts) >= 2:
                        node_ready[parts[0]] = parts[1] == 'True'
            
            nodes = []
            parsed_nodes = set()
            
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                
                parts = line.split()
                if len(parts) >= 5:
                    node_name = parts[0]
                    parsed_nodes.add(node_name)
                    
                    # Check if node is ready
                    is_ready = node_ready.get(node_name, True)
                    
                    # Parse CPU percentage
                    cpu_str = parts[2].replace('%', '')
                    cpu_percent = float(cpu_str) / 100
                    
                    # Parse Memory percentage
                    mem_str = parts[4].replace('%', '')
                    mem_percent = float(mem_str) / 100
                    
                    # If node is NOT ready, set CPU and MEM to MAX (1.0) to discourage selection
                    if not is_ready:
                        print(f"  ⚠️ Node {node_name} is NOT READY - setting CPU/MEM to MAX (1.0)")
                        cpu_percent = CPU_LIMIT
                        mem_percent = MEM_LIMIT
                    
                    nodes.append({
                        'name': node_name,
                        'cpu': min(cpu_percent, 1.0),
                        'mem': min(mem_percent, 1.0),
                        'pods': 0,
                        'ready': is_ready
                    })
            
            # Check for nodes that exist but didn't appear in kubectl top (metrics unavailable)
            for node_name, is_ready in node_ready.items():
                if node_name not in parsed_nodes:
                    print(f"  ⚠️ Node {node_name} has NO METRICS - setting CPU/MEM to MAX (1.0)")
                    nodes.append({
                        'name': node_name,
                        'cpu': CPU_LIMIT,  # MAX to discourage selection
                        'mem': MEM_LIMIT,  # MAX to discourage selection
                        'pods': 0,
                        'ready': is_ready
                    })
            
            nodes.sort(key=lambda x: x['name'])
            return nodes
            
        except Exception as e:
            print(f"[ENV] Error getting metrics: {e}")
            return self._get_fallback_nodes()
    
    def _get_fallback_nodes(self) -> List[Dict]:
        """Fallback nodes - set all to reasonable values but mark as ready"""
        return [
            {'name': 'minikube', 'cpu': 0.06, 'mem': 0.28, 'pods': 0, 'ready': True},
            {'name': 'minikube-m02', 'cpu': 0.01, 'mem': 0.08, 'pods': 0, 'ready': True},
            {'name': 'minikube-m03', 'cpu': 0.03, 'mem': 0.18, 'pods': 0, 'ready': True},
            {'name': 'minikube-m04', 'cpu': 0.01, 'mem': 0.06, 'pods': 0, 'ready': True},
        ]
    
    def _update_node_metrics(self):
        """Refresh node metrics from real cluster"""
        self.nodes = self._get_real_node_metrics()
        
        # Pad with virtual nodes if needed (these are NOT ready and at MAX)
        while len(self.nodes) < self.num_nodes:
            print(f"  ⚠️ Adding virtual node (NOT READY) - setting CPU/MEM to MAX ({CPU_LIMIT}/{MEM_LIMIT})")
            self.nodes.append({
                'name': f'virtual-node-{len(self.nodes)}',
                'cpu': CPU_LIMIT,  # MAX to discourage selection
                'mem': MEM_LIMIT,  # MAX to discourage selection
                'pods': 0,
                'ready': False
            })
        
        # Initialize response times for new nodes
        for node in self.nodes:
            if node['name'] not in self.response_time:
                self.response_time[node['name']] = np.random.uniform(50, 150)
    
    def _check_nodes_ready(self) -> bool:
        """Check if all nodes are ready"""
        for node in self.nodes:
            if not node.get('ready', True):
                print(f"  ⚠️ Node {node['name']} is NOT READY")
                return False
        return True
    
    def _generate_poisson_arrivals(self):
        """Generate new pods using exponential inter-arrival times"""
        current_time = time.time()
        
        while True:
            next_arrival = self.last_arrival_time + np.random.exponential(1.0 / self.pod_arrival_rate)
            
            if next_arrival > current_time:
                break
            
            self.last_arrival_time = next_arrival
            
            patterns = [
                {'cpu': 0.05, 'mem': 64, 'desc': 'tiny-nginx'},
                {'cpu': 0.10, 'mem': 128, 'desc': 'small-nginx'},
                {'cpu': 0.15, 'mem': 256, 'desc': 'medium-nginx'},
                {'cpu': 0.20, 'mem': 512, 'desc': 'large-nginx'},
            ]
            pattern = np.random.choice(patterns)
            pod = {
                'cpu': pattern['cpu'],
                'mem': pattern['mem'],
                'desc': pattern['desc'],
                'arrival_time': next_arrival
            }
            self.pod_stack.append(pod)
    
    def _get_next_pod_from_stack(self):
        """Pop a pod from the stack and set as current_pod"""
        if len(self.pod_stack) > 0:
            self.current_pod = self.pod_stack.popleft()
            return True
        return False
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        self._update_node_metrics()
        
        # Reset last deployment info
        self.last_deployed_node = -1
        self.last_response_time = 0
        
        self.pod_manager.cleanup_all()
        self.pod_stack = deque()
        self.last_arrival_time = time.time()
        self._generate_poisson_arrivals()
        
        if not self._get_next_pod_from_stack():
            self.current_pod = None
        
        self.step_count = 0
        self.total_reward = 0
        
        # Print node status at reset
        ready_count = sum(1 for n in self.nodes if n.get('ready', True))
        print(f"[ENV] Reset: {ready_count}/{self.num_nodes} nodes ready")
        for idx, node in enumerate(self.nodes):
            status = "READY" if node.get('ready', True) else "NOT READY"
            print(f"  action {idx} -> {node['name']} ({status}) CPU={node['cpu']*100:.0f}% MEM={node['mem']*100:.0f}%")
        
        return self._get_obs(), {'nodes': [n['name'] for n in self.nodes]}
    
    def _get_obs(self):
        """Get observation including last deployment response time"""
        obs = []
        
        # Node metrics (unhealthy nodes already have CPU/MEM = 1.0)
        for node in self.nodes:
            obs.extend([node['cpu'], node['mem']])
        
        # Current pod request
        if self.current_pod:
            obs.extend([self.current_pod['cpu'], self.current_pod['mem'] / 1000])
        else:
            obs.extend([0.0, 0.0])
        
        # Last deployed node (normalized) and its response time (normalized)
        normalized_node = self.last_deployed_node / self.num_nodes if self.last_deployed_node >= 0 else 0
        normalized_rt = min(self.last_response_time / 500, 1.0)
        obs.extend([normalized_node, normalized_rt])
        
        return np.array(obs, dtype=np.float32)
    
    def _is_node_usable(self, node: Dict) -> bool:
        """Check if node is usable (ready AND not at max utilization)"""
        if not node.get('ready', True):
            return False
        # If CPU or MEM is at max (1.0), node is effectively unusable
        if node['cpu'] >= 0.99 or node['mem'] >= 0.99:
            return False
        return True
    
    def step(self, action):
        # If no pod to deploy, try to get one
        if self.current_pod is None:
            self._generate_poisson_arrivals()
            if not self._get_next_pod_from_stack():
                truncated = self.step_count >= MAX_STEPS
                return self._get_obs(), 0.0, False, truncated, {
                    'success': False, 'pod': 'none', 'api_response_time': 0, 'stack_size': len(self.pod_stack)
                }
        
        self.step_count += 1
        reward = 0
        success = False
        api_response_time = 0
        
        self._generate_poisson_arrivals()
        self.pod_manager.cleanup_old_pods()
        
        # Track if deployment was attempted
        deployment_attempted = False
        target_node_idx = -1
        
        if action < len(self.nodes):
            target_node = self.nodes[action]
            node_name = target_node['name']
            target_node_idx = action
            
            if node_name not in self.response_time:
                self.response_time[node_name] = np.random.uniform(50, 150)
            
            # Check if node is usable
            if not self._is_node_usable(target_node):
                print(f"  [ENV] Node {node_name} is NOT USABLE (ready={target_node.get('ready')}, cpu={target_node['cpu']:.2f})")
                reward = -10.0  # Higher penalty for unusable nodes
                self.pod_stack.append(self.current_pod)
            else:
                new_cpu = target_node['cpu'] + self.current_pod['cpu']
                new_mem = target_node['mem'] + self.current_pod['mem'] / 1000
                
                if (new_cpu <= CPU_LIMIT and new_mem <= MEM_LIMIT and 
                    self.response_time[node_name] < RESPONSE_TIME_THRESHOLD):
                    
                    deployed, _, api_response_time = self.pod_manager.deploy_nginx_pod(
                        node_name=node_name,
                        cpu_request=self.current_pod['cpu'],
                        mem_request=self.current_pod['mem']
                    )
                    
                    if deployed:
                        deployment_attempted = True
                        self.response_time[node_name] = api_response_time
                        success = True
                        target_node['cpu'] = new_cpu
                        target_node['mem'] = new_mem
                        target_node['pods'] = target_node.get('pods', 0) + 1
                        
                        # Update last deployment info
                        self.last_deployed_node = target_node_idx
                        self.last_response_time = api_response_time
                        
                        reward = 10.0 + (target_node['cpu'] * 5)
                        loads = [n['cpu'] for n in self.nodes if self._is_node_usable(n)]
                        if loads:
                            reward -= np.std(loads) * 3
                    else:
                        reward = -4.0
                        self.pod_stack.append(self.current_pod)
                else:
                    reward = -3.0
                    self.pod_stack.append(self.current_pod)
        else:
            # Delay action
            reward = -0.40
            self.pod_stack.append(self.current_pod)
        
        # Get next pod
        pod_available = self._get_next_pod_from_stack()
        if not pod_available:
            self.current_pod = None
        
        self.total_reward += reward
        self._update_node_metrics()
        
        truncated = self.step_count >= MAX_STEPS
        done = False
        
        return self._get_obs(), reward, done, truncated, {
            'success': success,
            'pod': self.current_pod['desc'] if self.current_pod else 'none',
            'api_response_time': api_response_time,
            'stack_size': len(self.pod_stack),
            'target_node': target_node_idx if action < len(self.nodes) else -1,
            'last_response_time': self.last_response_time
        }
    
    def close(self):
        self.pod_manager.cleanup_all()