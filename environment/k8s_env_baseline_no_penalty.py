# environment/k8s_env_baseline.py - Kubernetes Environment WITHOUT Safety Projection for Baselines (DRS, RLSK, EPRS)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import subprocess
import copy
from collections import deque
from typing import List, Dict, Tuple, Optional
from config import (NUM_NODES, MAX_STEPS, POD_ARRIVAL_RATE, CPU_LIMIT, MEM_LIMIT, PPO_LRT_ALPHA,
                    PPO_LRT_BETA, PPO_LRT_CPU_UPPER, PPO_LRT_GAMMA_RT, PPO_LRT_MEM_UPPER,
                    REWARD_DELAY_PENALTY, REWARD_NODE_NOT_READY_PENALTY,
                    RT_EXCELLENT, RT_GOOD, RT_SLOW, RT_VERY_SLOW, PATTERNS,
                    generate_pod_ttl)
from environment.pod_manager import PodDeploymentManager
from utils.metrics_tracker import MetricsTracker


class K8sEnvBaseline(gym.Env):
    """
    Kubernetes Environment WITHOUT Safety Projection for Baselines (DRS, RLSK, EPRS).
    
    Features:
    - Full response time measurement (decision + API)
    - Available nodes tracking for scalability
    - Per-node LATEST POD DEPLOYMENT RESPONSE TIME in observation space
    - NO safety projection - ALL actions are executed as-is
    - Constraint violations result in penalties ONLY (deployment still happens)
    - Node readiness detection (penalty only, deployment still attempted)
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
        
        state_dim = self.num_nodes * 3 + 2
        self.observation_space = spaces.Box(0, 1, (state_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_nodes)
        
        self.pod_arrival_rate = POD_ARRIVAL_RATE
        self.pod_stack = deque()
        self.last_arrival_time = time.time()
        self.current_pod = None
        self.nodes = []
        
        self.latest_response_time = {}
        self.latest_response_time_normalized = {}
        self.node_response_history = {}
        self.max_response_time_ms = 500
        self.available_nodes_history = []
        
        self.reset()
        self.pod_manager.cleanup_all()
    
        # ========== NEW: Response Time Helper Methods ==========
    def _normalize_response_time(self, rt_ms: float) -> float:
        """Normalize response time to 0-1 range (1 = 500ms or more)"""
        return min(rt_ms / self.max_response_time_ms, 1.0)
    
    def _get_node_avg_response_time(self, node_name: str) -> float:
        """Get average response time for a node from history"""
        if node_name in self.node_response_history and self.node_response_history[node_name]:
            return np.mean(self.node_response_history[node_name])
        return 100.0  # Default
    
    def _get_node_response_time_std(self, node_name: str) -> float:
        """Get standard deviation of response time for a node"""
        if node_name in self.node_response_history and len(self.node_response_history[node_name]) > 1:
            return np.std(self.node_response_history[node_name])
        return 0.0
    # ========== END NEW ==========
    
    def _parse_k8s_resource(self, value: str) -> float:
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
        try:
            return float(value)
        except ValueError:
            return 0.0
    
    def _extract_percentage(self, value: str) -> float:
        if not value:
            return 0.0
        cleaned = value.strip().replace('(', '').replace(')', '').replace('%', '')
        try:
            return float(cleaned) / 100.0
        except ValueError:
            return 0.0
    
    def _get_node_conditions(self, conditions_text: str) -> Dict[str, bool]:
        conditions = {'Ready': False, 'MemoryPressure': False, 'DiskPressure': False, 
                      'PIDPressure': False, 'NetworkUnavailable': False}
        for line in conditions_text.split('\n'):
            line = line.strip()
            for cond in conditions.keys():
                if cond in line:
                    conditions[cond] = 'True' in line and 'False' not in line
                    break
        return conditions
    
    def _get_real_node_metrics(self) -> List[Dict]:
        try:
            result = subprocess.run(["kubectl", "get", "nodes", "--no-headers"],
                                    capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return self._get_fallback_nodes()
            
            node_names = [line.split()[0] for line in result.stdout.strip().split('\n') if line.strip()]
            nodes = []
            
            for node_name in node_names[:self.num_nodes]:
                desc_result = subprocess.run(["kubectl", "describe", "node", node_name],
                                            capture_output=True, text=True, timeout=10)
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
        fallback = [
            {'name': 'minikube', 'cpu': 0.6, 'mem': 1074.0,
             'cpu_percent': 0.07, 'mem_percent': 0.28, 'pods': 0, 'ready': True},
            {'name': 'minikube-m02', 'cpu': 0.4, 'mem': 512.0,
             'cpu_percent': 0.05, 'mem_percent': 0.13, 'pods': 0, 'ready': True},
            {'name': 'minikube-m03', 'cpu': 0.8, 'mem': 1024.0,
             'cpu_percent': 0.10, 'mem_percent': 0.26, 'pods': 0, 'ready': True},
            {'name': 'minikube-m04', 'cpu': 0.3, 'mem': 384.0,
             'cpu_percent': 0.04, 'mem_percent': 0.10, 'pods': 0, 'ready': True},
        ]
        return sorted(fallback, key=lambda x: x['name'])
    
    def _update_node_metrics(self):
        new_nodes = self._get_real_node_metrics()
        while len(new_nodes) < self.num_nodes:
            new_nodes.append({
                'name': f'virtual-node-{len(new_nodes)}',
                'cpu': 0.0, 'mem': 0.0,
                'cpu_percent': 0.0, 'mem_percent': 0.0,
                'pods': 0, 'ready': False
            })
        self.nodes = new_nodes[:self.num_nodes]
        for node in self.nodes:
            node_name = node['name']
            if node_name not in self.latest_response_time:
                self.latest_response_time[node_name] = 5.0
                self.latest_response_time_normalized[node_name] = self._normalize_response_time(5.0)
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
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self._update_node_metrics()
        #self.pod_manager.cleanup_all()
        self.pod_manager.cleanup_old_pods
        self.pod_stack = deque()
        self.last_arrival_time = time.time()
        self._generate_poisson_arrivals()
        # Safety projection tracking
        self.projection_active = False
        self.projection_count = 0
        self.delay_count = 0
        
        if not self._get_next_pod():
            self.current_pod = None
        
        self.step_count = 0
        self.total_reward = 0
        self.available_nodes_history = []
        self.latest_used_node = np.random.choice(self.nodes)
        self.latest_response_time = {n['name']: 5.0 for n in self.nodes}
        self.latest_response_time_normalized = {n['name']: self._normalize_response_time(5.0)\
            for n in self.nodes}
        self.node_response_history = {n['name']: [] for n in self.nodes}
        
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
            obs.append(node['cpu_percent'])
            obs.append(node['mem_percent'])
            
        node_name = self.latest_used_node['name']
        rt_normalized = self.latest_response_time_normalized.get(node_name, 0.01)
        obs.append(rt_normalized)
        
        if self.current_pod:
            obs.append(self.current_pod['cpu'])
            obs.append(self.current_pod['mem'] / 1000.0)
        else:
            obs.append(0.0)
            obs.append(0.0)
        
        return np.array(obs, dtype=np.float32)
    
    def _get_response_time_bonus(self, response_time_ms: float) -> float:
        if response_time_ms < RT_EXCELLENT:
            return 2.0
        elif response_time_ms < RT_GOOD:
            return 1.0
        elif response_time_ms < RT_SLOW:
            return 0.0
        elif response_time_ms < RT_VERY_SLOW:
            return -1.0
        else:
            return -3.0
    
        # ========== NEW: Update Response Time Method ==========
    def _update_latest_response_time(self, node_name: str, new_rt_ms: float):
        """Update the LATEST response time for a node after successful deployment"""
        self.latest_response_time[node_name] = new_rt_ms
        self.latest_response_time_normalized[node_name] = self._normalize_response_time(new_rt_ms)
        
        # Add to history
        if node_name not in self.node_response_history:
            self.node_response_history[node_name] = []
        self.node_response_history[node_name].append(new_rt_ms)
        
        # Keep history bounded (last 100 deployments)
        if len(self.node_response_history[node_name]) > 100:
            self.node_response_history[node_name] = self.node_response_history[node_name][-100:]
        
        avg_rt = self._get_node_avg_response_time(node_name)
        print(f"  📊 Node {node_name} response time: {new_rt_ms:.1f}ms (avg: {avg_rt:.1f}ms)")
    # ========== END NEW ==========
    
        # ========== NEW: Per-Node Response Statistics Method ==========
    def get_per_node_response_stats(self) -> Dict:
        """Get per-node response time statistics"""
        stats = {}
        for node_name in self.node_response_history.keys():
            if self.node_response_history[node_name]:
                stats[node_name] = {
                    'latest': self.latest_response_time.get(node_name, 0),
                    'avg': np.mean(self.node_response_history[node_name]),
                    'min': np.min(self.node_response_history[node_name]),
                    'max': np.max(self.node_response_history[node_name]),
                    'std': np.std(self.node_response_history[node_name]) if len(self.node_response_history[node_name]) > 1 else 0,
                    'count': len(self.node_response_history[node_name])
                }
            else:
                stats[node_name] = {
                    'latest': 100.0, 'avg': 100.0, 'min': 100.0, 'max': 100.0, 'std': 0.0, 'count': 0
                }
        return stats
    # ========== END NEW ==========
    
    def select_best_node_greedy(self, rt, env_nodes, pod_resources):
        """
        Greedy node selection based on PPO-LRT reward formula.
        Selects node that would give highest immediate reward.
        """
        best_node = 0
        best_reward = -float('inf')
        
        # Simulate deploying pod to each node
        for node_idx, node in enumerate(env_nodes):
            if not node.get('ready', True):
                continue
                
            # Projected loads after deployment
            projected_cpu = node['cpu_percent'] + pod_resources['cpu'] / node.get('cpu_capacity', 1.0)
            projected_mem = node['mem_percent'] + pod_resources['mem'] / node.get('mem_capacity', 1.0)
            
            # Calculate projected reward
            # (simplified - would need full cluster state projection)
            cpu_loads = [n['cpu_percent'] for n in env_nodes]
            mem_loads = [n['mem_percent'] for n in env_nodes]
            cpu_loads[node_idx] = projected_cpu
            mem_loads[node_idx] = projected_mem
            
            sigma_c = np.std(cpu_loads) + 1e-8
            sigma_m = np.std(mem_loads) + 1e-8
            
            load_balance = PPO_LRT_ALPHA * (1/sigma_c + 1/sigma_m)
            penalty = 0.0
            for cpu in cpu_loads:
                        if cpu > PPO_LRT_CPU_UPPER:
                            penalty += (cpu - PPO_LRT_CPU_UPPER)
            for mem in mem_loads:
                        if mem > PPO_LRT_MEM_UPPER:
                            penalty += (mem - PPO_LRT_MEM_UPPER)
            penalty_component = PPO_LRT_BETA * penalty
                    
                    # Response time component: γ * R_i
                    # Normalize response time to seconds
            rt_component = PPO_LRT_GAMMA_RT * (rt / 1000.0)
            reward = load_balance - penalty_component - rt_component
            if reward > best_reward:
                best_reward = reward
                best_node = node_idx
        
        return best_node
    
    def step(self, action):
        decision_start = time.perf_counter()
        
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
        projection_triggered = False
        done = False
        #complutre reward map for done
        old_nodes = copy.deepcopy(self.nodes)
        
        self._generate_poisson_arrivals()
        self.pod_manager.cleanup_old_pods()
        
        if action < len(self.nodes):
            target_node = self.nodes[action]
            node_name = target_node['name']
            
            pod_cpu = self.current_pod['cpu']
            pod_mem_mib = self.current_pod['mem']
            
            new_cpu = target_node['cpu'] + pod_cpu
            new_mem = target_node['mem'] + pod_mem_mib
            
            if target_node['cpu'] > 0:
                projected_cpu_util = target_node['cpu_percent'] * (new_cpu / target_node['cpu'])
            else:
                projected_cpu_util = 1.0
            
            if target_node['mem'] > 0:
                projected_mem_util = target_node['mem_percent'] * (new_mem / target_node['mem'])
            else:
                projected_mem_util = 1.0
            
            if projected_cpu_util > CPU_LIMIT or projected_mem_util > MEM_LIMIT:
                projection_triggered = True
                print(f"  ⚠️ CONSTRAINT VIOLATION on {node_name} - "
                      f"Projected CPU: {projected_cpu_util*100:.1f}% > {CPU_LIMIT*100:.0f}%, "
                      f"Projected MEM: {projected_mem_util*100:.1f}% > {MEM_LIMIT*100:.0f}%")
            
            if not target_node.get('ready', True):
                reward = REWARD_NODE_NOT_READY_PENALTY
                self.pod_stack.append(self.current_pod)
                print(f"  ⚠️ Node {node_name} not ready - penalty {reward}")
            else:
                api_start = time.perf_counter()
                
                deployed, _, api_response_time = self.pod_manager.deploy_nginx_pod(
                    node_name=node_name,
                    cpu_request=pod_cpu,
                    mem_request=int(pod_mem_mib),
                    custom_ttl=self.current_pod.get('ttl')
                )
                
                api_response_time = (time.perf_counter() - api_start) * 1000
                best_node = \
                        self.select_best_node_greedy(api_response_time, self.nodes, self.current_pod)
                done = (best_node == action)
                print('expected node {}, actual node {}, done {}'.
                                format(best_node, action, done))
                # In environment/k8s_env_baseline.py, replace the reward calculation in step()

                # ============================================================
                # PPO-LRT REWARD FUNCTION (from paper Equation 12-14)
                # reward = α * (1/σ_c + 1/σ_m) - β * p - γ * R_i
                # ============================================================

                # After successful deployment, calculate PPO-LRT reward
                if deployed:
                    success = True
                    self._update_latest_response_time(node_name, api_response_time)
                    self.latest_used_node = target_node
                    self._update_node_metrics()
                    
                    # Get loads across all ready nodes
                    ready_nodes = [n for n in self.nodes if n.get('ready', True)]
                    cpu_loads = np.array([n['cpu_percent'] for n in ready_nodes])
                    mem_loads = np.array([n['mem_percent'] for n in ready_nodes])
                    
                    # σ_c, σ_m: standard deviation of loads (Equation 13)
                    sigma_c = np.std(cpu_loads) + 1e-8
                    sigma_m = np.std(mem_loads) + 1e-8
                    
                    # Load balance component: α * (1/σ_c + 1/σ_m)
                    load_balance = PPO_LRT_ALPHA * (1.0 / sigma_c + 1.0 / sigma_m)
                    
                    # Penalty component: β * p (Equation 14)
                    penalty = 0.0
                    for cpu in cpu_loads:
                        if cpu > PPO_LRT_CPU_UPPER:
                            penalty += (cpu - PPO_LRT_CPU_UPPER)
                    for mem in mem_loads:
                        if mem > PPO_LRT_MEM_UPPER:
                            penalty += (mem - PPO_LRT_MEM_UPPER)
                    penalty_component = PPO_LRT_BETA * penalty
                    
                    # Response time component: γ * R_i
                    # Normalize response time to seconds
                    rt_component = PPO_LRT_GAMMA_RT * (api_response_time / 1000.0)
                    
                    # Final reward (Equation 12)
                    reward = load_balance - penalty_component - rt_component
                    # Clip reward to reasonable range
                    reward = np.clip(reward, -10.0, 10.0)
                    
                    print(f"  ✅ Deployed to {node_name} | API: {api_response_time:.1f}ms | "
                        f"Memory Utilization: {target_node['mem_percent'] * 100:.2f} % | ")
                else:
                    reward = -5.0  # Failure penalty
        else:
            reward = REWARD_DELAY_PENALTY
            self.pod_stack.append(self.current_pod)
            print(f"  ⏸️ Delay action - penalty {reward}")
        
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
            'projection_triggered': projection_triggered
        }
    
    def close(self):
        self.pod_manager.cleanup_all()
        if self.tracker:
            self.tracker.end_episode()
