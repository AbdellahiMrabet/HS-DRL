# utils/metrics_tracker.py
import numpy as np
from typing import Dict, List, Any

class MetricsTracker:
    """Track training metrics with per-node resource tracking"""
    
    __slots__ = [
        'rewards', 'successes', 'utils', 'disk_ios', 'imbalances', 'response_times',
        'api_response_times', 'full_response_times', 'projections', 'violations',
        'per_node_cpu', 'per_node_mem', 'episode_violation', 'episode_projection',
        'episode_rewards', 'episode_success', 'episode_utils', 'episode_disk_ios',
        'episode_imbalances', 'episode_responses', 'deployed_pods', 'arrival_pods',
        'available_nodes', 'episode_avg_available', 'episode_min_available',
        'node_changes', 'recovery_times', 'node_status_history',
        'step_count', 'per_node_response_time'
    ]
    
    def __init__(self):
        self.rewards = []
        self.successes = []
        self.utils = []
        self.disk_ios = []
        self.imbalances = []
        self.response_times = []
        self.api_response_times = []
        self.full_response_times = []
        self.arrival_pods = []
        self.available_nodes = []
        
        self.per_node_cpu = {}
        self.per_node_mem = {}
        
        self.episode_rewards = []
        self.episode_success = []
        self.episode_utils = []
        self.episode_disk_ios = []
        self.episode_imbalances = []
        self.episode_responses = []
        
        self.deployed_pods = 0
        self.projections = 0
        self.violations = 0 
        self.step_count = 0
        self.episode_violation = []
        self.episode_projection =[]
        
        self.episode_avg_available = []
        self.episode_min_available = []
        self.node_changes = []
        self.recovery_times = []
        self.node_status_history = {}
        self.per_node_response_time = {}

    def step(self, reward, success, util, disk_io, imbalance, 
             full_response_time, api_response_time, nodes_data=None,
             latest_response_time=None,
             projection=False, arrival_count=0, available_nodes=0):
        """Record a single step metrics including per-node data"""
        self.rewards.append(reward)
        self.successes.append(1 if success else 0)
        self.utils.append(util)
        self.disk_ios.append(disk_io)
        self.imbalances.append(imbalance)
        self.full_response_times.append(full_response_time)
        self.api_response_times.append(api_response_time)
        self.response_times.append(full_response_time)
        self.arrival_pods.append(arrival_count)
        self.available_nodes.append(available_nodes)
        
        if nodes_data and success:
            for node in nodes_data:
                name = node.get('name', f'node_{len(self.per_node_cpu)}')
                if name not in self.per_node_response_time:
                    self.per_node_response_time[name] = []
                if latest_response_time:
                    response_time = latest_response_time.get(name, 200)
                self.per_node_response_time[name].append(response_time)
        
        if nodes_data:
            for node in nodes_data:
                name = node.get('name', f'node_{len(self.per_node_cpu)}')
                if name not in self.per_node_cpu:
                    self.per_node_cpu[name] = []
                    self.per_node_mem[name] = []
                
                cpu_val = node.get('cpu_percent', node.get('cpu', 0))
                mem_val = node.get('mem_percent', node.get('mem', 0))
                
                self.per_node_cpu[name].append(cpu_val)
                self.per_node_mem[name].append(mem_val)
    
        if projection:
            self.violations += 1
            self.projections += 1
        if success:
            self.deployed_pods += 1
        
        self.step_count = len(self.successes)

    def end_episode(self):
        """Call at the end of each episode to aggregate step-level metrics"""
        self.episode_rewards.append(np.sum(self.rewards) if self.rewards else 0)
        self.episode_success.append(np.mean(self.successes) * 100 if self.successes else 0)
        self.episode_utils.append(np.mean(self.utils) * 100 if self.utils else 0)
        self.episode_disk_ios.append(np.mean(self.disk_ios) * 100 if self.disk_ios else 0)
        self.episode_imbalances.append(np.mean(self.imbalances) * 100 if self.imbalances else 0)
        self.episode_responses.append(np.mean(self.full_response_times) if self.full_response_times else 0)
        self.episode_avg_available.append(np.mean(self.available_nodes) if self.available_nodes else 4)
        self.episode_min_available.append(np.min(self.available_nodes) if self.available_nodes else 4)
        self.episode_projection.append(self.projections if self.projections else 0)
        self.episode_violation.append(self.violations if self.violations else 0)
        
        self.rewards = []
        self.successes = []
        self.utils = []
        self.disk_ios = []
        self.imbalances = []
        self.response_times = []
        self.api_response_times = []
        self.full_response_times = []
        self.arrival_pods = []
        self.available_nodes = []
        self.projections = 0
        self.violations = 0
        self.per_node_cpu = {}
        self.per_node_mem = {}
        self.per_node_response_time = {}

    def get_episode_summary(self, episode_num: int) -> Dict:
        """Get summary metrics for the current episode including per-node averages"""
        summary = {
            'episode': episode_num,
            'total_reward': self.episode_rewards[-1] if self.episode_rewards else 0,
            'success_rate': self.episode_success[-1] if self.episode_success else 0,
            'avg_util': self.episode_utils[-1] if self.episode_utils else 0,
            'avg_imbalance': self.episode_imbalances[-1] if self.episode_imbalances else 0,
            'avg_response_time': self.episode_responses[-1] if self.episode_responses else 0,
            'constraint_violations': self.episode_violation[-1] if self.episode_violation else 0,
            'projections': self.episode_projection[-1] if self.episode_projection[-1] else 0,
            'deployed_pods': self.deployed_pods,
            'avg_available_nodes': self.episode_avg_available[-1] if self.episode_avg_available else 4,
            'min_available_nodes': self.episode_min_available[-1] if self.episode_min_available else 4,
            'safety_compliance_rate': 100 - (self.violations / max(self.step_count, 1) * 100),
        }
        
        # Get per-node averages from the step-level data
        temp_cpu = {}
        temp_mem = {}
        for name in self.per_node_cpu.keys():
            if self.per_node_cpu[name]:
                temp_cpu[name] = np.mean(self.per_node_cpu[name]) * 100
            else:
                temp_cpu[name] = 0.0
        
        for name in self.per_node_mem.keys():
            if self.per_node_mem[name]:
                temp_mem[name] = np.mean(self.per_node_mem[name]) * 100
            else:
                temp_mem[name] = 0.0
        
        for name, val in temp_cpu.items():
            summary[f'{name}_avg_cpu'] = val
        for name, val in temp_mem.items():
            summary[f'{name}_avg_mem'] = val
            
        for name in self.per_node_response_time.keys():
            if self.per_node_response_time[name]:
                summary[f'{name}_avg_rt'] = np.mean(self.per_node_response_time[name])
            else:
                summary[f'{name}_avg_rt'] = 100.0
        return summary

    def get_per_node_stats(self) -> Dict:
        return {
            name: {
                'avg_cpu': np.mean(self.per_node_cpu[name]) * 100 if name in self.per_node_cpu and self.per_node_cpu[name] else 0,
                'avg_mem': np.mean(self.per_node_mem[name]) * 100 if name in self.per_node_mem and self.per_node_mem[name] else 0
            }
            for name in set(list(self.per_node_cpu.keys()) + list(self.per_node_mem.keys()))
        }

    def get_performance_metrics(self) -> Dict:
        return {
            'success_rate': np.mean(self.episode_success) if self.episode_success else 0,
            'deployed_pods': self.deployed_pods,
            'avg_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'avg_util': np.mean(self.episode_utils) if self.episode_utils else 0,
            'avg_imbalance': np.mean(self.episode_imbalances) if self.episode_imbalances else 0,
            'avg_response_time': np.mean(self.episode_responses) if self.episode_responses else 0,
        }

    def get_safety_metrics(self) -> Dict:
        return {
            'constraint_violations': self.violations,
            'projections': self.projections,
            'safety_compliance_rate': 100 - (self.violations / max(self.step_count, 1) * 100)
        }

    def get_scalability_metrics(self) -> Dict:
        return {
            'avg_available_nodes': np.mean(self.episode_avg_available) if self.episode_avg_available else 4,
            'min_available_nodes': np.min(self.episode_min_available) if self.episode_min_available else 4,
            'node_changes': len(self.node_changes),
            'avg_recovery_time': np.mean(self.recovery_times) if self.recovery_times else 0,
        }
    
    def summary(self) -> str:
        m = self.get_performance_metrics()
        s = self.get_safety_metrics()
        node_stats = self.get_per_node_stats()
        
        summary_str = f"\n{'='*40}\nMETRICS SUMMARY\n{'='*40}\n"
        summary_str += f"Success Rate: {m['success_rate']:.1f}%\nAvg Reward: {m['avg_reward']:.2f}\n"
        summary_str += f"Violations: {s['constraint_violations']}\nProjections: {s['projections']}\n"
        summary_str += "\nPER-NODE UTILIZATION:"
        for node, stats in node_stats.items():
            summary_str += f"\n  {node}: CPU {stats['avg_cpu']:.1f}%, Mem {stats['avg_mem']:.1f}%"
        return summary_str + f"\n{'='*40}"