# plot_results.py - Generate plots from saved CSV files

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.csv_saver import ResultsLoader
from config import COLORS


class ResultsPlotter:
    def __init__(self, results_dir: str = "results", plots_dir: str = "results/plots"):
        self.results_dir = results_dir
        self.plots_dir = plots_dir
        self.loader = ResultsLoader(results_dir)
        os.makedirs(plots_dir, exist_ok=True)
        self.colors = COLORS
        self.default_color = '#888888'
    
    def load_data(self):
        data = self.loader.load_all()
        return {name: episodes for name, episodes in data.items() if episodes}
    
    def smooth_curve(self, data, window=5):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    def _get_node_label(self, key):
        """Convert CSV column names to readable node labels"""
        # Handle CPU keys
        node_name = key.replace('_avg_cpu', '').replace('_cpu_avg', '')
        node_name = node_name.replace('_cpu_mean', '').replace('_cpu', '')
        # Handle Memory keys
        node_name = node_name.replace('_avg_mem', '').replace('_mem_avg', '')
        node_name = node_name.replace('_mem_mean', '').replace('_memory', '')
        # Handle Response Time keys
        node_name = node_name.replace('_avg_rt', '').replace('_rt_avg', '')
        node_name = node_name.replace('_response_time', '').replace('_resp_time', '')
        # Clean up
        node_name = node_name.replace('_', '-').strip('-')
        
        # Map to standardized names
        node_mapping = {
            'minikube': 'node1',
            'minikube-m02': 'node2',
            'minikube-m03': 'node3',
            'minikube-m04': 'node4',
            'node1': 'node1',
            'node2': 'node2',
            'node3': 'node3',
            'node4': 'node4'
        }
        
        return node_mapping.get(node_name, node_name.upper())
    
    def plot_learning_curves_rewards(self, data):
        """Plot episode rewards - separate figure"""
        plt.figure(figsize=(12, 6))
        eprs_rewards = 0
        rlsk_rewards = 0
        hs_hdr_rewards = 0
        for name, episodes in data.items():
            if name == 'EPRS':
                eprs_rewards = sum([e.get('total_reward') for e in episodes if name == 'EPRS' and 'total_reward' in e])
            elif name == 'RLSK':
                rlsk_rewards = sum([e.get('total_reward') for e in episodes if name == 'RLSK' and 'total_reward' in e])
            elif name == 'HS_HDR':
                hs_hdr_rewards = sum([e.get('total_reward') for e in episodes if name == 'HS_HDR' and 'total_reward' in e])
            if not episodes:
                continue
            try:
                ep_nums = [e['episode'] for e in episodes]
                rewards = [e['total_reward'] for e in episodes]
            except KeyError:
                continue

            sr = self.smooth_curve(rewards, 10)
            eps = ep_nums[len(ep_nums)-len(sr):]
            
            color = self.colors.get(name, self.default_color)
            plt.plot(eps, sr, label=name, color=color, linewidth=2)
            print(f"  ✓ Processed rewards for {name} (total: {sum(rewards):.1f})")
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Total Reward', fontsize=12)
        plt.title('Episode Rewards Over Training', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'learning_curves_rewards.png'), dpi=150)
        plt.close()
        print("  ✓ Saved: learning_curves_rewards.png")
    
    def plot_learning_curves_success(self, data):
        """Plot success rate - separate figure"""
        plt.figure(figsize=(12, 6))
        for name, episodes in data.items():
            if not episodes:
                continue
            try:
                ep_nums = [e['episode'] for e in episodes]
                success = [e['success_rate'] for e in episodes]
            except KeyError:
                continue

            ss = self.smooth_curve(success, 10)
            eps = ep_nums[len(ep_nums)-len(ss):]
            
            color = self.colors.get(name, self.default_color)
            plt.plot(eps, ss, label=name, color=color, linewidth=2)
        
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Success Rate (%)', fontsize=12)
        plt.title('Success Rate Over Training', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'learning_curves_success.png'), dpi=150)
        plt.close()
        print("  ✓ Saved: learning_curves_success.png")
    
    def plot_learning_curves(self, data):
        """Legacy method - calls the two separate methods for backward compatibility"""
        self.plot_learning_curves_rewards(data)
        self.plot_learning_curves_success(data)
    
    def plot_epsilon_decay(self, data):
        plt.figure(figsize=(10, 6))
        for name, episodes in data.items():
            if not episodes or 'epsilon' not in episodes[0]:
                continue
            ep_nums = [e['episode'] for e in episodes]
            epsilons = [e['epsilon'] for e in episodes]
            color = self.colors.get(name, self.default_color)
            plt.plot(ep_nums, epsilons, label=name, color=color, linewidth=2)
        
        plt.xlabel('Episode'); plt.ylabel('Epsilon')
        plt.title('Epsilon Decay (Exploration Rate)')
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'epsilon_decay.png'), dpi=150)
        plt.close()
        print("  ✓ Saved: epsilon_decay.png")
    
    def plot_final_comparison(self, data):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        names = list(data.keys())
        colors = [self.colors.get(n, self.default_color) for n in names]
        
        ax = axes[0, 0]
        success = []
        for episodes in data.values():
            if episodes:
                val = episodes[-1].get('success_rate')
                success.append(val if val is not None else 0)
            else:
                success.append(0)
        bars = ax.bar(names, success, color=colors)
        ax.set_ylabel('Success Rate (%)'); ax.set_title('Final Success Rate')
        for bar, val in zip(bars, success):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        
        ax = axes[0, 1]
        rewards = []
        for episodes in data.values():
            if episodes:
                val = episodes[-1].get('total_reward')
                rewards.append(val if val is not None else 0)
            else:
                rewards.append(0)
        bars = ax.bar(names, rewards, color=colors)
        ax.set_ylabel('Total Reward'); ax.set_title('Final Total Reward')
        for bar, val in zip(bars, rewards):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        ax = axes[1, 0]
        violations = []
        for episodes in data.values():
            total = 0
            for e in episodes:
                val = e.get('constraint_violations')
                if val is not None:
                    total += val
            violations.append(val)
        bars = ax.bar(names, violations, color=colors)
        ax.set_ylabel('Total Violations'); ax.set_title('Total Constraint Violations')
        for bar, val in zip(bars, violations):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{int(val)}', ha='center', va='bottom', fontsize=9)
        
        ax = axes[1, 1]
        rt = []
        for episodes in data.values():
            if episodes:
                val = episodes[-1].get('avg_response_time')
                rt.append(val if val is not None else 0)
            else:
                rt.append(0)
        bars = ax.bar(names, rt, color=colors)
        ax.set_ylabel('Response Time (ms)'); ax.set_title('Average Response Time')
        for bar, val in zip(bars, rt):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        for a in axes.flat:
            a.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'final_comparison.png'), dpi=150)
        plt.close()
        print("  ✓ Saved: final_comparison.png")
    
    def plot_safety_compliance_rate(self, data):
        """Plot safety compliance rate comparison - separate figure"""
        plt.figure(figsize=(10, 6))
        
        # Filter out unwanted agents
        names = [k for k in data.keys()]
        colors = [self.colors.get(n, self.default_color) for n in names]
        
        safety_rates = []
        for name in names:
            episodes = data[name]
            if episodes:
                total = 0
                count = 0
                for e in episodes:
                    count += 1
                    total += e.get('safety_compliance_rate')
                #val = episodes[-1].get('safety_compliance_rate')
                safety_rates.append(total / count)
            else:
                safety_rates.append(0)
        
        bars = plt.bar(names, safety_rates, color=colors)
        plt.ylabel('Safety Compliance (%)', fontsize=12)
        plt.title('Safety Compliance Rate Comparison', fontsize=14)
        plt.ylim(0, 105)
        
        for bar, val in zip(bars, safety_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'safety_compliance_rate.png'), dpi=150)
        plt.close()
        print("  ✓ Saved: safety_compliance_rate.png")


    def plot_constraint_violations_comparison(self, data):
        """Plot constraint violations comparison - separate figure"""
        plt.figure(figsize=(10, 6))
        
        # Filter out unwanted agents
        names = [k for k in data.keys()]
        colors = [self.colors.get(n, self.default_color) for n in names]
        
        violations = []
        for name in names:
            episodes = data[name]
            total = 0
            for e in episodes:
                #print('episode', e)
                val = e.get('projections')
                if val is not None:
                    total += val
            violations.append(total)
        
        bars = plt.bar(names, violations, color=colors)
        plt.ylabel('Total Violations (projected actions for HS-DRL)', fontsize=12)
        plt.title('Total Constraint Violations Comparison', fontsize=14)
        
        for bar, val in zip(bars, violations):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{int(val)}', ha='center', va='bottom', fontsize=10)
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'constraint_violations_comparison.png'), dpi=150)
        plt.close()
        print("  ✓ Saved: constraint_violations_comparison.png")


    def plot_safety_comparison(self, data):
        """Legacy method - calls the two separate methods for backward compatibility"""
        self.plot_safety_compliance_rate(data)
        self.plot_constraint_violations_comparison(data)
    
    def plot_constraint_violations_over_time(self, data):
        """Plot smoothed cumulative constraint violations over episodes"""
        plt.figure(figsize=(12, 6))
        
        for name, episodes in data.items():
            if not episodes:
                continue
            
            ep_nums = [e['episode'] for e in episodes]
            
            # Calculate cumulative violations
            cum_violations = []
            total = 0
            for e in episodes:
                val = e.get('constraint_violations')
                if val is not None:
                    total += val
                cum_violations.append(val)
            
            color = self.colors.get(name.split('_')[0], self.default_color)
            linestyle = '-' if 'hsdrl' in name.lower() else '--'
            
            # Plot smoothed cumulative
            if len(cum_violations) >= 15:
                smoothed = self.smooth_curve(cum_violations, 15)
                valid_eps = ep_nums[len(ep_nums) - len(smoothed):]
                plt.plot(valid_eps, smoothed, label=name, color=color,
                        linestyle=linestyle, linewidth=2.5)
            else:
                plt.plot(ep_nums, cum_violations, label=name, color=color,
                        linestyle=linestyle, linewidth=2.5)
        
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Cumulative Constraint Violations', fontsize=12)
        plt.title('Constraint Violations Over Time (Smoothed)', fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'constraint_violations_trend.png'), dpi=150)
        plt.close()
        print("  ✓ Saved: constraint_violations_trend.png")
    
    def plot_per_node_cpu(self, data):
        """Plot per-node CPU utilization - separate figure for each agent"""
        for name, episodes in data.items():
            if not episodes:
                print(f"  ⚠️ No episodes for {name}, skipping per_node_cpu")
                continue
            
            node_keys = []
            possible_patterns = ['_avg_cpu', '_cpu_avg', '_cpu_mean', '_cpu']
            for pattern in possible_patterns:
                keys = sorted([k for k in episodes[0].keys() if pattern in k.lower()])
                if keys:
                    node_keys = keys
                    break
            
            if not node_keys:
                print(f"  ⚠️ No per-node CPU data for {name}, skipping")
                continue
            
            plt.figure(figsize=(12, 7))
            ep_nums = [e['episode'] for e in episodes]
            
            for key in node_keys:
                node_label = self._get_node_label(key)  # Use helper method
                
                values = []
                for e in episodes:
                    val = e.get(key)
                    values.append(val if val is not None else 0)
                
                if len(values) > 5:
                    try:
                        smoothed = self.smooth_curve(values, 5)
                    except (TypeError, ValueError):
                        continue

                    valid_eps = ep_nums[:len(smoothed)]
                    plt.plot(valid_eps, smoothed, label=node_label, linewidth=2)
                else:
                    plt.plot(ep_nums, values, label=node_label, linewidth=2)
            
            plt.xlabel('Episode', fontsize=12)
            plt.ylabel('Avg CPU Utilization (%)', fontsize=12)
            plt.title(f'{name} - Per-Node CPU Utilization Over Training', fontsize=14)
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 100)
            plt.tight_layout()
            
            output_path = os.path.join(self.plots_dir, f'{name}_per_node_cpu.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: {name}_per_node_cpu.png (found {len(node_keys)} nodes)")
    
    def plot_per_node_mem(self, data):
        """Plot per-node Memory utilization - separate figure for each agent"""
        for name, episodes in data.items():
            if not episodes:
                print(f"  ⚠️ No episodes for {name}, skipping per_node_mem")
                continue
            
            node_keys = []
            possible_patterns = ['_avg_mem', '_mem_avg', '_mem_mean', '_memory']
            for pattern in possible_patterns:
                keys = sorted([k for k in episodes[0].keys() if pattern in k.lower()])
                if keys:
                    node_keys = keys
                    break
            
            if not node_keys:
                print(f"  ⚠️ No per-node Memory data for {name}, skipping")
                continue
            
            plt.figure(figsize=(12, 7))
            ep_nums = [e['episode'] for e in episodes]
            
            for key in node_keys:
                node_label = self._get_node_label(key)  # Use helper method
                
                values = []
                for e in episodes:
                    val = e.get(key)
                    values.append(val if val is not None else 0)
                
                if len(values) > 5:
                    try:
                        smoothed = self.smooth_curve(values, 5)
                    except (TypeError, ValueError):
                        continue
                    valid_eps = ep_nums[:len(smoothed)]
                    plt.plot(valid_eps, smoothed, label=node_label, linewidth=2)
                else:
                    plt.plot(ep_nums, values, label=node_label, linewidth=2)
            
            plt.xlabel('Episode', fontsize=12)
            plt.ylabel('Avg Memory Utilization (%)', fontsize=12)
            plt.title(f'{name} - Per-Node Memory Utilization Over Training', fontsize=14)
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 100)
            plt.tight_layout()
            
            output_path = os.path.join(self.plots_dir, f'{name}_per_node_mem.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: {name}_per_node_mem.png (found {len(node_keys)} nodes)")
    
    def plot_per_node_load_balance(self, data):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        names = list(data.keys())
        colors = [self.colors.get(n, self.default_color) for n in names]
        
        cpu_variances = []
        mem_variances = []
        for episodes in data.values():
            if not episodes:
                cpu_variances.append(0); mem_variances.append(0)
                continue
            last = episodes[-1]
            
            cpu_keys = []
            for pattern in ['_avg_cpu', '_cpu_avg', '_cpu']:
                keys = [k for k in last.keys() if pattern in k.lower()]
                if keys:
                    cpu_keys = keys
                    break
            
            mem_keys = []
            for pattern in ['_avg_mem', '_mem_avg', '_memory']:
                keys = [k for k in last.keys() if pattern in k.lower()]
                if keys:
                    mem_keys = keys
                    break
            
            cpu_vals = [last.get(k, 0) if last.get(k) is not None else 0 for k in cpu_keys]
            mem_vals = [last.get(k, 0) if last.get(k) is not None else 0 for k in mem_keys]
            
            cpu_variances.append(np.var(cpu_vals) if cpu_vals else 0)
            mem_variances.append(np.var(mem_vals) if mem_vals else 0)
        
        bars = axes[0].bar(names, cpu_variances, color=colors)
        axes[0].set_ylabel('CPU Variance'); axes[0].set_title('CPU Load Balance (Lower is Better)')
        for bar, val in zip(bars, cpu_variances):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        bars = axes[1].bar(names, mem_variances, color=colors)
        axes[1].set_ylabel('Memory Variance'); axes[1].set_title('Memory Load Balance (Lower is Better)')
        for bar, val in zip(bars, mem_variances):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        for ax in axes:
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'per_node_load_balance.png'), dpi=150)
        plt.close()
        print("  ✓ Saved: per_node_load_balance.png")
    
    def plot_node_heatmap(self, data):
        for name, episodes in data.items():
            if not episodes:
                continue
            
            cpu_keys = []
            for pattern in ['_avg_cpu', '_cpu_avg', '_cpu']:
                keys = sorted([k for k in episodes[0].keys() if pattern in k.lower()])
                if keys:
                    cpu_keys = keys
                    break
            
            if not cpu_keys:
                continue
            
            cpu_matrix = []
            mem_matrix = []
            for e in episodes:
                cpu_row = []
                mem_row = []
                for k in cpu_keys:
                    val = e.get(k)
                    cpu_row.append(val if val is not None else 0)
                    mem_key = k.replace('cpu', 'mem').replace('CPU', 'MEM')
                    val = e.get(mem_key)
                    mem_row.append(val if val is not None else 0)
                cpu_matrix.append(cpu_row)
                mem_matrix.append(mem_row)
            
            cpu_matrix = np.array(cpu_matrix)
            mem_matrix = np.array(mem_matrix)
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            node_labels = [k.replace('_avg_cpu', '').replace('_cpu_avg', '').replace('_cpu', '')
                           .replace('_', '-').strip('-').upper() for k in cpu_keys]
            
            im1 = axes[0].imshow(cpu_matrix.T, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=100)
            axes[0].set_yticks(range(len(node_labels))); axes[0].set_yticklabels(node_labels)
            axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Node')
            axes[0].set_title(f'{name} - CPU Heatmap')
            plt.colorbar(im1, ax=axes[0], label='CPU %')
            
            im2 = axes[1].imshow(mem_matrix.T, aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=100)
            axes[1].set_yticks(range(len(node_labels))); axes[1].set_yticklabels(node_labels)
            axes[1].set_xlabel('Episode'); axes[1].set_ylabel('Node')
            axes[1].set_title(f'{name} - Memory Heatmap')
            plt.colorbar(im2, ax=axes[1], label='Memory %')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plots_dir, f'{name}_node_heatmap.png'), dpi=150)
            plt.close()
        print("  ✓ Saved: node_heatmap plots")
    
    def plot_available_nodes_over_time(self, data):
        plt.figure(figsize=(12, 6))
        for name, episodes in data.items():
            if not episodes:
                continue
            ep_nums = [e['episode'] for e in episodes]
            avg_available = [e.get('avg_available_nodes', 4) if e.get('avg_available_nodes') is not None else 4 for e in episodes]
            color = self.colors.get(name, self.default_color)
            plt.plot(ep_nums, avg_available, 'o-', label=name, color=color, linewidth=2, markersize=4)
        
        plt.xlabel('Episode'); plt.ylabel('Available Nodes')
        plt.title('Available Nodes Over Time'); plt.legend(); plt.grid(True, alpha=0.3)
        plt.ylim(0, 5)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'available_nodes_over_time.png'), dpi=150)
        plt.close()
        print("  ✓ Saved: available_nodes_over_time.png")
    def plot_per_node_response_time_comparison(self, data):
        """Plot final per-node response time comparison across agents"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Filter out unwanted agents
        
        # Collect all unique node labels across agents
        all_nodes = set()
        agent_node_data = {}
        
        for name, episodes in data.items():
            if not episodes:
                continue
            
            # Find response time keys
            rt_keys = []
            possible_patterns = ['_avg_rt', '_rt_avg', '_response_time', '_resp_time']
            for pattern in possible_patterns:
                keys = [k for k in episodes[-1].keys() if pattern in k.lower()]
                if keys:
                    rt_keys = keys
                    break
            
            if not rt_keys:
                continue
            
            node_labels = [self._get_node_label(k) for k in rt_keys]
            rt_values = [episodes[-1].get(k, 0) if episodes[-1].get(k) is not None else 0 for k in rt_keys]
            
            for node, value in zip(node_labels, rt_values):
                all_nodes.add(node)
            
            agent_node_data[name] = dict(zip(node_labels, rt_values))
        
        if not all_nodes:
            print("  ⚠️ No per-node response time data found for comparison")
            return
        
        all_nodes = sorted(list(all_nodes))
        
        # Set up bar positions
        x = np.arange(len(all_nodes))
        width = 0.25
        multiplier = 0
        
        for name, node_data in agent_node_data.items():
            values = [node_data.get(node, 0) for node in all_nodes]
            offset = width * multiplier
            bars = ax.bar(x + offset, values, width, label=name, 
                        color=self.colors.get(name, self.default_color))
            multiplier += 1
        
        ax.set_ylabel('Response Time (ms)', fontsize=12)
        ax.set_title('Per-Node Response Time Comparison', fontsize=14)
        ax.set_xticks(x + width * (multiplier - 1) / 2)
        ax.set_xticklabels(all_nodes)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'per_node_response_time_comparison.png'), dpi=150)
        plt.close()
        print("  ✓ Saved: per_node_response_time_comparison.png")
    
    def plot_response_time(self, data):
        """Plot average response time over episodes"""
        plt.figure(figsize=(12, 6))
        for name, episodes in data.items():
            if not episodes:
                continue
            try:
                ep_nums = [e['episode'] for e in episodes]
                response_times = [e['avg_response_time'] for e in episodes]
            except KeyError:
                continue

            # Smooth the response time curve
            if len(response_times) > 10:
                smoothed = self.smooth_curve(response_times, 10)
                eps = ep_nums[len(ep_nums)-len(smoothed):]
            else:
                smoothed = response_times
                eps = ep_nums
            
            color = self.colors.get(name, self.default_color)
            plt.plot(eps, smoothed, label=name, color=color, linewidth=2)
        
        plt.xlabel('Episode', fontsize=12)
        plt.ylabel('Average Response Time (ms)', fontsize=12)
        plt.title('Average Response Time Over Training', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'response_time.png'), dpi=150)
        plt.close()
        print("  ✓ Saved: response_time.png")
    
    def plot_scalability_adaptation(self, data):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for name, episodes in data.items():
            if not episodes:
                continue
            
            avg_available = []
            success_rates = []
            response_times = []
            
            for e in episodes:
                val = e.get('avg_available_nodes')
                avg_available.append(val if val is not None else 4)
                val = e.get('success_rate')
                success_rates.append(val if val is not None else 0)
                val = e.get('avg_response_time')
                response_times.append(val if val is not None else 0)
            
            color = self.colors.get(name, self.default_color)
            axes[0].scatter(avg_available, success_rates, alpha=0.6, color=color, label=name, s=30)
            axes[1].scatter(avg_available, response_times, alpha=0.6, color=color, label=name, s=30)
        
        axes[0].set_xlabel('Available Nodes'); axes[0].set_ylabel('Success Rate (%)')
        axes[0].set_title('Success Rate vs Available Nodes'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
        axes[1].set_xlabel('Available Nodes'); axes[1].set_ylabel('Response Time (ms)')
        axes[1].set_title('Response Time vs Available Nodes'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'scalability_adaptation.png'), dpi=150)
        plt.close()
        print("  ✓ Saved: scalability_adaptation.png")
        # ========== NEW: Per-Node Response Time Plot ==========
    def plot_per_node_response_time(self, data):
        """Plot per-node response time - separate figure for each agent"""
        for name, episodes in data.items():
            if not episodes:
                continue
            
            # Find per-node response time keys
            rt_keys = sorted([k for k in episodes[0].keys() if k.endswith('_avg_rt')])
            if not rt_keys:
                print(f"  ⚠️ No per-node response time data for {name}, skipping")
                continue
            
            plt.figure(figsize=(12, 7))
            ep_nums = [e['episode'] for e in episodes]
            
            for key in rt_keys:
                node_name = key.replace('_avg_rt', '').replace('_', '-').upper()
                values = []
                for e in episodes:
                    val = e.get(key)
                    values.append(val if val is not None else 100.0)
                
                if len(values) > 5:
                    try:
                        smoothed = self.smooth_curve(values, 5)
                    except (TypeError, ValueError):
                        continue
                    valid_eps = ep_nums[:len(smoothed)]
                    plt.plot(valid_eps, smoothed, label=node_name, linewidth=2)
                else:
                    plt.plot(ep_nums, values, label=node_name, linewidth=2)
            
            plt.xlabel('Episode', fontsize=12)
            plt.ylabel('Avg Response Time (ms)', fontsize=12)
            plt.title(f'{name} - Per-Node Response Time Over Training', fontsize=14)
            plt.legend(loc='best', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            output_path = os.path.join(self.plots_dir, f'{name}_per_node_rt.png')
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  ✓ Saved: {name}_per_node_rt.png")
    # ========== END NEW ==========
    
    def generate_all_plots(self):
        print("\n" + "="*60)
        print("GENERATING PLOTS FROM CSV FILES")
        print("="*60)
        
        data = self.load_data()
        if not data:
            print("[!] No CSV files found. Please run training first.")
            return
        
        print(f"\nFound data for {len(data)} agents: {', '.join(data.keys())}")
        print("\nGenerating plots...")
        
        self.plot_learning_curves(data)
        self.plot_epsilon_decay(data)
        self.plot_final_comparison(data)
        self.plot_safety_comparison(data)
        self.plot_constraint_violations_over_time(data)
        self.plot_per_node_cpu(data)
        self.plot_per_node_mem(data)
        self.plot_per_node_response_time_comparison(data)
        self.plot_per_node_response_time(data)
        self.plot_per_node_load_balance(data)
        #self.plot_node_heatmap(data)
        self.plot_available_nodes_over_time(data)
        self.plot_scalability_adaptation(data)
        
        print(f"\n✓ All plots saved to '{self.plots_dir}/'")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', type=str, default='results')
    parser.add_argument('--plots-dir', type=str, default='results/plots')
    args = parser.parse_args()
    
    plotter = ResultsPlotter(args.results_dir, args.plots_dir)
    plotter.generate_all_plots()


if __name__ == "__main__":
    main()
