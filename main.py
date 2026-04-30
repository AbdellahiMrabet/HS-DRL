# main.py - Main entry point

import os
import time
import subprocess
import numpy as np
import torch
from config import NUM_EPISODES, MAX_STEPS, MATPLOTLIB_AVAILABLE, COLORS
from environment.k8s_env import K8sEnv
from agents.dqn_agent import DQNAgent
from agents.rlsk_agent import RLSKAgent
from agents.eprs_agent import EPRSAgent
from agents.hsdrl_agent import HSDRLAgent
from utils.metrics_tracker import MetricsTracker
from utils.plotting import (
    plot_episode_rewards, plot_success_rate, plot_resource_utilization,
    plot_load_imbalance, plot_response_time, plot_final_comparison,
    plot_improvement_summary, plot_disk_io_trend, plot_arrival_pods
)


def run_benchmark():
    """Run complete benchmark with all agents"""
    print("\n" + "=" * 70)
    print("BENCHMARK WITH ENHANCED NEURAL NETWORKS")
    print("HS-DRL (Ours) vs DRS, RLSK, EPRS")
    print("=" * 70)
    
    env = K8sEnv()
    obs, _ = env.reset()
    state_dim = len(obs)
    action_dim = env.action_space.n
    
    agents = {
        'DRS (DQN)': DQNAgent(state_dim, action_dim),
        'RLSK': RLSKAgent(state_dim, action_dim),
        'EPRS': EPRSAgent(state_dim, action_dim),
        'HS-DRL (Ours)': HSDRLAgent(state_dim, action_dim)
    }
    
    all_metrics = {}
    
    for name, agent in agents.items():
        print(f"\n[{name}]")
        metrics = MetricsTracker()
        
        for ep in range(NUM_EPISODES):
            state, _ = env.reset()
            ep_reward = 0
            ep_success_count = 0
            ep_utils = []
            ep_disk_ios = []
            ep_imbalances = []
            ep_responses = []
            
            for step in range(MAX_STEPS):
                start = time.perf_counter()
                action = agent.act(state, explore=True)
                decision_time = (time.perf_counter() - start) * 1000
                
                next_state, reward, done, truncated, info = env.step(action)
                api_response_time = info.get('api_response_time', 0)
                stack_size = info.get('stack_size', 0)
                total_response = decision_time + api_response_time
                
                # Update agent
                if hasattr(agent, 'update'):
                    agent.update(state, action, reward, next_state, done)
                
                # Update response time for HS-DRL
                if hasattr(agent, 'update_response_time') and action < env.num_nodes:
                    agent.update_response_time(action, api_response_time)
                
                ep_reward += reward
                success = info.get('success', False)
                if success:
                    ep_success_count += 1
                
                loads = [n['cpu'] for n in env.nodes]
                disk_ios = [n.get('disk_io', 0) for n in env.nodes]
                util = np.mean(loads)
                avg_disk_io = np.mean(disk_ios)
                imbalance = np.std(loads)
                projection = hasattr(agent, 'projection_count')
                
                metrics.step(reward, success, util, avg_disk_io, imbalance, 
                           total_response, projection, stack_size)
                ep_utils.append(util)
                ep_disk_ios.append(avg_disk_io)
                ep_imbalances.append(imbalance)
                ep_responses.append(total_response)
                
                if done or truncated:
                    break
                
                state = next_state
            
            success_rate = ep_success_count / MAX_STEPS
            metrics.episode(ep_reward, success_rate, 
                           np.mean(ep_utils), np.mean(ep_disk_ios),
                           np.mean(ep_imbalances), np.mean(ep_responses))
            
            if (ep + 1) % 3 == 0:
                print(f"  Ep {ep+1:2d}: Reward={np.mean(metrics.episode_rewards[-3:]):6.1f}, "
                      f"Success={success_rate*100:5.1f}%, Response={np.mean(ep_responses):5.1f}ms")
        
        all_metrics[name] = metrics.get()
        if hasattr(agent, 'get_stats'):
            stats = agent.get_stats()
            print(f"  Stats: {stats}")
        print(f"  Total Pods Deployed: {metrics.deployed_pods}")
        print(f"  Avg API Response: {metrics.get()['avg_response_time']:.1f}ms")
    
    env.close()
    return all_metrics


def print_results(metrics):
    """Print formatted results"""
    print("\n" + "=" * 80)
    print("FINAL RESULTS - ENHANCED NEURAL NETWORKS")
    print("=" * 80)
    
    print(f"\n{'Algorithm':<20} {'Success':<10} {'Safety':<10} {'CPU%':<10} "
          f"{'Disk%':<10} {'Imbalance':<10} {'API(ms)':<10} {'Pods':<8}")
    print("-" * 110)
    
    for name, m in metrics.items():
        print(f"{name:<20} {m['success_rate']:>5.1f}%     {m['safety_rate']:>5.1f}%     "
              f"{m['avg_util']:>5.1f}%     {m['avg_disk_io']:>5.1f}%     "
              f"{m['avg_imbalance']:>5.2f}%    {m['avg_response_time']:>5.1f}     {m['deployed_pods']:>4}")
    
    baseline_names = ['DRS (DQN)', 'RLSK', 'EPRS']
    baseline_response = np.mean([metrics[n]['avg_response_time'] for n in baseline_names])
    ours_response = metrics['HS-DRL (Ours)']['avg_response_time']
    
    print("\n" + "=" * 80)
    print(f"📊 KEY INSIGHT: Response Time = Actual K8s API Latency")
    print(f"   HS-DRL reduces API latency by {((baseline_response - ours_response) / baseline_response * 100):.1f}%")
    print("=" * 80)


def main():
    print("=" * 70)
    print("HS-DRL BENCHMARK - ENHANCED NEURAL NETWORKS")
    print(f"CPU threads: {torch.get_num_threads()}")
    print("=" * 70)
    
    # Check Kubernetes
    try:
        subprocess.run(["kubectl", "get", "nodes"], capture_output=True, check=True)
        print("[✓] kubectl available")
    except:
        print("[!] Please ensure Minikube is running")
        return
    
    # Run benchmark
    all_metrics = run_benchmark()
    
    # Print results
    print_results(all_metrics)
    
    # Generate plots
    if MATPLOTLIB_AVAILABLE:
        save_dir = 'benchmark_results'
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n[Generating plots...]")
        plot_episode_rewards(all_metrics, save_dir)
        plot_success_rate(all_metrics, save_dir)
        plot_resource_utilization(all_metrics, save_dir)
        plot_load_imbalance(all_metrics, save_dir)
        plot_response_time(all_metrics, save_dir)
        plot_final_comparison(all_metrics, save_dir)
        plot_improvement_summary(all_metrics, save_dir)
        plot_disk_io_trend(all_metrics, save_dir)
        plot_arrival_pods(all_metrics, save_dir)
        
        print(f"\n✓ All plots saved to '{save_dir}/'")
    
    print("\n✓ Benchmark complete!")


if __name__ == "__main__":
    main()
