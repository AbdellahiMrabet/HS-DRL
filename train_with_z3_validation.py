# train_with_z3_validation.py
# Training with BEFORE and AFTER Z3 validation

import os
import sys
import time
import argparse
import subprocess
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import NUM_EPISODES, MAX_STEPS
from agents.hsdrl_agent import HSDRLAgent
from environment.k8s_env import K8sEnv
from utils.csv_saver import CSVSaver
from utils.metrics_tracker import MetricsTracker
from verification.z3_validator import get_validator, reset_validator


class Z3ValidationTrainer:
    """
    Training wrapper that validates actions BEFORE and AFTER safety projection.
    """
    
    def __init__(self, agent_name: str, num_episodes: int = None, max_steps: int = None):
        self.agent_name = agent_name
        self.num_episodes = num_episodes or NUM_EPISODES
        self.max_steps = max_steps or MAX_STEPS
        self.env = None
        self.agent = None
        self.csv_saver = None
        self.tracker = None
        self.validator = None
    
    def _init_agent(self, state_dim: int, action_dim: int):
        return HSDRLAgent(state_dim, action_dim, name=self.agent_name)
    
    def _get_epsilon(self) -> float:
        if hasattr(self.agent, 'epsilon'):
            return self.agent.epsilon
        return 0.0
    
    def train(self) -> dict:
        print(f"\n{'='*70}")
        print(f"Training: {self.agent_name} (BEFORE + AFTER Z3 Validation)")
        print(f"Episodes: {self.num_episodes}")
        print(f"{'='*70}")
        
        # Reset validator for fresh run
        reset_validator()
        self.validator = get_validator()
        
        self.tracker = MetricsTracker()
        self.env = K8sEnv(tracker=self.tracker)
        
        obs, _ = self.env.reset()
        self.agent = self._init_agent(len(obs), self.env.action_space.n)
        self.csv_saver = CSVSaver(f"{self.agent_name}_BEFORE_AFTER")
        
        episode_rewards, episode_success_rates = [], []
        
        start_time = time.time()
        
        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            
            if hasattr(self.agent, '_adapt_to_new_state_dim'):
                self.agent._adapt_to_new_state_dim(len(state))
            
            episode_reward, episode_success_count = 0, 0
            
            for step in range(self.max_steps):
                # Step 1: Agent proposes action
                raw_action = self.agent.act(state, explore=True)
                
                # Step 2: VALIDATE BEFORE PROJECTION - Shows what WOULD happen without shield
                before_is_safe, before_reason = self.validator.validate_before_projection(self.env, raw_action)
                
                # Optional: Print only unsafe actions to reduce output
                if not before_is_safe:
                    print(f"  🔴 BEFORE: Raw action {raw_action} is UNSAFE: {before_reason}")
                    
                # Get the projected action from the environment
                safe_action, projection_triggered = self.env._hierarchical_safety_projection(raw_action)
                
                # Record projection in validator
                if projection_triggered:
                    self.validator.record_projection(raw_action, safe_action)
                    # Step 4: VALIDATE AFTER PROJECTION - Shows what ACTUALLY executes
                    after_is_safe, after_reason = self.validator.validate_after_projection(self.env, safe_action)
                    
                    if not after_is_safe:
                        print(f"  🔴🔴 AFTER: Projected action {safe_action} is STILL UNSAFE: {after_reason}")
                
                # Step 3: Execute action (safety projection happens here!)
                next_state, reward, done, truncated, info = self.env.step(raw_action)
                # Update agent
                self.agent.update(state, raw_action, reward, next_state, done)
                
                # Update response time tracking
                if safe_action < self.env.num_nodes:
                    self.agent.update_response_time(safe_action, info.get('api_response_time', 0))
                
                episode_reward += reward
                if info.get('success', False):
                    episode_success_count += 1
                
                if done or truncated:
                    break
                state = next_state
            
            success_rate = episode_success_count / self.max_steps
            episode_rewards.append(episode_reward)
            episode_success_rates.append(success_rate)
            
            # Get Z3 stats for this episode
            z3_stats = self.validator.get_episode_stats()
            
            # Build CSV row - keep only selected fields
            episode_summary = {
                'episode': episode + 1,
                'total_reward': episode_reward,
                'success_rate': success_rate * 100,
                'avg_util': self.tracker.episode_utils[-1] if self.tracker.episode_utils else 0,
                'avg_imbalance': self.tracker.episode_imbalances[-1] if self.tracker.episode_imbalances else 0,
                'avg_response_time': self.tracker.episode_responses[-1] if self.tracker.episode_responses else 0,
                'deployed_pods': self.tracker.deployed_pods,
                'avg_available_nodes': self.tracker.episode_avg_available[-1] if self.tracker.episode_avg_available else 4,
                'min_available_nodes': self.tracker.episode_min_available[-1] if self.tracker.episode_min_available else 4,
                # Z3-based safety compliance rate (percentage of actions that were safe AFTER projection)
                'safety_compliance_rate': z3_stats.get('z3_after_safety_rate', 100.0),
                # Number of unsafe actions prevented by the shield
                'z3_unsafe_prevented': z3_stats.get('z3_unsafe_prevented', 0),
                'epsilon': self._get_epsilon()
            }
            
            # Add per-node stats from tracker
            for name in self.tracker.per_node_cpu.keys():
                if self.tracker.per_node_cpu[name]:
                    episode_summary[f'{name}_avg_cpu'] = np.mean(self.tracker.per_node_cpu[name]) * 100
                if self.tracker.per_node_mem[name]:
                    episode_summary[f'{name}_avg_mem'] = np.mean(self.tracker.per_node_mem[name]) * 100
            
            for name in self.tracker.per_node_response_time.keys():
                if self.tracker.per_node_response_time[name]:
                    episode_summary[f'{name}_avg_rt'] = np.mean(self.tracker.per_node_response_time[name])
            
            self.csv_saver.add_row(episode_summary)
            
            # Reset validator for next episode
            self.validator.reset_episode(episode)
            
            if (episode + 1) % 5 == 0 or episode == 0 or episode == self.num_episodes - 1:
                print(f"\n{'─'*50}")
                print(f"Episode {episode+1:3d} Summary:")
                print(f"  Reward: {episode_reward:7.2f} | Success: {success_rate*100:5.1f}%")
                print(f"  Z3 BEFORE: Safe={z3_stats['z3_before_safe']:3d}, Unsafe={z3_stats['z3_before_unsafe']:3d} ({z3_stats['z3_before_safety_rate']:.1f}%)")
                print(f"  Z3 AFTER:  Safe={z3_stats['z3_after_safe']:3d}, Unsafe={z3_stats['z3_after_unsafe']:3d} ({z3_stats['z3_after_safety_rate']:.1f}%)")
                print(f"  🛡️ Shield prevented {z3_stats['z3_unsafe_prevented']} unsafe actions ({z3_stats['z3_shield_effectiveness']:.1f}% effective)")
                print(f"{'─'*50}")
        
        self.env.close()
        
        # Print final comparison report to console
        print(self.validator.get_comparison_report())
        
        # Save Z3 report to separate text file
        with open('z3_validation_report.txt', 'w') as f:
            f.write(self.validator.get_comparison_report())
        
        print(f"\n✓ Results saved to results/{self.agent_name}_BEFORE_AFTER_results.csv")
        print(f"✓ Z3 validation report saved to z3_validation_report.txt")
        
        return {'success_rate': np.mean(episode_success_rates) * 100}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=50, help='Number of episodes')
    parser.add_argument('--steps', type=int, default=50, help='Steps per episode')
    args = parser.parse_args()
    
    try:
        subprocess.run(["kubectl", "get", "nodes"], capture_output=True, check=True)
        print("[✓] kubectl available")
    except:
        print("[!] Please ensure Minikube is running")
        return
    
    trainer = Z3ValidationTrainer("HS-DRL", args.episodes, args.steps)
    trainer.train()


if __name__ == "__main__":
    main()