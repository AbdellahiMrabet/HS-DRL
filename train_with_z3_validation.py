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
from environment.pod_manager import PodDeploymentManager


class Z3ValidationTrainer:
    """
    Training wrapper that validates actions BEFORE and AFTER safety projection.
    """

    def __init__(self, agent_name: str, num_episodes: int = None, max_steps: int = None,
                 ou_theta: float = 0.15, ou_sigma: float = 0.5):
        self.agent_name = agent_name
        self.num_episodes = num_episodes or NUM_EPISODES
        self.max_steps = max_steps or MAX_STEPS
        self.ou_theta = ou_theta
        self.ou_sigma = ou_sigma
        self.env = None
        self.agent = None
        self.csv_saver = None
        self.tracker = None
        self.validator = None
        self.pod_manager = PodDeploymentManager()

    def _init_agent(self, state_dim: int, action_dim: int):
        return HSDRLAgent(state_dim, action_dim, name=self.agent_name,
                         ou_theta=self.ou_theta, ou_sigma=self.ou_sigma)

    def _get_exploration_stats(self) -> dict:
        """Get current exploration statistics (OU noise or epsilon)"""
        stats = {}
        if hasattr(self.agent, 'use_ou_noise') and self.agent.use_ou_noise:
            stats['exploration_type'] = 'OU Noise'
            if hasattr(self.agent, 'ou_noise') and self.agent.ou_noise is not None:
                stats['ou_noise_scale'] = self.agent.ou_noise.noise_scale
                stats['ou_theta'] = self.agent.ou_noise.theta
                stats['ou_sigma'] = self.agent.ou_noise.sigma
            else:
                stats['ou_noise_scale'] = 0
        else:
            stats['exploration_type'] = 'Epsilon-Greedy'
            stats['epsilon'] = self._get_epsilon()
        return stats

    def _get_epsilon(self) -> float:
        if hasattr(self.agent, 'epsilon'):
            return self.agent.epsilon
        return 0.0

    def train(self) -> dict:
        print(f"\n{'='*70}")
        print(f"Training: {self.agent_name} (BEFORE + AFTER Z3 Validation)")
        print(f"Episodes: {self.num_episodes}")

        # Show exploration configuration
        if hasattr(HSDRLAgent, 'use_ou_noise'):
            print(f"Exploration: OU Noise (θ={self.ou_theta}, σ={self.ou_sigma})")
        print(f"{'='*70}")

        # Reset validator for fresh run
        reset_validator()
        self.validator = get_validator()

        self.tracker = MetricsTracker()
        self.env = K8sEnv(tracker=self.tracker)

        obs, _ = self.env.reset()
        self.agent = self._init_agent(len(obs), self.env.action_space.n)
        self.csv_saver = CSVSaver(f"{self.agent_name}")

        episode_rewards, episode_success_rates = [], []

        start_time = time.time()

        for episode in range(self.num_episodes):
            state, _ = self.env.reset()

            # Reset OU noise for new episode (if using OU noise)
            if hasattr(self.agent, 'reset_noise'):
                self.agent.reset_noise()

            if hasattr(self.agent, '_adapt_to_new_state_dim'):
                self.agent._adapt_to_new_state_dim(len(state))

            episode_reward, episode_success_count, actual_steps = 0, 0, 0

            for step in range(self.max_steps):
                # Step 0: Agent proposes action
                raw_action = self.agent.act(state, explore=True)

                # Step 1: Clean old pods
                self.pod_manager.cleanup_old_pods()

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
                self.agent.update(state, raw_action, reward, next_state, done,\
                    info.get('safety_triggered', False))

                # Update response time tracking
                if safe_action < self.env.num_nodes - 1:
                    self.agent.update_response_time(safe_action, info.get('api_response_time', 0))

                episode_reward += reward
                actual_steps += 1
                if info.get('success', False):
                    episode_success_count += 1

                if done or truncated:
                    break
                state = next_state

            success_rate = episode_success_count / actual_steps if actual_steps > 0 else 0
            episode_rewards.append(episode_reward)
            episode_success_rates.append(success_rate)

            # Get Z3 stats for this episode
            z3_stats = self.validator.get_episode_stats()

            # Get exploration stats
            exploration_stats = self._get_exploration_stats()

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
                # Z3-based safety compliance rate (percentage of actions that were safe BEFORE projection)
                'safety_compliance_rate': z3_stats.get('z3_before_safety_rate', 100.0),
                'safety_compliance_rate_after': z3_stats.get('z3_after_safety_rate', 0.0),
                # Number of unsafe actions prevented by the shield
                'z3_unsafe_prevented': z3_stats.get('z3_unsafe_prevented', 0),
                'exploration_type': exploration_stats.get('exploration_type', 'Unknown'),
                'noise_scale': exploration_stats.get('ou_noise_scale', 0),
                'epsilon': exploration_stats.get('epsilon', self._get_epsilon())
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

                # Show exploration stats
                if exploration_stats.get('exploration_type') == 'OU Noise':
                    print(f"  🎲 OU Noise: scale={exploration_stats.get('ou_noise_scale', 0):.4f}, θ={exploration_stats.get('ou_theta', 0):.3f}, σ={exploration_stats.get('ou_sigma', 0):.3f}")
                else:
                    print(f"  🎲 Epsilon: {exploration_stats.get('epsilon', 0):.4f}")
                print(f"{'─'*50}")

        self.env.close()

        # Print final comparison report to console
        print(self.validator.get_comparison_report())

        # Print final exploration stats
        final_stats = self._get_exploration_stats()
        if final_stats.get('exploration_type') == 'OU Noise':
            print(f"\n{'='*50}")
            print("OU Noise Final Statistics:")
            print(f"  Final noise scale: {final_stats.get('ou_noise_scale', 0):.4f}")
            if hasattr(self.agent, 'ou_noise') and self.agent.ou_noise:
                print(f"  Final theta: {self.agent.ou_noise.theta:.3f}")
                print(f"  Final sigma: {self.agent.ou_noise.sigma:.3f}")
            print(f"  Total training steps: {self.agent.training_step}")
            print(f"{'='*50}")

        # Save Z3 report to separate text file
        with open('results/z3_validation_report.txt', 'w') as f:
            f.write(self.validator.get_comparison_report())

        print(f"\n✓ Results saved to results/{self.agent_name}_results.csv")
        print(f"✓ Z3 validation report saved to z3_validation_report.txt")

        return {'success_rate': np.mean(episode_success_rates) * 100}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=NUM_EPISODES, help='Number of episodes')
    parser.add_argument('--steps', type=int, default=50, help='Steps per episode')
    parser.add_argument('--ou-theta', type=float, default=0.15, help='OU noise mean reversion speed (default: 0.15)')
    parser.add_argument('--ou-sigma', type=float, default=0.5, help='OU noise volatility (default: 0.5)')
    args = parser.parse_args()

    try:
        subprocess.run(["kubectl", "get", "nodes"], capture_output=True, check=True)
        print("[✓] kubectl available")
    except:
        print("[!] Please ensure Minikube is running")
        return

    trainer = Z3ValidationTrainer("HS-DRL", args.episodes, args.steps,
                                  ou_theta=args.ou_theta, ou_sigma=args.ou_sigma)
    trainer.train()


if __name__ == "__main__":
    main()
