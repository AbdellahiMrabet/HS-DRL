# train_single.py - Train a single agent and save results to CSV

import os
import sys
import time
import argparse
import subprocess
import agents
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import NUM_EPISODES, MAX_STEPS
from agents.dqn_agent import DQNAgent
from agents.rlsk_agent import RLSKAgent
from agents.eprs_agent import EPRSAgent
from agents.hsdrl_agent import HSDRLAgent
from agents.ppo_lrt_agent import PPOLRTAgent
from utils.csv_saver import CSVSaver
from utils.metrics_tracker import MetricsTracker


class SingleTrainer:
    def __init__(self, agent_name: str, num_episodes: int = None, max_steps: int = None):
        self.agent_name = agent_name
        self.num_episodes = num_episodes or NUM_EPISODES
        self.max_steps = max_steps or MAX_STEPS
        self.env = None
        self.agent = None
        self.csv_saver = None
        self.tracker = None
    
    def _init_agent(self, state_dim: int, action_dim: int):
        agents = {
            'DRS': DQNAgent,
            'RLSK': RLSKAgent,
            'EPRS': EPRSAgent,
            'HS-DRL': HSDRLAgent,
            'PPO-LRT': PPOLRTAgent}
        return agents[self.agent_name](state_dim, action_dim)
    
    def _get_epsilon(self) -> float:
        if hasattr(self.agent, 'epsilon'):
            return self.agent.epsilon
        return 0.0
    
    def train(self) -> dict:
        print(f"\n{'='*60}\nTraining: {self.agent_name}\nEpisodes: {self.num_episodes}\n{'='*60}")
        
        self.tracker = MetricsTracker()
        
        if self.agent_name == "HS-DRL":
            from environment.k8s_env import K8sEnv
            self.env = K8sEnv(tracker=self.tracker)
        elif self.agent_name == "EPRS":
            from environment.k8s_env_baseline import K8sEnvBaseline
            self.env = K8sEnvBaseline(tracker=self.tracker)
        else:
            from environment.k8s_env_baseline_no_penalty import K8sEnvBaseline
            self.env = K8sEnvBaseline(tracker=self.tracker)
        
        obs, _ = self.env.reset()
        self.agent = self._init_agent(len(obs), self.env.action_space.n)
        self.csv_saver = CSVSaver(self.agent_name)
        
        episode_rewards, episode_success_rates, episode_epsilons = [], [], []
        
        start_time = time.time()
        
        for episode in range(self.num_episodes):
            state, _ = self.env.reset()
            
            if hasattr(self.agent, '_adapt_to_new_state_dim'):
                self.agent._adapt_to_new_state_dim(len(state))
            
            episode_reward, episode_success_count = 0, 0
            
            for step in range(self.max_steps):
                action = self.agent.act(state, explore=True)
                next_state, reward, done, truncated, info = self.env.step(action)
                
                if hasattr(self.agent, 'update'):
                    self.agent.update(state, action, reward, next_state, done)
                
                if hasattr(self.agent, 'update_response_time') and action < self.env.num_nodes:
                    self.agent.update_response_time(action, info.get('api_response_time', 0))
                
                episode_reward += reward
                if info.get('success', False):
                    episode_success_count += 1
                
                if done or truncated:
                    break
                state = next_state
            
            success_rate = episode_success_count / self.max_steps
            current_epsilon = self._get_epsilon()
            
            episode_rewards.append(episode_reward)
            episode_success_rates.append(success_rate)
            episode_epsilons.append(current_epsilon)
            
            # Get episode summary from tracker
            episode_summary = self.tracker.get_episode_summary(episode + 1)
            episode_summary['epsilon'] = current_epsilon
            
            self.csv_saver.add_row(episode_summary)
            
            if (episode + 1) % 3 == 0 or episode == 0 or episode == self.num_episodes - 1:
                if current_epsilon != 0:
                    print(f"  Episode {episode+1:4d}: Reward={episode_reward:7.2f}, "
                      f"Success={success_rate*100:5.1f}%, Epsilon={current_epsilon:.4f}")
                else:
                    print(f"  Episode {episode+1:4d}: Reward={episode_reward:7.2f}, "
                      f"Success={success_rate*100:5.1f}%")
        
        #self.csv_saver.save_final()
        self.env.close()
        
        print(f"\n{self.agent_name} Summary:")
        print(f"  Time: {time.time() - start_time:.1f}s")
        print(f"  Success: {np.mean(episode_success_rates)*100:.1f}%")
        print(f"  Final Epsilon: {episode_epsilons[-1]:.4f}")
        
        return {'success_rate': np.mean(episode_success_rates) * 100}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, required=True, 
                        choices=['DRS', 'RLSK', 'EPRS', 'HS-DRL', 'PPO-LRT'])
    parser.add_argument('--episodes', type=int, default=None)
    args = parser.parse_args()
    
    try:
        subprocess.run(["kubectl", "get", "nodes"], capture_output=True, check=True)
        print("[✓] kubectl available")
    except:
        print("[!] Please ensure Minikube is running")
        return
    
    trainer = SingleTrainer(args.agent, args.episodes)
    trainer.train()
    print(f"\n✓ Results saved to results/{args.agent}_results.csv")


if __name__ == "__main__":
    main()