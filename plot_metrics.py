import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config import COLORS

def plot_reward_smoothed(file1, file2, file3, labels, window_size=5):
    """
    Plot smoothed total_reward only - independent figure.
    """
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    
    dataframes = [df1, df2, df3]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [COLORS.get(label, '#333333') for label in labels]
    line_styles = ['-', '-', '-']
    line_width = 1.5
    
    for i, (df, label, color, ls) in enumerate(zip(dataframes, labels, colors, line_styles)):
        episodes = df['episode'].values
        rewards = df['total_reward'].values
        
        reward_ma = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        ax.plot(episodes[window_size-1:], reward_ma, color=color, linestyle=ls,
                linewidth=line_width, label=label, alpha=0.9)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward (Smoothed)', fontsize=12)
    ax.set_title(f'Training Rewards Over Episodes (Moving Average, Window={window_size})', 
                  fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('reward_smoothed.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_success_smoothed(file1, file2, file3, labels, window_size=5):
    """
    Plot smoothed success_rate only - independent figure.
    """
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    
    dataframes = [df1, df2, df3]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = [COLORS.get(label, '#333333') for label in labels]
    line_styles = ['-', '-', '-']
    line_width = 1.5
    
    for i, (df, label, color, ls) in enumerate(zip(dataframes, labels, colors, line_styles)):
        episodes = df['episode'].values
        success_rates = df['success_rate'].values
        
        success_ma = np.convolve(success_rates, np.ones(window_size)/window_size, mode='valid')
        ax.plot(episodes[window_size-1:], success_ma, color=color, linestyle=ls,
                linewidth=line_width, label=label, alpha=0.9)
    
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Success Rate % (Smoothed)', fontsize=12)
    ax.set_title(f'Success Rate Over Episodes (Moving Average, Window={window_size})', 
                  fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([-5, 105])
    
    plt.tight_layout()
    plt.savefig('success_smoothed.png', dpi=150, bbox_inches='tight')
    plt.show()

def plot_per_node_memory(file1, file2, file3, labels, window_size=5):
    """
    Plot per-node average memory usage for each method.
    Creates separate figures for each method with all nodes.
    """
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    
    dataframes = [df1, df2, df3]
    colors = [COLORS.get(label, '#333333') for label in labels]
    
    # Node names
    node_names = ['minikube', 'minikube-m02', 'minikube-m03', 'minikube-m04']
    node_colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12']
    
    # Create figure for each method
    for method_idx, (df, label, method_color) in enumerate(zip(dataframes, labels, colors)):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = df['episode'].values
        
        for node_name, node_color in zip(node_names, node_colors):
            mem_col = f'{node_name}_avg_mem'
            
            if mem_col in df.columns:
                mem_values = df[mem_col].values
                mem_ma = np.convolve(mem_values, np.ones(window_size)/window_size, mode='valid')
                ax.plot(episodes[window_size-1:], mem_ma, color=node_color,
                       linewidth=1.5, label=node_name, alpha=0.85)
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Average Memory Usage (%)', fontsize=12)
        ax.set_title(f'{label} - Per-Node Memory Usage (Moving Average, Window={window_size})', 
                      fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([-5, 105])
        
        plt.tight_layout()
        plt.savefig(f'per_node_memory_{label.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
        plt.show()

def plot_per_node_memory_bar(file1, file2, file3, labels):
    """
    Create a bar chart comparing average memory usage per node across all agents.
    """
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(file3)
    
    dataframes = [df1, df2, df3]
    
    node_names = ['minikube', 'minikube-m02', 'minikube-m03', 'minikube-m04']
    
    # Calculate mean memory for each node across all agents
    data = {}
    for label, df in zip(labels, dataframes):
        mem_means = []
        for node in node_names:
            mem_col = f'{node}_avg_mem'
            if mem_col in df.columns:
                mem_means.append(df[mem_col].mean())
            else:
                mem_means.append(0)
        data[label] = mem_means
    
    # Create DataFrame for bar plot
    bar_df = pd.DataFrame(data, index=node_names)
    
    # Plot bar chart
    colors = [COLORS.get(label, '#333333') for label in labels]
    ax = bar_df.plot(kind='bar', figsize=(12, 6), color=colors, width=0.7, edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Nodes', fontsize=12)
    ax.set_ylabel('Average Memory Usage (%)', fontsize=12)
    ax.set_title('Per-Node Average Memory Usage Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim([0, 105])
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', fontsize=9, padding=3)
    
    plt.tight_layout()
    plt.savefig('per_node_memory_bar.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    # File paths
    eprs_file = "results/EPRS_results.csv"
    hs_drl_file = "results/HS-DRL_BEFORE_AFTER_results.csv"
    ppo_lrt_file = "results/PPO-LRT_results.csv"
    
    # Labels for the methods
    method_labels = ["EPRS", "HS-DRL", "PPO-LRT"]
    
    # Plot smoothed rewards only (independent figure)
    plot_reward_smoothed(eprs_file, hs_drl_file, ppo_lrt_file, 
                         method_labels, window_size=5)
    
    # Plot smoothed success rate only (independent figure)
    plot_success_smoothed(eprs_file, hs_drl_file, ppo_lrt_file, 
                          method_labels, window_size=5)
    
    # Plot per-node memory for each method separately
    plot_per_node_memory(eprs_file, hs_drl_file, ppo_lrt_file, 
                         method_labels, window_size=5)
    
    # Print statistics
    plot_per_node_memory_bar(eprs_file, hs_drl_file, ppo_lrt_file, 
                            method_labels)