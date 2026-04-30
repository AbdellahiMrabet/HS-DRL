import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

# Read the CSV file
df = pd.read_csv('results/DRS_baseline_penalty_results.csv')

# Define metrics to plot (excluding non-numeric and redundant columns)
metrics_to_plot = [
    'total_reward',
    'success_rate',
    'avg_util',
    'avg_imbalance',
    'avg_response_time',
    'constraint_violations',
    'deployed_pods',
    'avg_available_nodes',
    'min_available_nodes',
    'safety_compliance_rate',
    'minikube_avg_cpu',
    'minikube-m02_avg_cpu',
    'minikube-m03_avg_cpu',
    'minikube-m04_avg_cpu',
    'minikube_avg_mem',
    'minikube-m02_avg_mem',
    'minikube-m03_avg_mem',
    'minikube-m04_avg_mem',
    'epsilon'
]

def apply_smoothing(data, window_size=11, poly_order=2):
    """
    Apply Savitzky-Golay smoothing filter to data
    """
    if len(data) < window_size:
        window_size = len(data) if len(data) % 2 == 1 else len(data) - 1
        if window_size < 3:
            return data
    
    try:
        # Ensure window_size is odd and less than data length
        if window_size % 2 == 0:
            window_size += 1
        if window_size > len(data):
            window_size = len(data) if len(data) % 2 == 1 else len(data) - 1
        
        if window_size >= 3:
            smoothed = savgol_filter(data, window_size, poly_order)
            return smoothed
        else:
            return data
    except:
        # Fallback to Gaussian smoothing
        try:
            smoothed = gaussian_filter1d(data, sigma=2)
            return smoothed
        except:
            return data

# Filter metrics that are numeric
numeric_metrics = []
for metric in metrics_to_plot:
    if metric in df.columns:
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(df[metric]):
            numeric_metrics.append(metric)
        else:
            print(f"Skipping non-numeric metric: '{metric}' (dtype: {df[metric].dtype})")

print(f"\nPlotting {len(numeric_metrics)} numeric metrics...")

# Create individual plots for each numeric metric
for metric in numeric_metrics:
    plt.figure(figsize=(14, 7))
    
    episodes = df['episode'].values
    values = df[metric].values
    
    # Remove any NaN values
    mask = ~np.isnan(values)
    episodes_clean = episodes[mask]
    values_clean = values[mask]
    
    # Create the plot with original data (semi-transparent)
    plt.plot(episodes_clean, values_clean, 'o', markersize=3, alpha=0.3, 
             label='Original Data', color='gray')
    
    # Apply smoothing
    if len(values_clean) > 10:
        # Try different smoothing windows based on data length
        if len(values_clean) > 100:
            smoothed = apply_smoothing(values_clean, window_size=21, poly_order=3)
        elif len(values_clean) > 50:
            smoothed = apply_smoothing(values_clean, window_size=15, poly_order=2)
        else:
            smoothed = apply_smoothing(values_clean, window_size=7, poly_order=2)
        
        # Plot smoothed curve
        plt.plot(episodes_clean, smoothed, linewidth=2.5, color='red', 
                label='Smoothed Trend', alpha=0.8)
        
        # Also add a rolling mean for comparison (optional)
        window = min(10, len(values_clean)//5)
        if window >= 3:
            rolling_mean = pd.Series(values_clean).rolling(window=window, center=True).mean()
            plt.plot(episodes_clean, rolling_mean, '--', linewidth=1.5, color='blue', 
                    alpha=0.6, label=f'Rolling Mean (window={window})')
    else:
        # If too few points, just plot the original
        plt.plot(episodes_clean, values_clean, 'o-', linewidth=1.5, markersize=4, 
                label='Original Data', color='blue')
    
    # Customize the plot
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    plt.title(f'{metric.replace("_", " ").title()} vs Episode (with Smoothing)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best')
    
    # Add some statistics as text on the plot
    try:
        mean_val = values_clean.mean()
        std_val = values_clean.std()
        max_val = values_clean.max()
        min_val = values_clean.min()
        
        stats_text = f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nMax: {max_val:.2f}\nMin: {min_val:.2f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    except Exception as e:
        print(f"Could not calculate statistics for {metric}: {e}")
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'results/plots/{metric}_plot_smoothed.png', dpi=150, bbox_inches='tight')
    
    # Display the plot
    plt.show()
    
    # Close the figure to free memory
    plt.close()

# Create a comprehensive summary plot with smoothed trends
print("\nCreating comprehensive summary plot...")

# Select a subset of key metrics for the summary plot
key_metrics = [
    'total_reward',
    'success_rate',
    'avg_util',
    'avg_imbalance',
    'avg_response_time',
    'deployed_pods'
]

# Filter key metrics to only include numeric ones
numeric_key_metrics = [m for m in key_metrics if m in df.columns and pd.api.types.is_numeric_dtype(df[m])]

if numeric_key_metrics:
    # Calculate number of rows needed
    n_metrics = len(numeric_key_metrics)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    # Create subplots for key metrics
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
    if n_rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, metric in enumerate(numeric_key_metrics):
        if idx < len(axes):
            episodes = df['episode'].values
            values = df[metric].values
            
            # Remove NaN values
            mask = ~np.isnan(values)
            episodes_clean = episodes[mask]
            values_clean = values[mask]
            
            # Plot original with low opacity
            axes[idx].plot(episodes_clean, values_clean, 'o', markersize=2, alpha=0.2, color='gray')
            
            # Add smoothed trend
            if len(values_clean) > 10:
                window_size = min(15, len(values_clean)//4)
                if window_size % 2 == 0:
                    window_size += 1
                if window_size >= 3:
                    smoothed = apply_smoothing(values_clean, window_size=window_size, poly_order=2)
                    axes[idx].plot(episodes_clean, smoothed, linewidth=2, color='red', alpha=0.8)
            
            axes[idx].set_xlabel('Episode', fontsize=10)
            axes[idx].set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
            axes[idx].set_title(metric.replace('_', ' ').title(), fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
    
    # Hide any unused subplots
    for idx in range(len(numeric_key_metrics), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Key Performance Metrics Over Episodes (with Smoothing)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('summary_metrics_plot_smoothed.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("No numeric key metrics found for summary plot")

# Create CPU utilization comparison plot with smoothing
cpu_metrics = ['minikube_avg_cpu', 'minikube-m02_avg_cpu', 'minikube-m03_avg_cpu', 'minikube-m04_avg_cpu']
numeric_cpu_metrics = [m for m in cpu_metrics if m in df.columns and pd.api.types.is_numeric_dtype(df[m])]

if len(numeric_cpu_metrics) > 0:
    plt.figure(figsize=(14, 7))
    colors = ['blue', 'green', 'orange', 'purple']
    
    for idx, metric in enumerate(numeric_cpu_metrics):
        episodes = df['episode'].values
        values = df[metric].values
        
        # Remove NaN values
        mask = ~np.isnan(values)
        episodes_clean = episodes[mask]
        values_clean = values[mask]
        
        # Plot original with low opacity
        plt.plot(episodes_clean, values_clean, 'o', markersize=2, alpha=0.2, 
                color=colors[idx % len(colors)])
        
        # Add smoothed trend
        if len(values_clean) > 10:
            window_size = min(15, len(values_clean)//4)
            if window_size % 2 == 0:
                window_size += 1
            if window_size >= 3:
                smoothed = apply_smoothing(values_clean, window_size=window_size, poly_order=2)
                plt.plot(episodes_clean, smoothed, linewidth=2.5, color=colors[idx % len(colors)], 
                        label=metric.replace('minikube', '').replace('_avg_cpu', ''), alpha=0.8)
        else:
            plt.plot(episodes_clean, values_clean, 'o-', linewidth=1.5, 
                    color=colors[idx % len(colors)], 
                    label=metric.replace('minikube', '').replace('_avg_cpu', ''))
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('CPU Usage (%)', fontsize=12)
    plt.title('CPU Usage Comparison Across Minikube Nodes (with Smoothing)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cpu_comparison_plot_smoothed.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("No numeric CPU metrics found")

# Create memory utilization comparison plot with smoothing
mem_metrics = ['minikube_avg_mem', 'minikube-m02_avg_mem', 'minikube-m03_avg_mem', 'minikube-m04_avg_mem']
numeric_mem_metrics = [m for m in mem_metrics if m in df.columns and pd.api.types.is_numeric_dtype(df[m])]

if len(numeric_mem_metrics) > 0:
    plt.figure(figsize=(14, 7))
    colors = ['blue', 'green', 'orange', 'purple']
    
    for idx, metric in enumerate(numeric_mem_metrics):
        episodes = df['episode'].values
        values = df[metric].values
        
        # Remove NaN values
        mask = ~np.isnan(values)
        episodes_clean = episodes[mask]
        values_clean = values[mask]
        
        # Plot original with low opacity
        plt.plot(episodes_clean, values_clean, 'o', markersize=2, alpha=0.2,
                color=colors[idx % len(colors)])
        
        # Add smoothed trend
        if len(values_clean) > 10:
            window_size = min(15, len(values_clean)//4)
            if window_size % 2 == 0:
                window_size += 1
            if window_size >= 3:
                smoothed = apply_smoothing(values_clean, window_size=window_size, poly_order=2)
                plt.plot(episodes_clean, smoothed, linewidth=2.5, color=colors[idx % len(colors)],
                        label=metric.replace('minikube', '').replace('_avg_mem', ''), alpha=0.8)
        else:
            plt.plot(episodes_clean, values_clean, 'o-', linewidth=1.5,
                    color=colors[idx % len(colors)],
                    label=metric.replace('minikube', '').replace('_avg_mem', ''))
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Memory Usage (MB)', fontsize=12)
    plt.title('Memory Usage Comparison Across Minikube Nodes (with Smoothing)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('memory_comparison_plot_smoothed.png', dpi=150, bbox_inches='tight')
    plt.show()
else:
    print("No numeric memory metrics found")

# Additional analysis: Plot success_rate vs total_reward with trend line
if ('success_rate' in df.columns and pd.api.types.is_numeric_dtype(df['success_rate']) and
    'total_reward' in df.columns and pd.api.types.is_numeric_dtype(df['total_reward'])):
    
    plt.figure(figsize=(12, 7))
    
    # Remove NaN values
    mask = ~(np.isnan(df['success_rate']) | np.isnan(df['total_reward']))
    success_rates = df['success_rate'][mask].values
    total_rewards = df['total_reward'][mask].values
    
    plt.scatter(success_rates, total_rewards, alpha=0.6, s=30, label='Data Points')
    
    # Add trend line
    z = np.polyfit(success_rates, total_rewards, 1)
    p = np.poly1d(z)
    
    # Sort for smooth line
    x_sorted = np.sort(success_rates)
    plt.plot(x_sorted, p(x_sorted), "r-", linewidth=2, 
             alpha=0.8, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    
    # Add confidence interval (optional)
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(success_rates, total_rewards)
    confidence_interval = 1.96 * std_err
    plt.fill_between(x_sorted, p(x_sorted) - confidence_interval, 
                     p(x_sorted) + confidence_interval, alpha=0.2, color='red')
    
    plt.xlabel('Success Rate (%)', fontsize=12)
    plt.ylabel('Total Reward', fontsize=12)
    plt.title('Success Rate vs Total Reward (with Trend Line)', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add correlation information
    corr_text = f'Correlation: r={r_value:.3f}\np-value: {p_value:.3e}'
    plt.text(0.05, 0.95, corr_text, transform=plt.gca().transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('success_vs_reward_with_trend.png', dpi=150, bbox_inches='tight')
    plt.show()

print("\n" + "="*60)
print("Plot generation with smoothing complete!")
print(f"Total episodes processed: {len(df)}")
print(f"Numeric metrics plotted: {len(numeric_metrics)}")
print("\nSmoothing techniques used:")
print("  - Savitzky-Golay filter (primary)")
print("  - Gaussian filter (fallback)")
print("  - Rolling mean (optional)")
print("\nFiles created:")
print(f"  - {len(numeric_metrics)} individual smoothed plots")
print("  - summary_metrics_plot_smoothed.png")
print("  - cpu_comparison_plot_smoothed.png")
print("  - memory_comparison_plot_smoothed.png")
print("  - success_vs_reward_with_trend.png")
print("="*60)
