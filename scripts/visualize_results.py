"""
Visualize training results and create plots
"""
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize training results')
    
    parser.add_argument('--log-dir', type=str, required=True,
                       help='Directory containing training logs')
    parser.add_argument('--output-dir', type=str, default='plots',
                       help='Output directory for plots')
    parser.add_argument('--format', type=str, default='png',
                       choices=['png', 'pdf', 'svg'],
                       help='Output format for plots')
    
    return parser.parse_args()


def load_episode_logs(log_dir):
    """Load episode logs from CSV."""
    episodes_path = Path(log_dir) / 'episodes.csv'
    if episodes_path.exists():
        return pd.read_csv(episodes_path)
    return None


def load_evaluation_logs(log_dir):
    """Load evaluation logs from CSV."""
    eval_path = Path(log_dir) / 'evaluations.csv'
    if eval_path.exists():
        return pd.read_csv(eval_path)
    return None


def plot_training_curve(df, output_path):
    """
    Plot training reward curve.
    
    Args:
        df: DataFrame with episode data
        output_path: Output file path
    """
    plt.figure(figsize=(12, 6))
    
    # Raw rewards
    plt.plot(df['episode'], df['reward'], alpha=0.3, label='Episode Reward')
    
    # Moving average
    window = 100
    if len(df) >= window:
        moving_avg = df['reward'].rolling(window=window, min_periods=1).mean()
        plt.plot(df['episode'], moving_avg, linewidth=2, 
                label=f'{window}-Episode Moving Average', color='red')
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('Training Progress', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved training curve to {output_path}")


def plot_episode_length(df, output_path):
    """
    Plot episode length over time.
    
    Args:
        df: DataFrame with episode data
        output_path: Output file path
    """
    plt.figure(figsize=(12, 6))
    
    plt.plot(df['episode'], df['steps'], alpha=0.4)
    
    # Moving average
    window = 50
    if len(df) >= window:
        moving_avg = df['steps'].rolling(window=window, min_periods=1).mean()
        plt.plot(df['episode'], moving_avg, linewidth=2, 
                label=f'{window}-Episode Moving Average', color='orange')
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Episode Length (steps)', fontsize=12)
    plt.title('Episode Length Over Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved episode length plot to {output_path}")


def plot_reward_distribution(df, output_path):
    """
    Plot reward distribution.
    
    Args:
        df: DataFrame with episode data
        output_path: Output file path
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df['reward'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(df['reward'].mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {df["reward"].mean():.2f}')
    axes[0].axvline(df['reward'].median(), color='green', linestyle='--', 
                   linewidth=2, label=f'Median: {df["reward"].median():.2f}')
    axes[0].set_xlabel('Reward', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Reward Distribution', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(df['reward'], vert=True)
    axes[1].set_ylabel('Reward', fontsize=12)
    axes[1].set_title('Reward Statistics', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved reward distribution to {output_path}")


def plot_evaluation_metrics(df, output_path):
    """
    Plot evaluation metrics over time.
    
    Args:
        df: DataFrame with evaluation data
        output_path: Output file path
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Mean reward with confidence interval
    axes[0].plot(df['episode'], df['mean_reward'], marker='o', 
                linewidth=2, markersize=6, label='Mean Reward')
    axes[0].fill_between(df['episode'], 
                         df['mean_reward'] - df['std_reward'],
                         df['mean_reward'] + df['std_reward'],
                         alpha=0.3, label='±1 Std Dev')
    axes[0].set_xlabel('Episode', fontsize=12)
    axes[0].set_ylabel('Mean Reward', fontsize=12)
    axes[0].set_title('Evaluation Performance', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Win rate
    axes[1].plot(df['episode'], df['win_rate'] * 100, marker='s',
                linewidth=2, markersize=6, color='green', label='Win Rate')
    axes[1].set_xlabel('Episode', fontsize=12)
    axes[1].set_ylabel('Win Rate (%)', fontsize=12)
    axes[1].set_title('Win Rate Over Training', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 105])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved evaluation metrics to {output_path}")


def plot_comparison(log_dirs, labels, output_path):
    """
    Compare multiple training runs.
    
    Args:
        log_dirs: List of log directories
        labels: List of labels for each run
        output_path: Output file path
    """
    plt.figure(figsize=(14, 7))
    
    for log_dir, label in zip(log_dirs, labels):
        df = load_episode_logs(log_dir)
        if df is not None:
            # Compute moving average
            window = 100
            if len(df) >= window:
                moving_avg = df['reward'].rolling(window=window, min_periods=1).mean()
                plt.plot(df['episode'], moving_avg, linewidth=2, label=label)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Reward (100-episode MA)', fontsize=12)
    plt.title('Training Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved comparison plot to {output_path}")


def create_summary_statistics(df):
    """
    Create summary statistics table.
    
    Args:
        df: DataFrame with episode data
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        'Total Episodes': len(df),
        'Mean Reward': df['reward'].mean(),
        'Std Reward': df['reward'].std(),
        'Min Reward': df['reward'].min(),
        'Max Reward': df['reward'].max(),
        'Median Reward': df['reward'].median(),
        'Mean Episode Length': df['steps'].mean(),
        'Total Steps': df['steps'].sum(),
    }
    
    # Calculate win rate if score available
    if 'score' in df.columns:
        stats['Win Rate (%)'] = (df['score'] > 0).mean() * 100
        stats['Mean Score'] = df['score'].mean()
    
    return stats


def print_summary_table(stats):
    """Print summary statistics as formatted table."""
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY STATISTICS")
    print("=" * 60)
    
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:.<45} {value:.2f}")
        else:
            print(f"{key:.<45} {value}")
    
    print("=" * 60 + "\n")


def main():
    """Main visualization function."""
    args = parse_args()
    
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nVisualizing results from: {log_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Load data
    print("\nLoading logs...")
    episode_df = load_episode_logs(log_dir)
    eval_df = load_evaluation_logs(log_dir)
    
    if episode_df is None:
        print("Error: No episode logs found!")
        return
    
    print(f"✓ Loaded {len(episode_df)} episodes")
    if eval_df is not None:
        print(f"✓ Loaded {len(eval_df)} evaluation points")
    
    # Create plots
    print("\nGenerating plots...")
    
    # Training curve
    plot_training_curve(
        episode_df,
        output_dir / f'training_curve.{args.format}'
    )
    
    # Episode length
    plot_episode_length(
        episode_df,
        output_dir / f'episode_length.{args.format}'
    )
    
    # Reward distribution
    plot_reward_distribution(
        episode_df,
        output_dir / f'reward_distribution.{args.format}'
    )
    
    # Evaluation metrics
    if eval_df is not None:
        plot_evaluation_metrics(
            eval_df,
            output_dir / f'evaluation_metrics.{args.format}'
        )
    
    # Summary statistics
    stats = create_summary_statistics(episode_df)
    print_summary_table(stats)
    
    # Save statistics to JSON
    stats_path = output_dir / 'summary_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved statistics to {stats_path}")
    
    print("\n" + "=" * 60)
    print("Visualization complete!")
    print(f"All plots saved to {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()