"""
Performance metrics and tracking for Football AI Agent
"""
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict


class PerformanceTracker:
    """Track and compute performance metrics during training/evaluation."""
    
    def __init__(self):
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_scores: List[int] = []
        self.episode_infos: List[Dict] = []
        
        # Advanced metrics
        self.goals_scored: List[int] = []
        self.goals_conceded: List[int] = []
        self.wins: int = 0
        self.losses: int = 0
        self.draws: int = 0
        
    def add_episode(self, reward: float, length: int, info: Dict[str, Any]):
        """
        Add episode results.
        
        Args:
            reward: Total episode reward
            length: Episode length in steps
            info: Info dictionary from environment
        """
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.episode_infos.append(info)
        
        # Extract score information
        score = info.get('score_reward', 0)
        self.episode_scores.append(score)
        
        # Extract detailed scoring if available
        our_score = info.get('score', [0, 0])[0]
        opp_score = info.get('score', [0, 0])[1]
        
        self.goals_scored.append(our_score)
        self.goals_conceded.append(opp_score)
        
        # Win/loss/draw tracking
        if our_score > opp_score:
            self.wins += 1
        elif our_score < opp_score:
            self.losses += 1
        else:
            self.draws += 1
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Compute and return performance statistics.
        
        Returns:
            Dictionary of statistics
        """
        if not self.episode_rewards:
            return {}
        
        total_episodes = len(self.episode_rewards)
        
        stats = {
            # Basic metrics
            'total_episodes': total_episodes,
            'mean_episode_reward': np.mean(self.episode_rewards),
            'std_episode_reward': np.std(self.episode_rewards),
            'min_episode_reward': np.min(self.episode_rewards),
            'max_episode_reward': np.max(self.episode_rewards),
            'median_episode_reward': np.median(self.episode_rewards),
            
            # Episode length
            'mean_episode_length': np.mean(self.episode_lengths),
            'std_episode_length': np.std(self.episode_lengths),
            
            # Scoring metrics
            'mean_goals_scored': np.mean(self.goals_scored) if self.goals_scored else 0,
            'mean_goals_conceded': np.mean(self.goals_conceded) if self.goals_conceded else 0,
            'total_goals_scored': sum(self.goals_scored) if self.goals_scored else 0,
            'total_goals_conceded': sum(self.goals_conceded) if self.goals_conceded else 0,
            
            # Win rate metrics
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'win_rate': self.wins / total_episodes if total_episodes > 0 else 0,
            'loss_rate': self.losses / total_episodes if total_episodes > 0 else 0,
            'draw_rate': self.draws / total_episodes if total_episodes > 0 else 0,
        }
        
        # Compute goal difference
        if self.goals_scored and self.goals_conceded:
            goal_diff = np.array(self.goals_scored) - np.array(self.goals_conceded)
            stats['mean_goal_difference'] = np.mean(goal_diff)
            stats['positive_goal_diff_rate'] = np.mean(goal_diff > 0)
        
        return stats
    
    def get_recent_stats(self, window: int = 100) -> Dict[str, float]:
        """
        Get statistics for recent episodes.
        
        Args:
            window: Number of recent episodes to consider
            
        Returns:
            Dictionary of recent statistics
        """
        if not self.episode_rewards:
            return {}
        
        recent_rewards = self.episode_rewards[-window:]
        recent_lengths = self.episode_lengths[-window:]
        
        return {
            'recent_mean_reward': np.mean(recent_rewards),
            'recent_std_reward': np.std(recent_rewards),
            'recent_mean_length': np.mean(recent_lengths),
        }
    
    def reset(self):
        """Reset all tracked metrics."""
        self.episode_rewards.clear()
        self.episode_lengths.clear()
        self.episode_scores.clear()
        self.episode_infos.clear()
        self.goals_scored.clear()
        self.goals_conceded.clear()
        self.wins = 0
        self.losses = 0
        self.draws = 0


def compute_advanced_metrics(episode_data: List[Dict]) -> Dict[str, float]:
    """
    Compute advanced metrics from episode data.
    
    Args:
        episode_data: List of episode dictionaries with actions and outcomes
        
    Returns:
        Dictionary of advanced metrics
    """
    if not episode_data:
        return {}
    
    metrics = {}
    
    # Action distribution analysis
    all_actions = []
    for ep in episode_data:
        all_actions.extend(ep.get('actions', []))
    
    if all_actions:
        action_counts = defaultdict(int)
        for action in all_actions:
            action_counts[action] += 1
        
        # Action diversity (entropy)
        total_actions = len(all_actions)
        action_probs = np.array([count / total_actions for count in action_counts.values()])
        action_entropy = -np.sum(action_probs * np.log(action_probs + 1e-10))
        
        metrics['action_entropy'] = action_entropy
        metrics['unique_actions_used'] = len(action_counts)
        
        # Most common actions
        sorted_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)
        metrics['most_common_action'] = sorted_actions[0][0] if sorted_actions else -1
        metrics['most_common_action_freq'] = sorted_actions[0][1] / total_actions if sorted_actions else 0
    
    # Episode consistency
    rewards = [ep['reward'] for ep in episode_data]
    if len(rewards) > 1:
        # Coefficient of variation
        metrics['reward_cv'] = np.std(rewards) / (np.abs(np.mean(rewards)) + 1e-10)
        
        # Reward trend (simple linear regression slope)
        x = np.arange(len(rewards))
        slope = np.polyfit(x, rewards, 1)[0]
        metrics['reward_trend'] = slope
    
    # Success rate in recent episodes
    recent_window = min(20, len(episode_data))
    recent_scores = [ep.get('score', 0) for ep in episode_data[-recent_window:]]
    metrics['recent_success_rate'] = np.mean(np.array(recent_scores) > 0)
    
    return metrics


def compute_pass_completion_rate(episode_actions: List[int], pass_action_ids: List[int] = [12, 13]) -> float:
    """
    Estimate pass completion rate from action sequence.
    This is a simplified heuristic.
    
    Args:
        episode_actions: List of actions taken in episode
        pass_action_ids: Action IDs corresponding to passing
        
    Returns:
        Estimated pass completion rate
    """
    pass_attempts = sum(1 for action in episode_actions if action in pass_action_ids)
    
    if pass_attempts == 0:
        return 0.0
    
    # Simplified heuristic: assume pass is completed if followed by our team's action
    # In practice, would need actual game state information
    successful_passes = 0
    for i in range(len(episode_actions) - 1):
        if episode_actions[i] in pass_action_ids:
            # Check if next action suggests we maintained possession
            # This is a very rough approximation
            next_action = episode_actions[i + 1]
            if next_action not in [0, 1, 2, 3]:  # Not just movement
                successful_passes += 1
    
    return successful_passes / pass_attempts if pass_attempts > 0 else 0.0


class RollingMetrics:
    """Track rolling/moving averages of metrics."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.values: Dict[str, List[float]] = defaultdict(list)
    
    def update(self, metric_name: str, value: float):
        """
        Update metric with new value.
        
        Args:
            metric_name: Name of the metric
            value: New value to add
        """
        self.values[metric_name].append(value)
        
        # Keep only recent values
        if len(self.values[metric_name]) > self.window_size:
            self.values[metric_name].pop(0)
    
    def get_mean(self, metric_name: str) -> float:
        """Get rolling mean of metric."""
        if metric_name not in self.values or not self.values[metric_name]:
            return 0.0
        return np.mean(self.values[metric_name])
    
    def get_std(self, metric_name: str) -> float:
        """Get rolling standard deviation of metric."""
        if metric_name not in self.values or not self.values[metric_name]:
            return 0.0
        return np.std(self.values[metric_name])
    
    def get_trend(self, metric_name: str) -> str:
        """
        Get trend direction of metric.
        
        Returns:
            'improving', 'declining', or 'stable'
        """
        if metric_name not in self.values or len(self.values[metric_name]) < 2:
            return 'stable'
        
        recent = self.values[metric_name]
        mid_point = len(recent) // 2
        
        early_mean = np.mean(recent[:mid_point])
        late_mean = np.mean(recent[mid_point:])
        
        threshold = 0.05 * abs(early_mean)  # 5% threshold
        
        if late_mean > early_mean + threshold:
            return 'improving'
        elif late_mean < early_mean - threshold:
            return 'declining'
        else:
            return 'stable'


def format_metrics_table(metrics: Dict[str, float]) -> str:
    """
    Format metrics dictionary as a pretty table.
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        Formatted string table
    """
    if not metrics:
        return "No metrics available"
    
    # Group metrics by category
    categories = {
        'Reward': ['mean_episode_reward', 'std_episode_reward', 'max_episode_reward'],
        'Performance': ['win_rate', 'mean_goals_scored', 'mean_goals_conceded'],
        'Efficiency': ['mean_episode_length', 'action_entropy'],
        'Trend': ['reward_trend', 'recent_success_rate']
    }
    
    lines = ["Metrics Summary", "=" * 60]
    
    for category, metric_names in categories.items():
        category_metrics = {k: v for k, v in metrics.items() if k in metric_names}
        if category_metrics:
            lines.append(f"\n{category}:")
            for name, value in category_metrics.items():
                formatted_name = name.replace('_', ' ').title()
                if isinstance(value, float):
                    if 'rate' in name.lower() or 'percentage' in name.lower():
                        lines.append(f"  {formatted_name:.<40} {value:.2%}")
                    else:
                        lines.append(f"  {formatted_name:.<40} {value:.3f}")
                else:
                    lines.append(f"  {formatted_name:.<40} {value}")
    
    return "\n".join(lines)