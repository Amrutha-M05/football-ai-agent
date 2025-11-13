"""
Evaluation script for trained Football AI agents
"""
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
import gfootball.env as football_env
from tqdm import tqdm
import json

from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.hybrid_agent import HybridAgent, AdaptiveHybridAgent
from src.environment.wrappers import FootballEnvWrapper, RewardShaper
from src.utils.metrics import PerformanceTracker, compute_advanced_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate Football AI Agent')
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (uses model dir if not specified)')
    parser.add_argument('--scenario', type=str, default=None,
                       help='Evaluation scenario (uses training scenario if not specified)')
    parser.add_argument('--episodes', type=int, default=100,
                       help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render environment')
    parser.add_argument('--save-video', action='store_true',
                       help='Save gameplay videos')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device for inference')
    parser.add_argument('--deterministic', action='store_true',
                       help='Use deterministic policy (no exploration)')
    
    return parser.parse_args()


def load_agent_from_checkpoint(checkpoint_path, config, device):
    """Load agent from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Determine agent type from config
    agent_type = config.get('agent', {}).get('type', 'ppo')
    
    # Get dimensions from checkpoint or config
    if 'q_network' in checkpoint:
        state_dim = list(checkpoint['q_network'].values())[0].shape[1]
        action_dim = list(checkpoint['q_network'].values())[-1].shape[0]
        agent = DQNAgent(state_dim, action_dim, config['agent'])
    elif 'policy' in checkpoint:
        # Get dimensions from first layer
        first_layer_key = [k for k in checkpoint['policy'].keys() if 'shared.0.weight' in k][0]
        state_dim = checkpoint['policy'][first_layer_key].shape[1]
        
        # Get from last layer
        actor_keys = [k for k in checkpoint['policy'].keys() if 'actor.2.weight' in k]
        if actor_keys:
            action_dim = checkpoint['policy'][actor_keys[0]].shape[0]
        else:
            action_dim = 19  # Default for football
        
        if agent_type == 'hybrid' or agent_type == 'adaptive':
            agent_class = AdaptiveHybridAgent if agent_type == 'adaptive' else HybridAgent
            agent = agent_class(state_dim, action_dim, config['agent'])
        else:
            agent = PPOAgent(state_dim, action_dim, config['agent'])
    else:
        raise ValueError("Unknown checkpoint format")
    
    # Load checkpoint
    agent.load(checkpoint_path)
    
    return agent


def evaluate_agent(env, agent, num_episodes, render=False, save_video=False):
    """
    Evaluate agent performance.
    
    Returns:
        Dictionary with evaluation metrics
    """
    tracker = PerformanceTracker()
    episode_data = []
    
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_actions = []
        done = False
        
        while not done:
            # Get game state for hybrid agents
            game_state = env.get_game_state() if hasattr(env, 'get_game_state') else None
            
            # Select action (deterministic)
            if isinstance(agent, HybridAgent):
                action = agent.select_action(state, training=False, game_state=game_state)
            else:
                action = agent.select_action(state, training=False)
            
            episode_actions.append(action)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            episode_reward += reward
            episode_steps += 1
            state = next_state
        
        # Store episode data
        episode_info = {
            'episode': episode,
            'reward': episode_reward,
            'steps': episode_steps,
            'score': info.get('score_reward', 0),
            'actions': episode_actions
        }
        episode_data.append(episode_info)
        tracker.add_episode(episode_reward, episode_steps, info)
    
    # Compute statistics
    stats = tracker.get_statistics()
    
    # Add advanced metrics
    advanced_metrics = compute_advanced_metrics(episode_data)
    stats.update(advanced_metrics)
    
    return stats, episode_data


def compare_scenarios(agent, scenarios, num_episodes_per_scenario=20):
    """
    Evaluate agent across multiple scenarios.
    
    Args:
        agent: Trained agent
        scenarios: List of scenario names
        num_episodes_per_scenario: Episodes per scenario
        
    Returns:
        Dictionary with results per scenario
    """
    results = {}
    
    for scenario in scenarios:
        print(f"\nEvaluating on scenario: {scenario}")
        
        # Create environment
        env = football_env.create_environment(
            env_name=scenario,
            stacked=False,
            representation='simple115v2',
            render=False
        )
        env = FootballEnvWrapper(env)
        
        # Evaluate
        stats, _ = evaluate_agent(env, agent, num_episodes_per_scenario)
        results[scenario] = stats
        
        # Print summary
        print(f"  Mean Reward: {stats['mean_episode_reward']:.2f}")
        print(f"  Win Rate: {stats.get('win_rate', 0):.2%}")
        
        env.close()
    
    return results


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    model_path = Path(args.model_path)
    if args.config:
        config_path = Path(args.config)
    else:
        # Try to find config in model directory
        config_path = model_path.parent.parent / 'config.yaml'
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print("Warning: Config file not found, using defaults")
        config = {'agent': {}, 'environment': {}}
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    config['agent']['use_cuda'] = (device.type == 'cuda')
    
    print(f"Using device: {device}")
    
    # Determine scenario
    scenario = args.scenario or config.get('environment', {}).get('scenario', 'academy_empty_goal')
    
    # Create environment
    print(f"Creating environment: {scenario}")
    env_kwargs = {
        'render': args.render,
        'write_video': args.save_video,
        'write_goal_dumps': False,
        'write_full_episode_dumps': False
    }
    
    env = football_env.create_environment(
        env_name=scenario,
        stacked=False,
        representation='simple115v2',
        **env_kwargs
    )
    env = FootballEnvWrapper(env)
    env = RewardShaper(env, config.get('environment', {}))
    
    # Load agent
    print(f"Loading agent from {args.model_path}")
    agent = load_agent_from_checkpoint(args.model_path, config, device)
    
    print(f"\nAgent type: {type(agent).__name__}")
    print(f"Total training steps: {agent.total_steps}")
    print(f"Total episodes: {agent.episodes}")
    
    # Evaluate
    print(f"\nEvaluating for {args.episodes} episodes...")
    stats, episode_data = evaluate_agent(
        env, agent, args.episodes, 
        render=args.render, save_video=args.save_video
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total Episodes: {args.episodes}")
    print(f"Scenario: {scenario}")
    print(f"\nPerformance Metrics:")
    print(f"  Mean Episode Reward: {stats['mean_episode_reward']:.2f} ± {stats['std_episode_reward']:.2f}")
    print(f"  Mean Episode Length: {stats['mean_episode_length']:.1f} steps")
    print(f"  Win Rate: {stats.get('win_rate', 0):.2%}")
    print(f"  Mean Goals Scored: {stats.get('mean_goals_scored', 0):.2f}")
    print(f"  Mean Goals Conceded: {stats.get('mean_goals_conceded', 0):.2f}")
    
    if 'pass_completion_rate' in stats:
        print(f"\nTeamwork Metrics:")
        print(f"  Pass Completion Rate: {stats['pass_completion_rate']:.2%}")
        print(f"  Ball Possession %: {stats.get('possession_percentage', 0):.1f}%")
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save statistics
    with open(output_dir / 'evaluation_stats.json', 'w') as f:
        # Convert numpy types to native Python types
        stats_serializable = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                             for k, v in stats.items()}
        json.dump(stats_serializable, f, indent=2)
    
    # Save episode data
    with open(output_dir / 'episode_data.json', 'w') as f:
        json.dump(episode_data, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    # Multi-scenario evaluation (optional)
    if input("\nRun multi-scenario evaluation? (y/n): ").lower() == 'y':
        test_scenarios = [
            'academy_empty_goal',
            'academy_empty_goal_close',
            'academy_run_to_score',
            'academy_3_vs_1_with_keeper',
            '11_vs_11_easy_stochastic'
        ]
        print("\nRunning multi-scenario evaluation...")
        scenario_results = compare_scenarios(agent, test_scenarios, num_episodes_per_scenario=10)
        
        # Save multi-scenario results
        with open(output_dir / 'multi_scenario_results.json', 'w') as f:
            results_serializable = {
                scenario: {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                          for k, v in metrics.items()}
                for scenario, metrics in scenario_results.items()
            }
            json.dump(results_serializable, f, indent=2)
        
        print(f"Multi-scenario results saved to {output_dir}")
    
    # Close environment
    env.close()
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()