"""
Main training script for Football AI Agent
"""
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
import gfootball.env as football_env
from datetime import datetime
from tqdm import tqdm

from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.hybrid_agent import HybridAgent, AdaptiveHybridAgent
from src.environment.wrappers import FootballEnvWrapper, RewardShaper
from src.utils.logger import TrainingLogger
from src.utils.metrics import PerformanceTracker


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Football AI Agent')
    
    # Agent configuration
    parser.add_argument('--agent', type=str, default='ppo',
                       choices=['dqn', 'ppo', 'hybrid', 'adaptive'],
                       help='Agent type to train')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config YAML file')
    
    # Environment configuration
    parser.add_argument('--scenario', type=str, default='academy_empty_goal',
                       help='Football environment scenario')
    parser.add_argument('--multi-agent', action='store_true',
                       help='Enable multi-agent mode')
    
    # Training configuration
    parser.add_argument('--episodes', type=int, default=10000,
                       help='Number of training episodes')
    parser.add_argument('--max-steps', type=int, default=3000,
                       help='Maximum steps per episode')
    parser.add_argument('--eval-frequency', type=int, default=100,
                       help='Evaluate every N episodes')
    parser.add_argument('--save-frequency', type=int, default=500,
                       help='Save checkpoint every N episodes')
    
    # LightGBM configuration
    parser.add_argument('--use-lgbm', action='store_true',
                       help='Use LightGBM advisor (for hybrid agents)')
    parser.add_argument('--lgbm-model', type=str, default=None,
                       help='Path to pre-trained LightGBM model')
    parser.add_argument('--lgbm-weight', type=float, default=0.3,
                       help='Weight for LightGBM advisor influence')
    
    # Curriculum learning
    parser.add_argument('--curriculum', action='store_true',
                       help='Enable curriculum learning')
    parser.add_argument('--start-scenario', type=str, default='academy_empty_goal',
                       help='Starting scenario for curriculum')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--render', action='store_true',
                       help='Render environment')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    return parser.parse_args()


def load_config(args):
    """Load configuration from file and command line args."""
    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            'agent': {},
            'training': {},
            'environment': {}
        }
    
    # Override with command line arguments
    config['agent']['type'] = args.agent
    config['training']['episodes'] = args.episodes
    config['training']['max_steps'] = args.max_steps
    config['training']['eval_frequency'] = args.eval_frequency
    config['training']['save_frequency'] = args.save_frequency
    config['environment']['scenario'] = args.scenario
    config['environment']['multi_agent'] = args.multi_agent
    config['environment']['render'] = args.render
    
    # LightGBM config
    if args.use_lgbm:
        config['agent']['use_lgbm'] = True
        config['agent']['lgbm_weight'] = args.lgbm_weight
        if args.lgbm_model:
            config['agent']['lgbm_model_path'] = args.lgbm_model
    
    # Device configuration
    if args.device == 'auto':
        config['agent']['use_cuda'] = torch.cuda.is_available()
    else:
        config['agent']['use_cuda'] = (args.device == 'cuda')
    
    return config


def create_agent(agent_type, state_dim, action_dim, config):
    """Create agent based on type."""
    agent_config = config.get('agent', {})
    
    if agent_type == 'dqn':
        return DQNAgent(state_dim, action_dim, agent_config)
    elif agent_type == 'ppo':
        return PPOAgent(state_dim, action_dim, agent_config)
    elif agent_type == 'hybrid':
        return HybridAgent(state_dim, action_dim, agent_config)
    elif agent_type == 'adaptive':
        return AdaptiveHybridAgent(state_dim, action_dim, agent_config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def train_episode(env, agent, max_steps, logger, episode_num):
    """Train for one episode."""
    state = env.reset()
    episode_reward = 0
    episode_steps = 0
    
    for step in range(max_steps):
        # Select action
        game_state = env.get_game_state() if hasattr(env, 'get_game_state') else None
        
        if isinstance(agent, HybridAgent):
            action = agent.select_action(state, training=True, game_state=game_state)
        else:
            action = agent.select_action(state, training=True)
        
        # Take action
        next_state, reward, done, info = env.step(action)
        
        # Shape reward if using hybrid agent with reward shaping
        if isinstance(agent, HybridAgent) and game_state is not None:
            reward = agent.compute_shaped_reward(reward, game_state, action)
        
        # Store transition
        if isinstance(agent, DQNAgent):
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update DQN
            if agent.total_steps > agent.batch_size:
                metrics = agent.update()
                logger.log_step(metrics)
                
        elif isinstance(agent, (PPOAgent, HybridAgent)):
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update PPO (less frequent)
            if len(agent.buffer) >= agent.rollout_length:
                metrics = agent.update()
                logger.log_step(metrics)
        
        # Update state and metrics
        state = next_state
        episode_reward += reward
        episode_steps += 1
        
        if done:
            break
    
    # End of episode
    agent.reset_episode()
    
    # Log episode metrics
    logger.log_episode({
        'episode': episode_num,
        'reward': episode_reward,
        'steps': episode_steps,
        'score': info.get('score_reward', 0)
    })
    
    return episode_reward, episode_steps, info


def evaluate(env, agent, num_episodes=10):
    """Evaluate agent performance."""
    total_rewards = []
    total_scores = []
    wins = 0
    
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            game_state = env.get_game_state() if hasattr(env, 'get_game_state') else None
            
            if isinstance(agent, HybridAgent):
                action = agent.select_action(state, training=False, game_state=game_state)
            else:
                action = agent.select_action(state, training=False)
            
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state
        
        total_rewards.append(episode_reward)
        score = info.get('score_reward', 0)
        total_scores.append(score)
        if score > 0:
            wins += 1
    
    return {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_score': np.mean(total_scores),
        'win_rate': wins / num_episodes
    }


def main():
    """Main training loop."""
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load configuration
    config = load_config(args)
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.agent}_{args.scenario}_{timestamp}"
    output_dir = Path(f"outputs/{run_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Save configuration
    with open(output_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Create environment
    print(f"Creating environment: {args.scenario}")
    env = football_env.create_environment(
        env_name=args.scenario,
        stacked=False,
        representation='simple115v2',
        render=args.render,
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        write_video=False
    )
    
    # Wrap environment
    env = FootballEnvWrapper(env)
    env = RewardShaper(env, config.get('environment', {}))
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Create agent
    print(f"Creating {args.agent} agent...")
    agent = create_agent(args.agent, state_dim, action_dim, config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        agent.load(args.checkpoint)
    
    # Create logger and tracker
    logger = TrainingLogger(output_dir / "logs")
    tracker = PerformanceTracker()
    
    # Training loop
    print(f"\nStarting training for {args.episodes} episodes...")
    print(f"Output directory: {output_dir}")
    print("-" * 50)
    
    best_eval_reward = -float('inf')
    
    try:
        for episode in tqdm(range(1, args.episodes + 1), desc="Training"):
            # Train episode
            episode_reward, episode_steps, info = train_episode(
                env, agent, args.max_steps, logger, episode
            )
            
            # Track performance
            tracker.add_episode(episode_reward, episode_steps, info)
            
            # Evaluate periodically
            if episode % args.eval_frequency == 0:
                print(f"\n[Episode {episode}] Evaluating...")
                eval_metrics = evaluate(env, agent, num_episodes=10)
                logger.log_evaluation(episode, eval_metrics)
                
                print(f"Eval - Mean Reward: {eval_metrics['mean_reward']:.2f}, "
                      f"Win Rate: {eval_metrics['win_rate']:.2%}")
                
                # Save best model
                if eval_metrics['mean_reward'] > best_eval_reward:
                    best_eval_reward = eval_metrics['mean_reward']
                    best_path = checkpoint_dir / f"{args.agent}_best.pth"
                    agent.save(str(best_path))
                    print(f"New best model saved! Reward: {best_eval_reward:.2f}")
            
            # Save checkpoint periodically
            if episode % args.save_frequency == 0:
                checkpoint_path = checkpoint_dir / f"{args.agent}_ep{episode}.pth"
                agent.save(str(checkpoint_path))
                print(f"\nCheckpoint saved: {checkpoint_path}")
            
            # Adjust adaptive agent if applicable
            if isinstance(agent, AdaptiveHybridAgent):
                agent.adjust_lgbm_weight(episode_reward)
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    finally:
        # Save final model
        final_path = checkpoint_dir / f"{args.agent}_final.pth"
        agent.save(str(final_path))
        print(f"\nFinal model saved: {final_path}")
        
        # Save training statistics
        stats = tracker.get_statistics()
        logger.save_statistics(stats)
        
        # Close environment
        env.close()
        
        print("\nTraining completed!")
        print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()