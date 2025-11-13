"""
Benchmark script for CPU vs GPU performance comparison
"""
import argparse
import time
import numpy as np
import torch
import psutil
from pathlib import Path
import json
from typing import Dict, List
import gfootball.env as football_env

from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.hybrid_agent import HybridAgent
from src.environment.wrappers import FootballEnvWrapper


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Benchmark Football AI Agent')
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device to benchmark')
    parser.add_argument('--num-episodes', type=int, default=50,
                       help='Number of episodes for benchmarking')
    parser.add_argument('--num-warmup', type=int, default=5,
                       help='Number of warmup episodes')
    parser.add_argument('--scenario', type=str, default='academy_empty_goal',
                       help='Football scenario')
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output file for results')
    parser.add_argument('--profile-memory', action='store_true',
                       help='Profile memory usage')
    
    return parser.parse_args()


class PerformanceBenchmark:
    """Benchmark agent performance metrics."""
    
    def __init__(self, device: str):
        self.device = device
        self.inference_times: List[float] = []
        self.episode_times: List[float] = []
        self.memory_usage: List[float] = []
        self.cpu_usage: List[float] = []
        
    def start_episode(self):
        """Mark start of episode timing."""
        self.episode_start_time = time.time()
        
    def end_episode(self):
        """Mark end of episode and record timing."""
        episode_time = time.time() - self.episode_start_time
        self.episode_times.append(episode_time)
        
    def record_inference(self, inference_time: float):
        """Record single inference time."""
        self.inference_times.append(inference_time)
    
    def record_memory(self):
        """Record current memory usage."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.memory_usage.append(memory_mb)
        
    def record_cpu(self):
        """Record CPU usage."""
        cpu_percent = psutil.cpu_percent(interval=None)
        self.cpu_usage.append(cpu_percent)
    
    def get_statistics(self) -> Dict[str, float]:
        """Compute and return benchmark statistics."""
        stats = {}
        
        # Inference timing
        if self.inference_times:
            stats['mean_inference_time_ms'] = np.mean(self.inference_times) * 1000
            stats['std_inference_time_ms'] = np.std(self.inference_times) * 1000
            stats['min_inference_time_ms'] = np.min(self.inference_times) * 1000
            stats['max_inference_time_ms'] = np.max(self.inference_times) * 1000
            stats['fps'] = 1.0 / np.mean(self.inference_times) if np.mean(self.inference_times) > 0 else 0
        
        # Episode timing
        if self.episode_times:
            stats['mean_episode_time_s'] = np.mean(self.episode_times)
            stats['std_episode_time_s'] = np.std(self.episode_times)
            stats['total_time_s'] = np.sum(self.episode_times)
        
        # Memory usage
        if self.memory_usage:
            stats['mean_memory_mb'] = np.mean(self.memory_usage)
            stats['peak_memory_mb'] = np.max(self.memory_usage)
            stats['min_memory_mb'] = np.min(self.memory_usage)
        
        # CPU usage
        if self.cpu_usage:
            stats['mean_cpu_percent'] = np.mean(self.cpu_usage)
            stats['max_cpu_percent'] = np.max(self.cpu_usage)
        
        # GPU metrics (if applicable)
        if self.device == 'cuda' and torch.cuda.is_available():
            stats['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            stats['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
            stats['gpu_max_memory_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        stats['device'] = self.device
        
        return stats


def benchmark_agent(agent, env, num_episodes: int, num_warmup: int,
                   device: str, profile_memory: bool = False) -> Dict[str, float]:
    """
    Benchmark agent performance.
    
    Args:
        agent: Trained agent
        env: Environment
        num_episodes: Number of episodes to benchmark
        num_warmup: Number of warmup episodes
        device: Device being benchmarked
        profile_memory: Whether to profile memory usage
        
    Returns:
        Dictionary of benchmark statistics
    """
    benchmark = PerformanceBenchmark(device)
    
    # Warmup
    print(f"Warming up with {num_warmup} episodes...")
    for _ in range(num_warmup):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, training=False)
            state, _, done, _ = env.step(action)
    
    # Clear GPU cache if using CUDA
    if device == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Benchmark
    print(f"\nBenchmarking with {num_episodes} episodes on {device.upper()}...")
    
    for episode in range(num_episodes):
        benchmark.start_episode()
        state = env.reset()
        done = False
        step_count = 0
        
        while not done:
            # Time inference
            inference_start = time.time()
            
            with torch.no_grad():
                action = agent.select_action(state, training=False)
            
            # Synchronize for accurate GPU timing
            if device == 'cuda':
                torch.cuda.synchronize()
            
            inference_time = time.time() - inference_start
            benchmark.record_inference(inference_time)
            
            # Take step
            state, _, done, _ = env.step(action)
            step_count += 1
            
            # Profile memory periodically
            if profile_memory and step_count % 50 == 0:
                benchmark.record_memory()
                benchmark.record_cpu()
        
        benchmark.end_episode()
        
        # Progress update
        if (episode + 1) % 10 == 0:
            print(f"  Completed {episode + 1}/{num_episodes} episodes")
    
    # Get statistics
    stats = benchmark.get_statistics()
    
    return stats


def load_agent_for_benchmark(model_path: str, device: str):
    """Load agent for benchmarking."""
    device_obj = torch.device(device)
    checkpoint = torch.load(model_path, map_location=device_obj)
    
    # Determine agent type and create agent
    # This is simplified - in practice, would load from config
    if 'q_network' in checkpoint:
        state_dim = list(checkpoint['q_network'].values())[0].shape[1]
        action_dim = list(checkpoint['q_network'].values())[-1].shape[0]
        agent = DQNAgent(state_dim, action_dim, {'use_cuda': device == 'cuda'})
    else:
        first_key = [k for k in checkpoint['policy'].keys() if 'shared.0.weight' in k][0]
        state_dim = checkpoint['policy'][first_key].shape[1]
        actor_keys = [k for k in checkpoint['policy'].keys() if 'actor.2.weight' in k]
        action_dim = checkpoint['policy'][actor_keys[0]].shape[0] if actor_keys else 19
        agent = PPOAgent(state_dim, action_dim, {'use_cuda': device == 'cuda'})
    
    agent.load(model_path)
    agent.policy.eval() if hasattr(agent, 'policy') else agent.q_network.eval()
    
    return agent


def compare_devices(model_path: str, scenarios: List[str], 
                   num_episodes: int, num_warmup: int) -> Dict:
    """
    Compare performance across CPU and GPU.
    
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    for device in ['cpu', 'cuda']:
        if device == 'cuda' and not torch.cuda.is_available():
            print(f"Skipping {device.upper()} - not available")
            continue
        
        print(f"\n{'='*60}")
        print(f"Benchmarking on {device.upper()}")
        print(f"{'='*60}")
        
        device_results = {}
        
        for scenario in scenarios:
            print(f"\nScenario: {scenario}")
            
            # Create environment
            env = football_env.create_environment(
                env_name=scenario,
                stacked=False,
                representation='simple115v2',
                render=False
            )
            env = FootballEnvWrapper(env)
            
            # Load agent
            agent = load_agent_for_benchmark(model_path, device)
            
            # Benchmark
            stats = benchmark_agent(
                agent, env, num_episodes, num_warmup, 
                device, profile_memory=True
            )
            
            device_results[scenario] = stats
            
            # Print results
            print(f"\nResults:")
            print(f"  Mean FPS: {stats['fps']:.1f}")
            print(f"  Mean Inference Time: {stats['mean_inference_time_ms']:.2f} ms")
            print(f"  Mean Memory Usage: {stats.get('mean_memory_mb', 0):.1f} MB")
            
            env.close()
        
        results[device] = device_results
    
    return results


def print_comparison_table(results: Dict):
    """Print comparison table of CPU vs GPU results."""
    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON: CPU vs GPU")
    print("="*80)
    
    if 'cpu' not in results or 'cuda' not in results:
        print("Incomplete comparison data")
        return
    
    scenarios = list(results['cpu'].keys())
    
    print(f"\n{'Metric':<30} {'CPU':>15} {'GPU':>15} {'Speedup':>15}")
    print("-"*80)
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario}")
        
        cpu_stats = results['cpu'][scenario]
        gpu_stats = results['cuda'][scenario]
        
        # FPS comparison
        cpu_fps = cpu_stats['fps']
        gpu_fps = gpu_stats['fps']
        speedup = gpu_fps / cpu_fps if cpu_fps > 0 else 0
        print(f"  {'FPS':<28} {cpu_fps:>15.1f} {gpu_fps:>15.1f} {speedup:>14.2f}x")
        
        # Inference time comparison
        cpu_inf = cpu_stats['mean_inference_time_ms']
        gpu_inf = gpu_stats['mean_inference_time_ms']
        speedup = cpu_inf / gpu_inf if gpu_inf > 0 else 0
        print(f"  {'Inference Time (ms)':<28} {cpu_inf:>15.2f} {gpu_inf:>15.2f} {speedup:>14.2f}x")
        
        # Memory comparison
        cpu_mem = cpu_stats.get('mean_memory_mb', 0)
        gpu_mem = gpu_stats.get('gpu_memory_allocated_mb', 0)
        print(f"  {'Memory Usage (MB)':<28} {cpu_mem:>15.1f} {gpu_mem:>15.1f}")
    
    print("\n" + "="*80)


def main():
    """Main benchmark function."""
    args = parse_args()
    
    # Single device benchmark
    print(f"Benchmarking on {args.device.upper()}")
    print(f"Model: {args.model_path}")
    print(f"Scenario: {args.scenario}")
    print(f"Episodes: {args.num_episodes}")
    
    # Create environment
    env = football_env.create_environment(
        env_name=args.scenario,
        stacked=False,
        representation='simple115v2',
        render=False
    )
    env = FootballEnvWrapper(env)
    
    # Load agent
    agent = load_agent_for_benchmark(args.model_path, args.device)
    
    # Benchmark
    stats = benchmark_agent(
        agent, env, args.num_episodes, args.num_warmup,
        args.device, profile_memory=args.profile_memory
    )
    
    # Print results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"\nDevice: {stats['device'].upper()}")
    print(f"\nTiming Metrics:")
    print(f"  Mean FPS: {stats['fps']:.1f}")
    print(f"  Mean Inference Time: {stats['mean_inference_time_ms']:.2f} ± {stats['std_inference_time_ms']:.2f} ms")
    print(f"  Min Inference Time: {stats['min_inference_time_ms']:.2f} ms")
    print(f"  Max Inference Time: {stats['max_inference_time_ms']:.2f} ms")
    print(f"  Mean Episode Time: {stats['mean_episode_time_s']:.2f} ± {stats['std_episode_time_s']:.2f} s")
    print(f"  Total Time: {stats['total_time_s']:.2f} s")
    
    print(f"\nResource Usage:")
    print(f"  Mean Memory: {stats.get('mean_memory_mb', 0):.1f} MB")
    print(f"  Peak Memory: {stats.get('peak_memory_mb', 0):.1f} MB")
    print(f"  Mean CPU Usage: {stats.get('mean_cpu_percent', 0):.1f}%")
    
    if args.device == 'cuda':
        print(f"\nGPU Metrics:")
        print(f"  GPU Memory Allocated: {stats.get('gpu_memory_allocated_mb', 0):.1f} MB")
        print(f"  GPU Memory Reserved: {stats.get('gpu_memory_reserved_mb', 0):.1f} MB")
        print(f"  GPU Max Memory: {stats.get('gpu_max_memory_allocated_mb', 0):.1f} MB")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Optional: Full comparison
    if input("\nRun full CPU vs GPU comparison? (y/n): ").lower() == 'y':
        test_scenarios = [
            'academy_empty_goal',
            'academy_run_to_score',
            'academy_3_vs_1_with_keeper'
        ]
        
        comparison_results = compare_devices(
            args.model_path, test_scenarios, 
            args.num_episodes, args.num_warmup
        )
        
        print_comparison_table(comparison_results)
        
        # Save comparison
        with open('cpu_gpu_comparison.json', 'w') as f:
            json.dump(comparison_results, f, indent=2)
    
    env.close()
    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()