"""
Export trained models to ONNX format for deployment
"""
import argparse
import torch
import numpy as np
from pathlib import Path

from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.hybrid_agent import HybridAgent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Export model to ONNX format')
    
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--output-path', type=str, default=None,
                       help='Output path for ONNX model')
    parser.add_argument('--state-dim', type=int, default=115,
                       help='State dimension')
    parser.add_argument('--action-dim', type=int, default=19,
                       help='Action dimension')
    parser.add_argument('--opset-version', type=int, default=11,
                       help='ONNX opset version')
    
    return parser.parse_args()


def export_dqn_to_onnx(agent, output_path, state_dim, opset_version):
    """
    Export DQN model to ONNX.
    
    Args:
        agent: Trained DQN agent
        output_path: Output file path
        state_dim: State dimension
        opset_version: ONNX opset version
    """
    agent.q_network.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, state_dim).to(agent.device)
    
    # Export
    torch.onnx.export(
        agent.q_network,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['state'],
        output_names=['q_values'],
        dynamic_axes={
            'state': {0: 'batch_size'},
            'q_values': {0: 'batch_size'}
        }
    )
    
    print(f"DQN model exported to {output_path}")


def export_ppo_to_onnx(agent, output_path, state_dim, opset_version):
    """
    Export PPO model to ONNX.
    
    Args:
        agent: Trained PPO agent
        output_path: Output file path
        state_dim: State dimension
        opset_version: ONNX opset version
    """
    agent.policy.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, state_dim).to(agent.device)
    
    # Export actor-critic
    torch.onnx.export(
        agent.policy,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['state'],
        output_names=['action_logits', 'value'],
        dynamic_axes={
            'state': {0: 'batch_size'},
            'action_logits': {0: 'batch_size'},
            'value': {0: 'batch_size'}
        }
    )
    
    print(f"PPO model exported to {output_path}")


def verify_onnx_model(onnx_path, state_dim, device):
    """
    Verify exported ONNX model.
    
    Args:
        onnx_path: Path to ONNX model
        state_dim: State dimension
        device: Device for PyTorch model
    """
    try:
        import onnx
        import onnxruntime as ort
        
        # Load and check ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid")
        
        # Create ONNX runtime session
        ort_session = ort.InferenceSession(onnx_path)
        
        # Test inference
        dummy_input = np.random.randn(1, state_dim).astype(np.float32)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
        ort_outputs = ort_session.run(None, ort_inputs)
        
        print("✓ ONNX inference successful")
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shapes: {[out.shape for out in ort_outputs]}")
        
        return True
        
    except ImportError:
        print("⚠ Warning: onnx and onnxruntime not installed. Skipping verification.")
        print("  Install with: pip install onnx onnxruntime")
        return False
    except Exception as e:
        print(f"✗ ONNX verification failed: {e}")
        return False


def benchmark_onnx_inference(onnx_path, state_dim, num_iterations=1000):
    """
    Benchmark ONNX model inference speed.
    
    Args:
        onnx_path: Path to ONNX model
        state_dim: State dimension
        num_iterations: Number of inference iterations
    """
    try:
        import onnxruntime as ort
        import time
        
        # Create session
        ort_session = ort.InferenceSession(onnx_path)
        
        # Warmup
        dummy_input = np.random.randn(1, state_dim).astype(np.float32)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
        for _ in range(10):
            ort_session.run(None, ort_inputs)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            ort_session.run(None, ort_inputs)
        elapsed_time = time.time() - start_time
        
        avg_time_ms = (elapsed_time / num_iterations) * 1000
        fps = num_iterations / elapsed_time
        
        print(f"\nONNX Inference Benchmark ({num_iterations} iterations):")
        print(f"  Average inference time: {avg_time_ms:.2f} ms")
        print(f"  Throughput: {fps:.1f} FPS")
        
    except ImportError:
        print("⚠ Warning: onnxruntime not installed. Skipping benchmark.")
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")


def main():
    """Main export function."""
    args = parse_args()
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location='cpu')
    
    # Determine agent type
    if 'q_network' in checkpoint:
        agent_type = 'dqn'
        agent = DQNAgent(args.state_dim, args.action_dim, {'use_cuda': False})
    elif 'policy' in checkpoint:
        agent_type = 'ppo'
        agent = PPOAgent(args.state_dim, args.action_dim, {'use_cuda': False})
    else:
        print("Error: Unknown model type")
        return
    
    # Load weights
    agent.load(args.model_path)
    print(f"Loaded {agent_type.upper()} model from {args.model_path}")
    
    # Determine output path
    if args.output_path:
        output_path = args.output_path
    else:
        model_path = Path(args.model_path)
        output_path = model_path.parent / f"{model_path.stem}.onnx"
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Export to ONNX
    print(f"\nExporting to ONNX format...")
    print(f"Output path: {output_path}")
    
    if agent_type == 'dqn':
        export_dqn_to_onnx(agent, str(output_path), args.state_dim, args.opset_version)
    else:
        export_ppo_to_onnx(agent, str(output_path), args.state_dim, args.opset_version)
    
    # Verify
    print("\nVerifying ONNX model...")
    if verify_onnx_model(str(output_path), args.state_dim, agent.device):
        # Benchmark
        benchmark_onnx_inference(str(output_path), args.state_dim)
    
    # Print file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nModel file size: {file_size_mb:.2f} MB")
    
    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    print("\nYou can now use the ONNX model for deployment:")
    print(f"  import onnxruntime as ort")
    print(f"  session = ort.InferenceSession('{output_path}')")
    print(f"  outputs = session.run(None, {{'state': state_input}})")


if __name__ == "__main__":
    main()