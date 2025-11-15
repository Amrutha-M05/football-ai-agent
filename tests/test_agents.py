"""
Unit tests for agents
"""
import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile

from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.agents.hybrid_agent import HybridAgent


@pytest.fixture
def agent_config():
    """Common agent configuration."""
    return {
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'batch_size': 32,
        'use_cuda': False
    }


@pytest.fixture
def state_action_dims():
    """Common state and action dimensions."""
    return {'state_dim': 115, 'action_dim': 19}


class TestDQNAgent:
    """Test DQN agent."""
    
    def test_initialization(self, state_action_dims, agent_config):
        """Test agent initialization."""
        agent = DQNAgent(
            state_action_dims['state_dim'],
            state_action_dims['action_dim'],
            agent_config
        )
        assert agent.state_dim == 115
        assert agent.action_dim == 19
        assert agent.device.type in ['cpu', 'cuda']
    
    def test_select_action(self, state_action_dims, agent_config):
        """Test action selection."""
        agent = DQNAgent(
            state_action_dims['state_dim'],
            state_action_dims['action_dim'],
            agent_config
        )
        state = np.random.randn(115)
        
        # Training mode (with exploration)
        action = agent.select_action(state, training=True)
        assert 0 <= action < 19
        
        # Eval mode (no exploration)
        action = agent.select_action(state, training=False)
        assert 0 <= action < 19
    
    def test_store_transition(self, state_action_dims, agent_config):
        """Test storing transitions."""
        agent = DQNAgent(
            state_action_dims['state_dim'],
            state_action_dims['action_dim'],
            agent_config
        )
        
        state = np.random.randn(115)
        action = 5
        reward = 1.0
        next_state = np.random.randn(115)
        done = False
        
        agent.store_transition(state, action, reward, next_state, done)
        assert len(agent.replay_buffer) == 1
    
    def test_update(self, state_action_dims, agent_config):
        """Test agent update."""
        agent = DQNAgent(
            state_action_dims['state_dim'],
            state_action_dims['action_dim'],
            agent_config
        )
        
        # Fill buffer with transitions
        for _ in range(100):
            state = np.random.randn(115)
            action = np.random.randint(0, 19)
            reward = np.random.randn()
            next_state = np.random.randn(115)
            done = False
            agent.store_transition(state, action, reward, next_state, done)
        
        # Update should work
        metrics = agent.update()
        assert 'loss' in metrics
        assert isinstance(metrics['loss'], float)
    
    def test_save_load(self, state_action_dims, agent_config):
        """Test saving and loading."""
        agent = DQNAgent(
            state_action_dims['state_dim'],
            state_action_dims['action_dim'],
            agent_config
        )
        
        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_agent.pth'
            agent.save(str(save_path))
            assert save_path.exists()
            
            # Load
            new_agent = DQNAgent(
                state_action_dims['state_dim'],
                state_action_dims['action_dim'],
                agent_config
            )
            new_agent.load(str(save_path))
            assert new_agent.total_steps == agent.total_steps


class TestPPOAgent:
    """Test PPO agent."""
    
    def test_initialization(self, state_action_dims, agent_config):
        """Test agent initialization."""
        config = agent_config.copy()
        config['rollout_length'] = 128
        
        agent = PPOAgent(
            state_action_dims['state_dim'],
            state_action_dims['action_dim'],
            config
        )
        assert agent.state_dim == 115
        assert agent.action_dim == 19
    
    def test_select_action(self, state_action_dims, agent_config):
        """Test action selection."""
        config = agent_config.copy()
        config['rollout_length'] = 128
        
        agent = PPOAgent(
            state_action_dims['state_dim'],
            state_action_dims['action_dim'],
            config
        )
        state = np.random.randn(115)
        
        action = agent.select_action(state, training=True)
        assert 0 <= action < 19
    
    def test_buffer_operations(self, state_action_dims, agent_config):
        """Test rollout buffer."""
        config = agent_config.copy()
        config['rollout_length'] = 128
        
        agent = PPOAgent(
            state_action_dims['state_dim'],
            state_action_dims['action_dim'],
            config
        )
        
        # Store transitions
        for _ in range(10):
            state = np.random.randn(115)
            action = agent.select_action(state, training=True)
            reward = np.random.randn()
            next_state = np.random.randn(115)
            done = False
            agent.store_transition(state, action, reward, next_state, done)
        
        assert len(agent.buffer) == 10


class TestHybridAgent:
    """Test Hybrid agent."""
    
    def test_initialization(self, state_action_dims, agent_config):
        """Test hybrid agent initialization."""
        config = agent_config.copy()
        config['rollout_length'] = 128
        config['use_lgbm'] = False  # Don't load LightGBM for tests
        
        agent = HybridAgent(
            state_action_dims['state_dim'],
            state_action_dims['action_dim'],
            config
        )
        assert agent.state_dim == 115
        assert agent.action_dim == 19
    
    def test_select_action_without_lgbm(self, state_action_dims, agent_config):
        """Test action selection without LightGBM."""
        config = agent_config.copy()
        config['rollout_length'] = 128
        config['use_lgbm'] = False
        
        agent = HybridAgent(
            state_action_dims['state_dim'],
            state_action_dims['action_dim'],
            config
        )
        state = np.random.randn(115)
        
        action = agent.select_action(state, training=True)
        assert 0 <= action < 19


def test_epsilon_decay():
    """Test epsilon decay mechanism."""
    config = {'use_cuda': False}
    agent = DQNAgent(115, 19, config)
    
    initial_epsilon = agent.get_epsilon()
    
    # Decay multiple times
    for _ in range(100):
        agent.decay_epsilon()
    
    final_epsilon = agent.get_epsilon()
    
    # Should decay but not below minimum
    assert final_epsilon < initial_epsilon
    assert final_epsilon >= agent.epsilon_end


def test_device_handling():
    """Test device handling (CPU/CUDA)."""
    config = {'use_cuda': torch.cuda.is_available()}
    agent = DQNAgent(115, 19, config)
    
    if torch.cuda.is_available():
        assert agent.device.type == 'cuda'
    else:
        assert agent.device.type == 'cpu'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])