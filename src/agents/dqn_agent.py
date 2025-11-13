"""
Deep Q-Network (DQN) Agent Implementation
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Any, Tuple
import random
from collections import deque

from .base_agent import BaseAgent, EpsilonGreedyMixin


class DQNNetwork(nn.Module):
    """Deep Q-Network architecture."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [512, 256]):
        super(DQNNetwork, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> None:
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample random batch from buffer."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent(BaseAgent, EpsilonGreedyMixin):
    """Deep Q-Network agent with experience replay and target network."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        BaseAgent.__init__(self, state_dim, action_dim, config)
        EpsilonGreedyMixin.__init__(
            self,
            epsilon_start=config.get("epsilon_start", 1.0),
            epsilon_end=config.get("epsilon_end", 0.01),
            epsilon_decay=config.get("epsilon_decay", 0.995)
        )
        
        # Hyperparameters
        self.gamma = config.get("gamma", 0.99)
        self.learning_rate = config.get("learning_rate", 1e-4)
        self.batch_size = config.get("batch_size", 64)
        self.target_update_freq = config.get("target_update_freq", 1000)
        self.buffer_size = config.get("buffer_size", 100000)
        
        # Networks
        hidden_dims = config.get("hidden_dims", [512, 256])
        self.q_network = DQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network = DQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size)
        
        # Loss function
        self.criterion = nn.SmoothL1Loss()  # Huber loss
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if training and self.should_explore():
            return np.random.randint(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax(dim=1).item()
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool) -> None:
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1
    
    def update(self) -> Dict[str, float]:
        """
        Update Q-network using experience replay.
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": 0.0}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network
        if self.total_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.decay_epsilon()
        
        # Store loss
        loss_value = loss.item()
        self.training_losses.append(loss_value)
        
        return {
            "loss": loss_value,
            "epsilon": self.get_epsilon(),
            "q_value": current_q_values.mean().item()
        }
    
    def save(self, path: str) -> None:
        """Save agent checkpoint."""
        checkpoint = {
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
            "episodes": self.episodes
        }
        torch.save(checkpoint, path)
        print(f"Agent saved to {path}")
    
    def load(self, path: str) -> None:
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint["epsilon"]
        self.total_steps = checkpoint["total_steps"]
        self.episodes = checkpoint["episodes"]
        print(f"Agent loaded from {path}")