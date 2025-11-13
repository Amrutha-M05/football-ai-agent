"""
Base Agent Interface for Football AI
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np
import torch


class BaseAgent(ABC):
    """Abstract base class for all agents."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        """
        Initialize base agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Configuration dictionary
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and config.get("use_cuda", True) else "cpu"
        )
        
        # Training statistics
        self.total_steps = 0
        self.episodes = 0
        self.training_losses = []
        
    @abstractmethod
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action given current state.
        
        Args:
            state: Current state observation
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action
        """
        pass
    
    @abstractmethod
    def update(self, *args, **kwargs) -> Dict[str, float]:
        """
        Update agent parameters.
        
        Returns:
            Dictionary of training metrics
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save agent checkpoint."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load agent checkpoint."""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            "total_steps": self.total_steps,
            "episodes": self.episodes,
            "avg_loss": np.mean(self.training_losses[-100:]) if self.training_losses else 0.0
        }
    
    def reset_episode(self) -> None:
        """Reset episode-specific variables."""
        self.episodes += 1


class EpsilonGreedyMixin:
    """Mixin for epsilon-greedy exploration."""
    
    def __init__(self, epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995):
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
    
    def get_epsilon(self) -> float:
        """Get current epsilon value."""
        return max(self.epsilon_end, self.epsilon)
    
    def decay_epsilon(self) -> None:
        """Decay epsilon value."""
        self.epsilon *= self.epsilon_decay
    
    def should_explore(self) -> bool:
        """Decide whether to explore based on epsilon."""
        return np.random.random() < self.get_epsilon()