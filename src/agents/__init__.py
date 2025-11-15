# src/agents/__init__.py
"""
Agent implementations for Football AI.
"""
from .base_agent import BaseAgent, EpsilonGreedyMixin
from .dqn_agent import DQNAgent
from .ppo_agent import PPOAgent
from .hybrid_agent import HybridAgent, AdaptiveHybridAgent

__all__ = [
    'BaseAgent',
    'EpsilonGreedyMixin',
    'DQNAgent',
    'PPOAgent',
    'HybridAgent',
    'AdaptiveHybridAgent',
]

# ============================================

# src/models/__init__.py
"""
Neural network models and LightGBM advisor.
"""
from .lgbm_model import FootballLGBMAdvisor, create_synthetic_match_data

__all__ = [
    'FootballLGBMAdvisor',
    'create_synthetic_match_data',
]

# ============================================

# src/environment/__init__.py
"""
Environment wrappers and utilities.
"""
from .wrappers import (
    FootballEnvWrapper,
    RewardShaper,
    FrameSkipWrapper,
    StackedFramesWrapper,
    MultiAgentWrapper,
    create_football_env
)

__all__ = [
    'FootballEnvWrapper',
    'RewardShaper',
    'FrameSkipWrapper',
    'StackedFramesWrapper',
    'MultiAgentWrapper',
    'create_football_env',
]

# ============================================

# src/training/__init__.py
"""
Training utilities and components.
"""

__all__ = []

# ============================================

# src/utils/__init__.py
"""
Utility functions for logging, metrics, and visualization.
"""
from .logger import TrainingLogger, ConsoleLogger
from .metrics import PerformanceTracker, compute_advanced_metrics, RollingMetrics

__all__ = [
    'TrainingLogger',
    'ConsoleLogger',
    'PerformanceTracker',
    'compute_advanced_metrics',
    'RollingMetrics',
]