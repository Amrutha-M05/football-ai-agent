"""
Proximal Policy Optimization (PPO) Agent Implementation
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, Any, List, Tuple

from .base_agent import BaseAgent


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO."""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [512, 256]):
        super(ActorCritic, self).__init__()
        
        # Shared feature extractor
        shared_layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims[:-1]:
            shared_layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            input_dim = hidden_dim
        
        self.shared = nn.Sequential(*shared_layers)
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            action_logits: Logits for action distribution
            value: State value estimate
        """
        features = self.shared(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: State value estimate
        """
        action_logits, value = self.forward(state)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value


class RolloutBuffer:
    """Buffer for storing rollout experiences for PPO."""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def push(self, state: np.ndarray, action: int, reward: float,
             value: float, log_prob: float, done: bool) -> None:
        """Add experience to buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_returns_and_advantages(self, last_value: float, gamma: float,
                                       gae_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute returns and advantages using GAE.
        
        Args:
            last_value: Value estimate of final state
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            
        Returns:
            returns: Discounted returns
            advantages: Generalized advantage estimates
        """
        returns = []
        advantages = []
        gae = 0
        
        # Compute advantages in reverse
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_value * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[t])
        
        return np.array(returns), np.array(advantages)
    
    def get(self) -> Dict[str, np.ndarray]:
        """Get all stored data."""
        return {
            "states": np.array(self.states),
            "actions": np.array(self.actions),
            "rewards": np.array(self.rewards),
            "values": np.array(self.values),
            "log_probs": np.array(self.log_probs),
            "dones": np.array(self.dones)
        }
    
    def clear(self) -> None:
        """Clear buffer."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
    
    def __len__(self) -> int:
        return len(self.states)


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization agent."""
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        super().__init__(state_dim, action_dim, config)
        
        # Hyperparameters
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.clip_epsilon = config.get("clip_epsilon", 0.2)
        self.learning_rate = config.get("learning_rate", 3e-4)
        self.value_coef = config.get("value_coef", 0.5)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)
        self.update_epochs = config.get("update_epochs", 4)
        self.batch_size = config.get("batch_size", 64)
        self.rollout_length = config.get("rollout_length", 2048)
        
        # Network
        hidden_dims = config.get("hidden_dims", [512, 256])
        self.policy = ActorCritic(state_dim, action_dim, hidden_dims).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action from policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            if training:
                action, log_prob, value = self.policy.get_action(state_tensor)
                # Store for later use in buffer
                self._last_log_prob = log_prob.item()
                self._last_value = value.item()
                return action
            else:
                action_logits, _ = self.policy(state_tensor)
                return action_logits.argmax(dim=1).item()
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool) -> None:
        """Store transition in rollout buffer."""
        self.buffer.push(
            state, action, reward,
            self._last_value, self._last_log_prob, done
        )
        self.total_steps += 1
    
    def update(self) -> Dict[str, float]:
        """
        Update policy using PPO.
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.buffer) < self.rollout_length:
            return {"loss": 0.0}
        
        # Get rollout data
        data = self.buffer.get()
        
        # Get last value estimate for GAE computation
        with torch.no_grad():
            last_state = torch.FloatTensor(data["states"][-1]).unsqueeze(0).to(self.device)
            _, last_value = self.policy(last_state)
            last_value = last_value.item()
        
        # Compute returns and advantages
        returns, advantages = self.buffer.compute_returns_and_advantages(
            last_value, self.gamma, self.gae_lambda
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        states = torch.FloatTensor(data["states"]).to(self.device)
        actions = torch.LongTensor(data["actions"]).to(self.device)
        old_log_probs = torch.FloatTensor(data["log_probs"]).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        update_count = 0
        
        for _ in range(self.update_epochs):
            # Sample mini-batches
            indices = np.random.permutation(len(states))
            
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Forward pass
                action_logits, values = self.policy(batch_states)
                dist = Categorical(logits=action_logits)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # Policy loss (clipped)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track metrics
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                update_count += 1
        
        # Clear buffer
        self.buffer.clear()
        
        # Average metrics
        avg_policy_loss = total_policy_loss / update_count
        avg_value_loss = total_value_loss / update_count
        avg_entropy = total_entropy / update_count
        
        self.training_losses.append(avg_policy_loss + avg_value_loss)
        
        return {
            "policy_loss": avg_policy_loss,
            "value_loss": avg_value_loss,
            "entropy": avg_entropy,
            "total_loss": avg_policy_loss + avg_value_loss
        }
    
    def save(self, path: str) -> None:
        """Save agent checkpoint."""
        checkpoint = {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "episodes": self.episodes
        }
        torch.save(checkpoint, path)
        print(f"Agent saved to {path}")
    
    def load(self, path: str) -> None:
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint["total_steps"]
        self.episodes = checkpoint["episodes"]
        print(f"Agent loaded from {path}")