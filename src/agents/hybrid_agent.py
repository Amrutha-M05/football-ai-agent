"""
Hybrid Agent: Combines RL (PPO) with LightGBM Advisor
"""
import numpy as np
import torch
from typing import Dict, Any, Optional

from .ppo_agent import PPOAgent
from ..models.lgbm_model import FootballLGBMAdvisor


class HybridAgent(PPOAgent):
    """
    Hybrid agent combining PPO with LightGBM advisor.
    Uses LightGBM predictions to guide RL policy.
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        super().__init__(state_dim, action_dim, config)
        
        # LightGBM advisor
        self.use_lgbm = config.get("use_lgbm", True)
        self.lgbm_weight = config.get("lgbm_weight", 0.3)  # Weight for advisor influence
        self.lgbm_advisor: Optional[FootballLGBMAdvisor] = None
        
        if self.use_lgbm:
            lgbm_config = config.get("lgbm_config", {})
            lgbm_config['num_actions'] = action_dim
            self.lgbm_advisor = FootballLGBMAdvisor(lgbm_config)
            
            # Load pre-trained advisor if available
            lgbm_path = config.get("lgbm_model_path")
            if lgbm_path:
                try:
                    self.lgbm_advisor.load(lgbm_path)
                    print(f"Loaded LightGBM advisor from {lgbm_path}")
                except FileNotFoundError:
                    print(f"Warning: LightGBM model not found at {lgbm_path}")
                    self.use_lgbm = False
        
        # Integration method
        self.integration_method = config.get("integration_method", "weighted")  # weighted, filtering, or reward_shaping
        
    def select_action(self, state: np.ndarray, training: bool = True, 
                     game_state: Optional[Dict] = None) -> int:
        """
        Select action combining PPO and LightGBM predictions.
        
        Args:
            state: Current state observation
            training: Whether in training mode
            game_state: Full game state for LightGBM (optional)
            
        Returns:
            Selected action
        """
        # Get PPO action distribution
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_logits, value = self.policy(state_tensor)
            ppo_probs = torch.softmax(action_logits, dim=1).cpu().numpy()[0]
        
        # If not using LightGBM or no game state, use pure PPO
        if not self.use_lgbm or game_state is None:
            return super().select_action(state, training)
        
        # Get LightGBM action distribution
        lgbm_probs = self.lgbm_advisor.get_action_distribution(game_state)
        
        # Combine predictions based on integration method
        if self.integration_method == "weighted":
            # Weighted combination of distributions
            combined_probs = (1 - self.lgbm_weight) * ppo_probs + self.lgbm_weight * lgbm_probs
            action = np.random.choice(self.action_dim, p=combined_probs)
            
        elif self.integration_method == "filtering":
            # Use LightGBM to filter top-k actions, then use PPO to select
            k = max(3, self.action_dim // 4)  # Top 25% actions
            top_k_actions = np.argsort(lgbm_probs)[-k:]
            
            # Renormalize PPO probs over top-k actions
            filtered_probs = ppo_probs[top_k_actions]
            filtered_probs = filtered_probs / filtered_probs.sum()
            
            action = np.random.choice(top_k_actions, p=filtered_probs)
            
        else:  # reward_shaping - just use PPO action, reward shaping happens during training
            action = np.random.choice(self.action_dim, p=ppo_probs)
        
        # Store value for buffer (if training)
        if training:
            self._last_value = value.item()
            
            # Compute combined log prob for training
            if self.integration_method == "weighted":
                combined_probs_tensor = torch.FloatTensor(combined_probs).to(self.device)
                self._last_log_prob = torch.log(combined_probs_tensor[action] + 1e-10).item()
            else:
                from torch.distributions import Categorical
                dist = Categorical(logits=action_logits)
                self._last_log_prob = dist.log_prob(torch.tensor([action]).to(self.device)).item()
        
        return action
    
    def compute_shaped_reward(self, base_reward: float, game_state: Dict, 
                            action: int) -> float:
        """
        Shape reward using LightGBM advisor.
        
        Args:
            base_reward: Original environment reward
            game_state: Current game state
            action: Action taken
            
        Returns:
            Shaped reward
        """
        if not self.use_lgbm or self.integration_method != "reward_shaping":
            return base_reward
        
        # Get LightGBM's action recommendation
        lgbm_action, lgbm_probs = self.lgbm_advisor.predict_action(game_state, return_probs=True)
        
        # Add bonus if action aligns with LightGBM recommendation
        if action == lgbm_action:
            bonus = self.lgbm_weight * 0.1  # Small bonus for alignment
        else:
            # Penalty proportional to how bad the action is according to LightGBM
            action_quality = lgbm_probs[action]
            bonus = -self.lgbm_weight * 0.05 * (1 - action_quality)
        
        return base_reward + bonus
    
    def train_lgbm_advisor(self, match_data_path: str) -> Dict[str, float]:
        """
        Train the LightGBM advisor on real match data.
        
        Args:
            match_data_path: Path to match data CSV
            
        Returns:
            Training metrics
        """
        if self.lgbm_advisor is None:
            raise ValueError("LightGBM advisor not initialized")
        
        return self.lgbm_advisor.train(match_data_path)
    
    def save(self, path: str) -> None:
        """Save hybrid agent checkpoint."""
        # Save PPO component
        super().save(path)
        
        # Save LightGBM advisor
        if self.lgbm_advisor is not None and self.use_lgbm:
            lgbm_path = path.replace('.pth', '_lgbm.pkl')
            self.lgbm_advisor.save(lgbm_path)
    
    def load(self, path: str) -> None:
        """Load hybrid agent checkpoint."""
        # Load PPO component
        super().load(path)
        
        # Load LightGBM advisor
        if self.use_lgbm:
            lgbm_path = path.replace('.pth', '_lgbm.pkl')
            try:
                if self.lgbm_advisor is None:
                    self.lgbm_advisor = FootballLGBMAdvisor(self.config.get("lgbm_config", {}))
                self.lgbm_advisor.load(lgbm_path)
            except FileNotFoundError:
                print(f"Warning: LightGBM model not found at {lgbm_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics including LightGBM info."""
        stats = super().get_stats()
        stats['using_lgbm'] = self.use_lgbm
        stats['lgbm_weight'] = self.lgbm_weight
        stats['integration_method'] = self.integration_method
        return stats


class AdaptiveHybridAgent(HybridAgent):
    """
    Adaptive version that adjusts LightGBM weight based on performance.
    """
    
    def __init__(self, state_dim: int, action_dim: int, config: Dict[str, Any]):
        super().__init__(state_dim, action_dim, config)
        
        self.initial_lgbm_weight = self.lgbm_weight
        self.min_lgbm_weight = config.get("min_lgbm_weight", 0.1)
        self.max_lgbm_weight = config.get("max_lgbm_weight", 0.5)
        
        # Track performance
        self.recent_rewards = []
        self.window_size = 100
        
    def adjust_lgbm_weight(self, episode_reward: float) -> None:
        """
        Adjust LightGBM weight based on recent performance.
        
        Args:
            episode_reward: Reward from latest episode
        """
        self.recent_rewards.append(episode_reward)
        
        if len(self.recent_rewards) > self.window_size:
            self.recent_rewards.pop(0)
        
        if len(self.recent_rewards) >= self.window_size:
            # Calculate performance trend
            mid_point = self.window_size // 2
            early_avg = np.mean(self.recent_rewards[:mid_point])
            late_avg = np.mean(self.recent_rewards[mid_point:])
            
            # If performance improving, reduce LightGBM weight (trust RL more)
            if late_avg > early_avg * 1.1:
                self.lgbm_weight = max(self.min_lgbm_weight, self.lgbm_weight * 0.95)
            # If performance declining, increase LightGBM weight
            elif late_avg < early_avg * 0.9:
                self.lgbm_weight = min(self.max_lgbm_weight, self.lgbm_weight * 1.05)
            
            print(f"Adjusted LightGBM weight to {self.lgbm_weight:.3f}")