"""
LightGBM Advisor for Football Strategy
Trained on real match data to provide action recommendations
"""
import numpy as np
import lightgbm as lgb
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import pickle
from pathlib import Path


class FootballLGBMAdvisor:
    """LightGBM model for football strategy recommendation."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LightGBM advisor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.model: Optional[lgb.Booster] = None
        self.feature_names: List[str] = []
        self.action_mapping: Dict[int, str] = {}
        
        # Training parameters
        self.params = {
            'objective': 'multiclass',
            'num_class': config.get('num_actions', 19),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': config.get('num_leaves', 31),
            'learning_rate': config.get('learning_rate', 0.05),
            'feature_fraction': config.get('feature_fraction', 0.9),
            'bagging_fraction': config.get('bagging_fraction', 0.8),
            'bagging_freq': 5,
            'verbose': -1
        }
        
    def extract_features(self, game_state: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from game state.
        
        Args:
            game_state: Current game state dictionary
            
        Returns:
            Feature vector
        """
        features = []
        
        # Ball possession features
        features.append(game_state.get('ball_owned_player', -1))
        features.append(game_state.get('ball_owned_team', -1))
        
        # Ball position
        ball_pos = game_state.get('ball', [0, 0, 0])
        features.extend(ball_pos)
        
        # Player positions (controlled player)
        player_pos = game_state.get('left_team', [[0, 0]])[0]
        features.extend(player_pos)
        
        # Distance to ball
        dist_to_ball = np.linalg.norm(np.array(player_pos) - np.array(ball_pos[:2]))
        features.append(dist_to_ball)
        
        # Opponent positions (nearest)
        right_team = game_state.get('right_team', [[0, 0]])
        if len(right_team) > 0:
            nearest_opponent = min(right_team, key=lambda x: np.linalg.norm(np.array(player_pos) - np.array(x)))
            features.extend(nearest_opponent)
            dist_to_opponent = np.linalg.norm(np.array(player_pos) - np.array(nearest_opponent))
            features.append(dist_to_opponent)
        else:
            features.extend([0, 0, 1.0])
        
        # Game state features
        features.append(game_state.get('steps_left', 3001))
        features.append(game_state.get('score', [0, 0])[0])  # Our score
        features.append(game_state.get('score', [0, 0])[1])  # Opponent score
        
        # Game mode (attack/defense indicator)
        our_score, opp_score = game_state.get('score', [0, 0])
        features.append(1 if our_score > opp_score else 0)  # Winning
        features.append(1 if ball_pos[0] > 0 else 0)  # Attacking half
        
        return np.array(features, dtype=np.float32)
    
    def train(self, match_data_path: str, val_split: float = 0.2) -> Dict[str, float]:
        """
        Train LightGBM model on real match data.
        
        Args:
            match_data_path: Path to CSV file with match data
            val_split: Validation split ratio
            
        Returns:
            Training metrics
        """
        # Load match data
        df = pd.read_csv(match_data_path)
        
        # Separate features and labels
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        X = df[feature_cols].values
        y = df['action'].values
        
        self.feature_names = feature_cols
        
        # Train-validation split
        split_idx = int(len(X) * (1 - val_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        print(f"Training LightGBM on {len(X_train)} samples...")
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.config.get('num_boost_round', 100),
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=10),
                lgb.log_evaluation(period=10)
            ]
        )
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_acc = np.mean(np.argmax(train_pred, axis=1) == y_train)
        val_acc = np.mean(np.argmax(val_pred, axis=1) == y_val)
        
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")
        
        return {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'num_samples': len(X_train)
        }
    
    def predict_action(self, game_state: Dict[str, Any], 
                      return_probs: bool = False) -> Tuple[int, Optional[np.ndarray]]:
        """
        Predict best action for given game state.
        
        Args:
            game_state: Current game state
            return_probs: Whether to return action probabilities
            
        Returns:
            Predicted action and optionally action probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        features = self.extract_features(game_state).reshape(1, -1)
        probs = self.model.predict(features)[0]
        action = np.argmax(probs)
        
        if return_probs:
            return action, probs
        return action, None
    
    def get_action_distribution(self, game_state: Dict[str, Any]) -> np.ndarray:
        """
        Get probability distribution over actions.
        
        Args:
            game_state: Current game state
            
        Returns:
            Action probability distribution
        """
        features = self.extract_features(game_state).reshape(1, -1)
        return self.model.predict(features)[0]
    
    def save(self, path: str) -> None:
        """Save model to file."""
        if self.model is None:
            raise ValueError("No model to save.")
        
        save_dict = {
            'model': self.model,
            'feature_names': self.feature_names,
            'action_mapping': self.action_mapping,
            'config': self.config
        }
        
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"LightGBM model saved to {path}")
    
    def load(self, path: str) -> None:
        """Load model from file."""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.model = save_dict['model']
        self.feature_names = save_dict['feature_names']
        self.action_mapping = save_dict['action_mapping']
        self.config = save_dict['config']
        
        print(f"LightGBM model loaded from {path}")


def create_synthetic_match_data(num_samples: int = 10000, 
                               num_actions: int = 19) -> pd.DataFrame:
    """
    Create synthetic match data for testing.
    In production, use real match data from StatsBomb or similar sources.
    
    Args:
        num_samples: Number of samples to generate
        num_actions: Number of action classes
        
    Returns:
        DataFrame with synthetic match data
    """
    np.random.seed(42)
    
    data = {
        # Ball features
        'feature_ball_owned_player': np.random.randint(-1, 11, num_samples),
        'feature_ball_owned_team': np.random.randint(-1, 2, num_samples),
        'feature_ball_x': np.random.uniform(-1, 1, num_samples),
        'feature_ball_y': np.random.uniform(-0.42, 0.42, num_samples),
        'feature_ball_z': np.random.uniform(0, 0.5, num_samples),
        
        # Player position
        'feature_player_x': np.random.uniform(-1, 1, num_samples),
        'feature_player_y': np.random.uniform(-0.42, 0.42, num_samples),
        
        # Distances
        'feature_dist_to_ball': np.random.uniform(0, 2, num_samples),
        
        # Opponent
        'feature_opponent_x': np.random.uniform(-1, 1, num_samples),
        'feature_opponent_y': np.random.uniform(-0.42, 0.42, num_samples),
        'feature_dist_to_opponent': np.random.uniform(0, 2, num_samples),
        
        # Game state
        'feature_steps_left': np.random.randint(0, 3001, num_samples),
        'feature_our_score': np.random.randint(0, 5, num_samples),
        'feature_opp_score': np.random.randint(0, 5, num_samples),
        'feature_winning': np.random.randint(0, 2, num_samples),
        'feature_attacking_half': np.random.randint(0, 2, num_samples),
    }
    
    # Generate actions with some logic
    actions = []
    for i in range(num_samples):
        # Simple heuristic: if close to ball, more likely to shoot/pass
        if data['feature_dist_to_ball'][i] < 0.1:
            action = np.random.choice([12, 13, 14], p=[0.5, 0.3, 0.2])  # shoot, pass, dribble
        elif data['feature_attacking_half'][i] == 1:
            action = np.random.choice(range(num_actions), p=np.ones(num_actions)/num_actions)
        else:
            action = np.random.choice([0, 1, 2, 3], p=[0.25]*4)  # movement actions
        actions.append(action)
    
    data['action'] = actions
    
    return pd.DataFrame(data)


if __name__ == "__main__":
    # Example usage
    config = {
        'num_actions': 19,
        'num_leaves': 31,
        'learning_rate': 0.05
    }
    
    advisor = FootballLGBMAdvisor(config)
    
    # Create synthetic data
    print("Creating synthetic match data...")
    df = create_synthetic_match_data(num_samples=10000)
    df.to_csv('data/synthetic_match_data.csv', index=False)
    
    # Train model
    metrics = advisor.train('data/synthetic_match_data.csv')
    print(f"Training metrics: {metrics}")
    
    # Save model
    advisor.save('checkpoints/lgbm_advisor.pkl')