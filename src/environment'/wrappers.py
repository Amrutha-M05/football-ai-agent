"""
Environment wrappers for Google Research Football
"""
import gym
import numpy as np
from typing import Dict, Any, Tuple


class FootballEnvWrapper(gym.Wrapper):
    """
    Wrapper for Football environment with state preprocessing.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._current_obs = None
        
    def reset(self):
        """Reset environment."""
        obs = self.env.reset()
        self._current_obs = obs
        return self._preprocess_observation(obs)
    
    def step(self, action):
        """Take action in environment."""
        obs, reward, done, info = self.env.step(action)
        self._current_obs = obs
        processed_obs = self._preprocess_observation(obs)
        return processed_obs, reward, done, info
    
    def _preprocess_observation(self, obs):
        """Preprocess raw observation."""
        # Flatten if needed
        if isinstance(obs, dict):
            # Extract relevant features
            processed = obs
        else:
            processed = obs
        
        # Normalize if numpy array
        if isinstance(processed, np.ndarray):
            # Clip extreme values
            processed = np.clip(processed, -10, 10)
        
        return processed
    
    def get_game_state(self) -> Dict[str, Any]:
        """
        Get structured game state for LightGBM.
        This is a simplified representation.
        """
        obs = self._current_obs
        
        if obs is None:
            return {}
        
        # Parse observation based on 'simple115v2' representation
        # This is a simplified version - adjust based on actual observation format
        game_state = {
            'ball_owned_player': -1,  # Placeholder
            'ball_owned_team': -1,
            'ball': [0.0, 0.0, 0.0],
            'left_team': [[0.0, 0.0]],
            'right_team': [[0.0, 0.0]],
            'steps_left': 3000,
            'score': [0, 0]
        }
        
        # Try to extract real values if observation is structured
        if isinstance(obs, dict):
            game_state.update(obs)
        elif isinstance(obs, np.ndarray) and len(obs) >= 115:
            # Parse simple115v2 format
            # This is approximate - adjust based on actual format
            game_state['ball'] = obs[88:91].tolist() if len(obs) > 90 else [0, 0, 0]
            game_state['left_team'] = [obs[0:2].tolist()]
            game_state['score'] = [int(obs[114]), int(obs[115])] if len(obs) > 115 else [0, 0]
        
        return game_state


class RewardShaper(gym.Wrapper):
    """
    Wrapper for reward shaping in football environment.
    """
    
    def __init__(self, env, config: Dict[str, Any]):
        super().__init__(env)
        self.config = config
        
        # Reward shaping parameters
        self.use_shaping = config.get('reward_shaping', True)
        self.possession_reward = config.get('possession_reward', 0.001)
        self.movement_reward = config.get('movement_reward', 0.0001)
        self.goal_reward = config.get('goal_reward', 1.0)
        self.win_reward = config.get('win_reward', 5.0)
        self.lose_penalty = config.get('lose_penalty', -1.0)
        
        # State tracking
        self._last_ball_position = None
        self._last_player_position = None
        self._last_score = [0, 0]
        
    def reset(self):
        """Reset environment and tracking variables."""
        obs = self.env.reset()
        self._last_ball_position = None
        self._last_player_position = None
        self._last_score = [0, 0]
        return obs
    
    def step(self, action):
        """Take action and apply reward shaping."""
        obs, reward, done, info = self.env.step(action)
        
        if self.use_shaping:
            reward = self._shape_reward(obs, reward, done, info)
        
        return obs, reward, done, info
    
    def _shape_reward(self, obs, base_reward, done, info):
        """
        Apply reward shaping based on game state.
        
        Args:
            obs: Current observation
            base_reward: Original environment reward
            done: Whether episode is done
            info: Info dictionary
            
        Returns:
            Shaped reward
        """
        shaped_reward = base_reward
        
        # Extract game state
        game_state = self.env.get_game_state() if hasattr(self.env, 'get_game_state') else {}
        
        # Ball possession reward
        if game_state.get('ball_owned_team', -1) == 0:  # Our team
            shaped_reward += self.possession_reward
        
        # Progress toward goal reward
        ball_pos = game_state.get('ball', [0, 0, 0])
        if self._last_ball_position is not None:
            # Reward for moving ball toward opponent's goal
            progress = ball_pos[0] - self._last_ball_position[0]
            if progress > 0:
                shaped_reward += self.movement_reward * progress
        self._last_ball_position = ball_pos
        
        # Goal rewards
        current_score = game_state.get('score', [0, 0])
        if current_score[0] > self._last_score[0]:  # We scored
            shaped_reward += self.goal_reward
        elif current_score[1] > self._last_score[1]:  # Opponent scored
            shaped_reward += self.lose_penalty
        self._last_score = current_score
        
        # Win/loss rewards
        if done:
            if current_score[0] > current_score[1]:  # We won
                shaped_reward += self.win_reward
            elif current_score[1] > current_score[0]:  # We lost
                shaped_reward += self.lose_penalty
        
        return shaped_reward


class FrameSkipWrapper(gym.Wrapper):
    """
    Wrapper to skip frames for computational efficiency.
    Repeats actions for N frames.
    """
    
    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        self.skip = skip
        
    def step(self, action):
        """Repeat action for skip frames."""
        total_reward = 0.0
        done = False
        info = {}
        
        for _ in range(self.skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        
        return obs, total_reward, done, info


class StackedFramesWrapper(gym.Wrapper):
    """
    Wrapper to stack consecutive frames for temporal information.
    """
    
    def __init__(self, env, num_stack: int = 4):
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = []
        
        # Update observation space
        low = np.repeat(env.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(env.observation_space.high[np.newaxis, ...], num_stack, axis=0)
        self.observation_space = gym.spaces.Box(
            low=low.flatten(),
            high=high.flatten(),
            dtype=env.observation_space.dtype
        )
    
    def reset(self):
        """Reset and initialize frame stack."""
        obs = self.env.reset()
        self.frames = [obs] * self.num_stack
        return self._get_stacked_obs()
    
    def step(self, action):
        """Step and update frame stack."""
        obs, reward, done, info = self.env.step(action)
        self.frames.append(obs)
        self.frames.pop(0)
        return self._get_stacked_obs(), reward, done, info
    
    def _get_stacked_obs(self):
        """Stack frames along new axis."""
        return np.concatenate(self.frames, axis=0)


class MultiAgentWrapper(gym.Wrapper):
    """
    Wrapper for multi-agent coordination.
    Allows controlling multiple players simultaneously.
    """
    
    def __init__(self, env, num_agents: int = 3):
        super().__init__(env)
        self.num_agents = num_agents
        
        # Expand action space for multiple agents
        # Each agent can take one of the original actions
        # Total action space becomes combinatorial
        self.single_action_dim = env.action_space.n
        
    def reset(self):
        """Reset environment."""
        return self.env.reset()
    
    def step(self, action):
        """
        Step with multi-agent action.
        For simplicity, we control one agent at a time in round-robin fashion.
        
        Args:
            action: Action index for current controlled agent
        """
        # This is a simplified version
        # Full multi-agent would require environment modifications
        return self.env.step(action)
    
    def get_controlled_agent_id(self) -> int:
        """Get ID of currently controlled agent."""
        # Placeholder - would need actual implementation
        return 0


def create_football_env(scenario: str = 'academy_empty_goal',
                       frame_skip: int = 4,
                       stack_frames: int = 1,
                       reward_shaping: bool = True,
                       multi_agent: bool = False,
                       **kwargs) -> gym.Env:
    """
    Create wrapped football environment.
    
    Args:
        scenario: Football scenario name
        frame_skip: Number of frames to skip
        stack_frames: Number of frames to stack
        reward_shaping: Whether to apply reward shaping
        multi_agent: Whether to use multi-agent mode
        **kwargs: Additional arguments for env creation
        
    Returns:
        Wrapped environment
    """
    import gfootball.env as football_env
    
    # Create base environment
    env = football_env.create_environment(
        env_name=scenario,
        stacked=False,
        representation='simple115v2',
        **kwargs
    )
    
    # Apply wrappers
    env = FootballEnvWrapper(env)
    
    if reward_shaping:
        env = RewardShaper(env, {
            'reward_shaping': True,
            'possession_reward': 0.001,
            'movement_reward': 0.0001,
            'goal_reward': 1.0,
            'win_reward': 5.0,
            'lose_penalty': -1.0
        })
    
    if frame_skip > 1:
        env = FrameSkipWrapper(env, skip=frame_skip)
    
    if stack_frames > 1:
        env = StackedFramesWrapper(env, num_stack=stack_frames)
    
    if multi_agent:
        env = MultiAgentWrapper(env)
    
    return env