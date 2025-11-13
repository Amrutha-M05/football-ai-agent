"""
Training logger for Football AI Agent
"""
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class TrainingLogger:
    """Logger for training metrics and statistics."""
    
    def __init__(self, log_dir: Path, use_tensorboard: bool = True):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
            use_tensorboard: Whether to use TensorBoard
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loggers
        self.use_tensorboard = use_tensorboard and TENSORBOARD_AVAILABLE
        if self.use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=str(self.log_dir / 'tensorboard'))
        
        # CSV loggers
        self.episode_log_path = self.log_dir / 'episodes.csv'
        self.step_log_path = self.log_dir / 'training_steps.csv'
        self.eval_log_path = self.log_dir / 'evaluations.csv'
        
        # Initialize CSV files
        self._init_csv_files()
        
        # In-memory storage
        self.episode_metrics: List[Dict] = []
        self.step_metrics: List[Dict] = []
        self.eval_metrics: List[Dict] = []
        
        # Counters
        self.global_step = 0
        
    def _init_csv_files(self):
        """Initialize CSV files with headers."""
        # Episodes CSV
        with open(self.episode_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'reward', 'steps', 'score', 'timestamp'])
        
        # Training steps CSV
        with open(self.step_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'loss', 'metric_type', 'value', 'timestamp'])
        
        # Evaluations CSV
        with open(self.eval_log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'mean_reward', 'std_reward', 'win_rate', 'timestamp'])
    
    def log_episode(self, metrics: Dict[str, Any]):
        """
        Log episode-level metrics.
        
        Args:
            metrics: Dictionary containing episode metrics
        """
        timestamp = datetime.now().isoformat()
        metrics['timestamp'] = timestamp
        
        # Store in memory
        self.episode_metrics.append(metrics)
        
        # Write to CSV
        with open(self.episode_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                metrics.get('episode', 0),
                metrics.get('reward', 0),
                metrics.get('steps', 0),
                metrics.get('score', 0),
                timestamp
            ])
        
        # Log to TensorBoard
        if self.use_tensorboard:
            episode = metrics.get('episode', 0)
            self.tb_writer.add_scalar('Episode/Reward', metrics.get('reward', 0), episode)
            self.tb_writer.add_scalar('Episode/Steps', metrics.get('steps', 0), episode)
            self.tb_writer.add_scalar('Episode/Score', metrics.get('score', 0), episode)
    
    def log_step(self, metrics: Dict[str, float]):
        """
        Log training step metrics.
        
        Args:
            metrics: Dictionary containing step metrics (loss, etc.)
        """
        self.global_step += 1
        timestamp = datetime.now().isoformat()
        
        # Store in memory
        metrics['step'] = self.global_step
        metrics['timestamp'] = timestamp
        self.step_metrics.append(metrics)
        
        # Write to CSV
        with open(self.step_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for key, value in metrics.items():
                if key not in ['step', 'timestamp']:
                    writer.writerow([self.global_step, value, key, value, timestamp])
        
        # Log to TensorBoard
        if self.use_tensorboard:
            for key, value in metrics.items():
                if key not in ['step', 'timestamp'] and isinstance(value, (int, float)):
                    self.tb_writer.add_scalar(f'Training/{key}', value, self.global_step)
    
    def log_evaluation(self, episode: int, metrics: Dict[str, float]):
        """
        Log evaluation metrics.
        
        Args:
            episode: Current episode number
            metrics: Dictionary containing evaluation metrics
        """
        timestamp = datetime.now().isoformat()
        metrics['episode'] = episode
        metrics['timestamp'] = timestamp
        
        # Store in memory
        self.eval_metrics.append(metrics)
        
        # Write to CSV
        with open(self.eval_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                metrics.get('mean_reward', 0),
                metrics.get('std_reward', 0),
                metrics.get('win_rate', 0),
                timestamp
            ])
        
        # Log to TensorBoard
        if self.use_tensorboard:
            self.tb_writer.add_scalar('Evaluation/MeanReward', metrics.get('mean_reward', 0), episode)
            self.tb_writer.add_scalar('Evaluation/WinRate', metrics.get('win_rate', 0), episode)
            self.tb_writer.add_scalar('Evaluation/MeanScore', metrics.get('mean_score', 0), episode)
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """
        Log hyperparameters.
        
        Args:
            hparams: Dictionary of hyperparameters
        """
        # Save to JSON
        with open(self.log_dir / 'hyperparameters.json', 'w') as f:
            json.dump(hparams, f, indent=2)
        
        # Log to TensorBoard
        if self.use_tensorboard:
            # TensorBoard expects metric dict for hyperparameters
            # We'll just save a dummy metric
            self.tb_writer.add_hparams(
                hparams,
                {'dummy': 0}  # Placeholder metric
            )
    
    def save_statistics(self, stats: Dict[str, Any]):
        """
        Save final training statistics.
        
        Args:
            stats: Dictionary of statistics
        """
        with open(self.log_dir / 'final_statistics.json', 'w') as f:
            # Convert numpy types to native Python types
            stats_serializable = {}
            for key, value in stats.items():
                if isinstance(value, (np.integer, np.floating)):
                    stats_serializable[key] = float(value)
                elif isinstance(value, np.ndarray):
                    stats_serializable[key] = value.tolist()
                else:
                    stats_serializable[key] = value
            
            json.dump(stats_serializable, f, indent=2)
    
    def close(self):
        """Close logger and cleanup."""
        if self.use_tensorboard:
            self.tb_writer.close()


class ConsoleLogger:
    """Simple console logger for formatted output."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        
    def info(self, message: str):
        """Log info message."""
        if self.verbose:
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] INFO: {message}")
    
    def success(self, message: str):
        """Log success message."""
        if self.verbose:
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] ✓ {message}")
    
    def warning(self, message: str):
        """Log warning message."""
        if self.verbose:
            timestamp = datetime.now().strftime('%H:%M:%S')
            print(f"[{timestamp}] ⚠ WARNING: {message}")
    
    def error(self, message: str):
        """Log error message."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] ✗ ERROR: {message}")
    
    def progress(self, current: int, total: int, prefix: str = ''):
        """Log progress bar."""
        if self.verbose:
            bar_length = 40
            filled = int(bar_length * current / total)
            bar = '█' * filled + '░' * (bar_length - filled)
            percent = 100 * current / total
            print(f"\r{prefix} |{bar}| {percent:.1f}% ({current}/{total})", end='')
            if current == total:
                print()  # New line when complete