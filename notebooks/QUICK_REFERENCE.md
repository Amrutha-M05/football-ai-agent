# Football AI Agent - Quick Reference Card

## 🚀 Essential Commands

### Setup
```bash
./scripts/setup_project.sh
source venv/bin/activate
```

### Training Commands

| Command | Description |
|---------|-------------|
| `python train.py --agent dqn --episodes 10000` | Train DQN |
| `python train.py --agent ppo --episodes 50000` | Train PPO |
| `python train.py --agent hybrid --use-lgbm` | Train Hybrid |
| `python train.py --config configs/ppo_config.yaml` | Use config file |
| `python train.py --checkpoint path.pth` | Resume training |

### Evaluation Commands

| Command | Description |
|---------|-------------|
| `python evaluate.py --model-path path.pth` | Evaluate model |
| `python evaluate.py --model-path path.pth --render` | With visualization |
| `python evaluate.py --model-path path.pth --episodes 100` | 100 episodes |

### Benchmarking

| Command | Description |
|---------|-------------|
| `python benchmark.py --model-path path.pth --device cpu` | CPU benchmark |
| `python benchmark.py --model-path path.pth --device cuda` | GPU benchmark |

### Utilities

| Command | Description |
|---------|-------------|
| `python scripts/visualize_results.py --log-dir outputs/run/logs` | Plot results |
| `python scripts/export_onnx.py --model-path path.pth` | Export ONNX |
| `tensorboard --logdir outputs/` | View training |
| `pytest tests/ -v` | Run tests |

## 📁 Key Files

| File | Purpose |
|------|---------|
| `train.py` | Main training script |
| `evaluate.py` | Model evaluation |
| `benchmark.py` | Performance benchmarking |
| `configs/*.yaml` | Agent configurations |
| `src/agents/` | Agent implementations |
| `checkpoints/` | Saved models |
| `outputs/` | Training logs |

## 🎯 Common Scenarios

### Start Simple
```bash
python train.py --agent dqn --scenario academy_empty_goal --episodes 1000
```

### Full Training
```bash
python train.py --agent ppo --scenario 11_vs_11_stochastic --episodes 50000 --config configs/ppo_config.yaml
```

### Quick Experiment
```bash
python train.py --agent ppo --episodes 5000 --eval-frequency 500
```

### Hybrid Training
```bash
python train.py --agent hybrid --use-lgbm --lgbm-weight 0.3 --episodes 50000
```

## 📊 Important Parameters

### DQN
- `learning_rate`: 0.0001
- `epsilon_start`: 1.0
- `epsilon_end`: 0.01
- `buffer_size`: 100000
- `batch_size`: 64

### PPO
- `learning_rate`: 0.0003
- `gamma`: 0.99
- `gae_lambda`: 0.95
- `clip_epsilon`: 0.2
- `rollout_length`: 2048

### Environment
- `frame_skip`: 4 (faster) or 1 (precise)
- `reward_shaping`: true (recommended)

## 🏃 Scenarios

| Scenario | Difficulty | Description |
|----------|-----------|-------------|
| `academy_empty_goal` | Easy | Score on empty goal |
| `academy_empty_goal_close` | Easy | Close-range scoring |
| `academy_run_to_score` | Medium | Run and score |
| `academy_3_vs_1_with_keeper` | Medium | 3v1 with keeper |
| `11_vs_11_stochastic` | Hard | Full match |
| `11_vs_11_hard_stochastic` | Very Hard | Hardest scenario |

## 💾 Checkpoints

Saved models location:
- Best model: `checkpoints/{agent}_best.pth`
- Periodic: `checkpoints/{agent}_ep{N}.pth`
- Final: `checkpoints/{agent}_final.pth`

## 📈 Monitoring Training

### TensorBoard
```bash
tensorboard --logdir outputs/
# Open browser: http://localhost:6006
```

### Log Files
- Episodes: `outputs/run/logs/episodes.csv`
- Training: `outputs/run/logs/training_steps.csv`
- Eval: `outputs/run/logs/evaluations.csv`

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| GPU not detected | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| Out of memory | Reduce `batch_size` or `rollout_length` |
| Slow training | Use GPU, increase `frame_skip` |
| Agent not learning | Enable `reward_shaping`, check logs |
| Import errors | Run `pip install -r requirements.txt` |

## 🔧 Configuration Files

Edit `configs/*.yaml` to customize:
- Network architecture (`hidden_dims`)
- Learning rate
- Training episodes
- Reward weights
- Scenario selection

## 📦 Python API

```python
from src.agents import DQNAgent, PPOAgent, HybridAgent
from src.environment import create_football_env

# Create environment
env = create_football_env(scenario='academy_empty_goal')

# Create agent
config = {'learning_rate': 3e-4, 'use_cuda': True}
agent = PPOAgent(state_dim=115, action_dim=19, config=config)

# Training loop
state = env.reset()
action = agent.select_action(state, training=True)
next_state, reward, done, info = env.step(action)
```

## 📊 Performance Metrics

Key metrics tracked:
- **Win Rate**: % of games won
- **Goals/Game**: Average goals scored
- **Episode Reward**: Total reward per episode
- **Episode Length**: Steps per episode
- **FPS**: Inference speed

## 🎓 For Case Study

### Data to Collect
1. Training curves (reward over time)
2. Win rates by agent type
3. CPU vs GPU benchmarks
4. Sample efficiency (reward vs steps)
5. Computational costs

### Visualizations
```bash
python scripts/visualize_results.py --log-dir outputs/YOUR_RUN/logs
```

Generates:
- Training curves
- Reward distributions
- Evaluation metrics
- Summary statistics

## 🌟 Pro Tips

1. **Start small**: Train on `academy_empty_goal` first
2. **Use curriculum**: Progress from easy to hard scenarios
3. **Monitor closely**: Check TensorBoard regularly
4. **Save often**: Use `--save-frequency 500`
5. **GPU training**: 4-5x faster
6. **Batch experiments**: Run multiple seeds

## 📞 Need Help?

1. Check `USAGE_GUIDE.md` for detailed instructions
2. Review `README.md` for project overview
3. Run tests: `pytest tests/ -v`
4. Check logs in `outputs/` directory

## ⚡ Speed Optimizations

| Optimization | Speedup |
|--------------|---------|
| Use GPU | 4-5x |
| Frame skip = 8 | 2x |
| Larger batch | 1.5x |
| Reduce logging | 1.2x |

## 🎯 Success Criteria

**Good Performance:**
- DQN: 60%+ win rate on simple scenarios
- PPO: 75%+ win rate on medium scenarios
- Hybrid: 80%+ win rate with team coordination

**Training Time (GPU):**
- Simple scenarios: 2-4 hours
- Medium scenarios: 6-12 hours
- Full match: 24-48 hours

---

**Keep this handy while working on your project! 🚀**