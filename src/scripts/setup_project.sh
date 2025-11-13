#!/bin/bash
# Setup script for Football AI Agent project

echo "================================================"
echo "Football AI Agent - Project Setup"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Found Python $python_version"

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
    echo -e "${RED}Error: Python 3.8 or higher is required${NC}"
    exit 1
fi

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created${NC}"
else
    echo -e "${GREEN}Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Install Google Research Football
echo -e "\n${YELLOW}Installing Google Research Football Environment...${NC}"
pip install gfootball

# Create necessary directories
echo -e "\n${YELLOW}Creating project directories...${NC}"
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/match_data
mkdir -p checkpoints
mkdir -p logs
mkdir -p outputs
mkdir -p evaluation_results

echo -e "${GREEN}Directories created${NC}"

# Generate synthetic match data for LightGBM
echo -e "\n${YELLOW}Generating synthetic match data...${NC}"
python -c "
from src.models.lgbm_model import create_synthetic_match_data
import pandas as pd
print('Generating 10,000 synthetic match samples...')
df = create_synthetic_match_data(num_samples=10000)
df.to_csv('data/match_data/synthetic_matches.csv', index=False)
print('Synthetic data saved to data/match_data/synthetic_matches.csv')
"

# Download pre-trained models (if available)
echo -e "\n${YELLOW}Checking for pre-trained models...${NC}"
if [ -f "scripts/download_models.py" ]; then
    python scripts/download_models.py
else
    echo "No download script found. Skipping pre-trained models."
fi

# Run tests
echo -e "\n${YELLOW}Running tests...${NC}"
if [ -d "tests" ]; then
    python -m pytest tests/ -v
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
    else
        echo -e "${YELLOW}Some tests failed. Please review.${NC}"
    fi
else
    echo "No tests directory found. Skipping tests."
fi

# Check GPU availability
echo -e "\n${YELLOW}Checking GPU availability...${NC}"
python -c "
import torch
if torch.cuda.is_available():
    print(f'GPU available: {torch.cuda.get_device_name(0)}')
    print(f'CUDA version: {torch.version.cuda}')
else:
    print('No GPU available. Will use CPU for training.')
"

# Setup complete
echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}Setup completed successfully!${NC}"
echo -e "${GREEN}================================================${NC}"

echo -e "\n${YELLOW}Quick Start Commands:${NC}"
echo "  # Train DQN agent:"
echo "  python train.py --agent dqn --scenario academy_empty_goal --episodes 10000"
echo ""
echo "  # Train PPO agent:"
echo "  python train.py --agent ppo --scenario 11_vs_11_stochastic --episodes 50000"
echo ""
echo "  # Train Hybrid agent (PPO + LightGBM):"
echo "  python train.py --agent hybrid --use-lgbm --episodes 50000"
echo ""
echo "  # Evaluate trained model:"
echo "  python evaluate.py --model-path checkpoints/ppo_best.pth --episodes 100 --render"
echo ""
echo "  # Benchmark performance:"
echo "  python benchmark.py --model-path checkpoints/ppo_best.pth --device cpu"
echo ""

echo -e "\n${YELLOW}Documentation:${NC}"
echo "  - Training Guide: docs/training_guide.md"
echo "  - API Reference: docs/api_reference.md"
echo "  - Architecture: docs/architecture.md"
echo ""

echo -e "${GREEN}Happy training! ⚽🤖${NC}"