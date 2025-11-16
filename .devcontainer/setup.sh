#!/bin/bash

echo "🏗️  Setting up Football AI Agent in Codespaces..."

# Update package lists
sudo apt-get update

# Install essential build tools
echo "📦 Installing build essentials..."
sudo apt-get install -y \
    git \
    cmake \
    build-essential \
    libgl1-mesa-dev \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-ttf-dev \
    libsdl2-gfx-dev \
    libboost-all-dev \
    libdirectfb-dev \
    libst-dev \
    mesa-utils \
    xvfb \
    x11vnc \
    python3-pip \
    python3-dev \
    libpython3-dev \
    swig

# Upgrade pip and setuptools
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install other requirements FIRST (without gfootball)
echo "📚 Installing Python packages..."
pip install torch numpy gym==0.21.0 lightgbm scikit-learn pandas \
    matplotlib seaborn tensorboard plotly pyyaml tqdm \
    opencv-python-headless pillow pytest

# Install gfootball separately with special flags
echo "⚽ Installing Google Research Football..."
pip install --no-cache-dir gfootball

# Verify installation
python3 -c "import gfootball; print('✅ GFootball installed successfully!')" || {
    echo "❌ GFootball installation failed. Trying alternative method..."
    
    # Alternative: Build from source
    cd /tmp
    git clone https://github.com/google-research/football.git
    cd football
    pip install .
    cd /workspaces/football-ai-agent
}

# Create directories
echo "📁 Creating project directories..."
mkdir -p data/{raw,processed,match_data}
mkdir -p checkpoints logs outputs

# Generate synthetic data
echo "🎲 Generating synthetic match data..."
python3 << EOF
try:
    from src.models.lgbm_model import create_synthetic_match_data
    df = create_synthetic_match_data(10000)
    df.to_csv('data/match_data/synthetic_matches.csv', index=False)
    print('✅ Synthetic data created!')
except Exception as e:
    print(f'⚠️  Could not generate synthetic data: {e}')
    print('   You can generate it later after setup')
EOF

echo ""
echo "✅ Setup complete!"
echo "🚀 Ready to train agents!"