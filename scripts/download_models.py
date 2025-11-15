"""
Download pre-trained models and datasets
"""
import os
import urllib.request
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """
    Download file from URL with progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save file
    """
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_pretrained_models():
    """Download pre-trained models."""
    models_dir = Path('checkpoints')
    models_dir.mkdir(exist_ok=True)
    
    # Note: Replace these URLs with actual model hosting URLs
    # This is a template - you'll need to host models somewhere
    models = {
        'dqn_pretrained.pth': 'https://example.com/models/dqn_pretrained.pth',
        'ppo_pretrained.pth': 'https://example.com/models/ppo_pretrained.pth',
        'lgbm_advisor.pkl': 'https://example.com/models/lgbm_advisor.pkl',
    }
    
    print("Downloading pre-trained models...")
    print("=" * 60)
    
    for filename, url in models.items():
        output_path = models_dir / filename
        
        if output_path.exists():
            print(f"✓ {filename} already exists, skipping...")
            continue
        
        try:
            print(f"\nDownloading {filename}...")
            download_url(url, str(output_path))
            print(f"✓ Downloaded {filename}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            print(f"  You can manually download from: {url}")
    
    print("\n" + "=" * 60)
    print("Download complete!")


def download_sample_data():
    """Download sample match data."""
    data_dir = Path('data/match_data')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Note: Replace with actual data URLs
    datasets = {
        'sample_matches.csv': 'https://example.com/data/sample_matches.csv',
    }
    
    print("\nDownloading sample datasets...")
    print("=" * 60)
    
    for filename, url in datasets.items():
        output_path = data_dir / filename
        
        if output_path.exists():
            print(f"✓ {filename} already exists, skipping...")
            continue
        
        try:
            print(f"\nDownloading {filename}...")
            download_url(url, str(output_path))
            print(f"✓ Downloaded {filename}")
        except Exception as e:
            print(f"✗ Failed to download {filename}: {e}")
            
            # Generate synthetic data as fallback
            print(f"  Generating synthetic data instead...")
            from src.models.lgbm_model import create_synthetic_match_data
            df = create_synthetic_match_data(num_samples=10000)
            df.to_csv(output_path, index=False)
            print(f"✓ Generated synthetic data at {output_path}")
    
    print("\n" + "=" * 60)
    print("Data download complete!")


def main():
    """Main download function."""
    print("\nFootball AI Agent - Model & Data Downloader")
    print("=" * 60)
    
    # Check if running from project root
    if not Path('requirements.txt').exists():
        print("Error: Please run this script from the project root directory")
        return
    
    # Download models
    download_pretrained_models()
    
    # Download data
    download_sample_data()
    
    print("\n" + "=" * 60)
    print("All downloads complete!")
    print("\nNote: Some downloads may have failed if URLs are not configured.")
    print("Pre-trained models are optional - you can train from scratch.")
    print("=" * 60)


if __name__ == "__main__":
    main()