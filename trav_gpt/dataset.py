import os
from pathlib import Path

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
import requests

from trav_gpt import ROOT_DIR


def download_dataset(cfg: DictConfig) -> Path:
    """
    Downloads the dataset from the URL specified in the config.

    Args:
        cfg: Configuration object containing dataset information

    Returns:
        Path to the downloaded file
    """
    # Create the external data directory if it doesn't exist
    external_dir = Path(cfg.paths.external)
    external_dir.mkdir(parents=True, exist_ok=True)

    # Define the output file path
    output_path = external_dir / cfg.dataset.filename

    # Download the file if it doesn't already exist
    if not output_path.exists():
        print(f"Downloading dataset from {cfg.dataset.url}...")
        try:
            response = requests.get(cfg.dataset.url)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Save the content to the output file
            with open(output_path, "wb") as f:
                f.write(response.content)
            print(f"Dataset downloaded to {output_path}")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading dataset: {e}")
            raise
    else:
        print(f"Dataset already exists at {output_path}")

    return output_path


if __name__ == "__main__":
    with initialize(config_path="../conf", version_base=None):
        cfg = compose(config_name="config")
    cfg.paths.root = ROOT_DIR

    # Download the dataset
    download_dataset(cfg)
