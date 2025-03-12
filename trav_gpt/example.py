from pathlib import Path

import hydra
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from trav_gpt import ROOT_DIR


# @hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(f"Config directory: {cfg.paths.cwd}")
    # print(f"Parent directory: {cfg.paths.config_parent_dir}")


if __name__ == "__main__":
    with initialize(config_path="../conf", version_base=None):
        cfg = compose(config_name="config")
    cfg.paths.root = ROOT_DIR

    print(OmegaConf.to_yaml(cfg, resolve=True))
