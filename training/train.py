import hydra
import wandb
from omegaconf import DictConfig, OmegaConf

from model import TinyConfig


@hydra.main(version_base=None, config_path="configs", config_name="pretrain_config")
def train(cfg: DictConfig):
    pass

if __name__ == "__main__":
    train()