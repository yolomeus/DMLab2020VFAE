import os

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything

from lightning_wrapper import LightningModel


@hydra.main(config_path='conf/config.yaml')
def test(cfg: DictConfig):
    """Test a pytorch model specified by the config file"""

    seed_everything(cfg.random_seed)

    ckpt_path = os.path.join(get_original_cwd(), cfg.testing.checkpoint)
    model = LightningModel.load_from_checkpoint(ckpt_path)
    # make sure we're using the current test config and not the saved one
    model.test_conf = cfg.testing
    trainer = Trainer(max_epochs=cfg.testing.epochs, gpus=cfg.gpus, deterministic=True)
    trainer.test(model)


if __name__ == '__main__':
    test()
