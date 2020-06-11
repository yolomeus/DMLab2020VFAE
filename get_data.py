import os

import hydra
from hydra.utils import instantiate, get_original_cwd
from omegaconf import DictConfig


@hydra.main(config_path='conf/data_prepare.yaml')
def get_data(cfg: DictConfig):
    """Download and preprocess all datasets that are specified in the data_prepare config file.
    """
    cwd = get_original_cwd()
    download_basedir = os.path.join(cwd, cfg.download_basedir)
    target_basedir = os.path.join(cwd, cfg.target_basedir)

    for cls in cfg.prepares:
        dl_dir = os.path.join(download_basedir, cls.name)
        target_dir = os.path.join(target_basedir, cls.name)
        prepare = instantiate(cls, download_dir=dl_dir, target_dir=target_dir)
        if cfg.download:
            prepare.dowload()
        prepare.prepare()


if __name__ == '__main__':
    get_data()
