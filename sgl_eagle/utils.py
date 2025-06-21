import os
from omegaconf import OmegaConf


def load_config(config_path: str):
    """
    Load the config from the given path. 

    Args:
        config_path (str): The path to the config file.
    """
    config = OmegaConf.load(config_path)
    base_config_path = config.base

    # check if base_config_path is a relative path
    if not os.path.isabs(base_config_path):
        base_config_path = os.path.join(os.path.dirname(config_path), base_config_path)
    base_config = OmegaConf.load(base_config_path)

    # merge config
    config = OmegaConf.merge(base_config, config)
    return config


def build_optimizer(model, config):
    """
    Build the optimizer from the config.
    """
    from torch import optim
    optim_cls = getattr(optim, config.type)
    optim_kwargs = config.copy().pop('type')
    return optim_cls(model.parameters(), **optim_kwargs)\

def build_criterion(config):
    """
    Build the criterion from the config.
    """
    from torch import nn
    criterion_cls = getattr(nn, config.type)
    criterion_kwargs = config.copy().pop('type')
    return criterion_cls(**criterion_kwargs)