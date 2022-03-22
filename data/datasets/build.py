
from fvcore.common.registry import Registry
from omegaconf import DictConfig
from data.transforms.build import build_train_val_transforms, build_test_transforms
DATASET_REGISTRY = Registry("DATASET")
DATASET_REGISTRY.__doc__ = """ Registry for DATASET """

def build_train_val_datasets(cfg: DictConfig, datasets):
    name = cfg.training.datasets.name
    transforms = build_train_val_transforms(cfg)
    datasets = DATASET_REGISTRY.get(name)(cfg, datasets, transforms)
    return datasets

def build_test_datasets(cfg: DictConfig, datasets):
    name = cfg.training.datasets.name
    transforms = build_test_transforms(cfg)
    datasets = DATASET_REGISTRY.get(name)(cfg, datasets, transforms)
    return datasets