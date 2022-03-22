
import torch
from fvcore.common.registry import Registry

LAYER_REGISTRY = Registry("LAYER")
LAYER_REGISTRY.__doc__ = """ Registry for Solver """


def build_downsample(cfg):
    name = cfg.training.layer.down_sample.name
    down = LAYER_REGISTRY.get(name)()
    return down


def build_upsample(cfg):
    name = cfg.training.layer.up_sample.name
    up = LAYER_REGISTRY.get(name)()
    return up