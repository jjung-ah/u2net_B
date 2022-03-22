import torch
import torch.nn as nn
from .build import SOLVER_REGISTRY

@SOLVER_REGISTRY.register()
def Leaky_relu(cfg):
    return nn.LeakyReLU(
        negative_slope=cfg.meta_arch.activate.Leaky_relu.negative_slope,
        inplace=cfg.meta_arch.activate.Leaky_relu.inplace
    )

@SOLVER_REGISTRY.register()
def relu(cfg):
    return nn.ReLU(
        inplace=cfg.meta_arch.activate.Leaky_relu.inplace
    )