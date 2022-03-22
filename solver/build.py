
import torch
from fvcore.common.registry import Registry

SOLVER_REGISTRY = Registry("SOLVER")
SOLVER_REGISTRY.__doc__ = """ Registry for Solver """

def build_optimizer(cfg, model_paramters: torch.nn.Module):
    name = cfg.training.solver.optimizer.name
    optims = SOLVER_REGISTRY.get(name)(model_paramters, cfg)
    return optims


def build_loss(cfg):
    name = cfg.training.solver.loss.name
    criterion = SOLVER_REGISTRY.get(name)(cfg)
    return criterion


def build_activate(cfg):
    name = cfg.training.solver.activate.name
    activation = SOLVER_REGISTRY.get(name)(cfg)
    return activation