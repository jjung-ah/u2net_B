

import torch
import torch.optim as optim
from .build import SOLVER_REGISTRY

@SOLVER_REGISTRY.register()
def Adam(model_parameters, cfg) -> torch.optim.Optimizer:
    return optim.Adam(
        model_parameters,
        lr=cfg.meta_arch.optimizer.Adam.learning_rate,
        weight_decay=cfg.meta_arch.optimizer.Adam.weight_decay,
        eps=cfg.meta_arch.optimizer.Adam.eps
    )

@SOLVER_REGISTRY.register()
def SGD(model_parameters, cfg) -> torch.optim.Optimizer:
    return optim.SGD(
        model_parameters,
        lr=cfg.meta_arch.optimizer.SGD.learning_rate,
        momentum=cfg.meta_arch.optimizer.SGD.momentum,
        dampening=cfg.meta_arch.optimizer.SGD.dampening,
        weight_decay=cfg.meta_arch.optimizer.SGD.weight_decay,
        nesterov=cfg.meta_arch.optimizer.SGD.nesterov
    )

