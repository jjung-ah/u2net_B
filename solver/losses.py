
import torch
import numpy as np
from .build import SOLVER_REGISTRY
from collections import defaultdict

@SOLVER_REGISTRY.register()
def default(cfg):
    criterion = torch.nn.BCELoss()
    def loss(saliency_maps: list, fuse_map, ground_truth):
        fuse_loss = criterion(fuse_map, ground_truth)
        saliency_loss = 0.
        for saliency_map in saliency_maps:
            loss = criterion(saliency_map, ground_truth)
            saliency_loss += loss

        return {
            'saliency_loss': saliency_loss,
            'fuse_loss': fuse_loss,
            'total_loss': saliency_loss + fuse_loss
        }

    return loss

def split_losses(losses: dict):
    saliency_loss = losses.get('saliency_loss').item()
    fuse_loss = losses.get('fuse_loss').item()
    total_loss = losses.get('total_loss').item()

    return saliency_loss, fuse_loss, total_loss

def cal_losses(total, saliency, fuse, nums):
    return total / nums, saliency / nums, fuse / nums

class LossesInfo:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._losses = defaultdict(list)

    @property
    def losses(self):

        return self._losses

    def reset(self):
        self._losses = defaultdict(list)

    def add(self, losses: dict):
        for key, value in losses.items():
            self._losses[key].append(value.item())

    def calculate(self):
        for key, value in self._losses.items():
            self._losses[key] = np.mean(value)
