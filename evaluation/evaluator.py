
import torch
import numpy as np
from collections import defaultdict
from .measures import Measures

class Evaluator:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, cfg):
        self.cfg = cfg
        self._measures = Measures(cfg)
        self._results = defaultdict(list)
        self._best_epoch: int = 0
        self._metric: float = 0.0

    def reset(self):
        self._results = defaultdict(list)

    @property
    def results(self):
        return self._results

    def calculate(self):
        for key, value in self._results.items():
            self._results[key] = np.mean(value)

    def _preprocessing(self, masks):
        if not masks.dtype == 'torch.int32':
            ones = torch.ones(masks.shape).to(masks.device)
            zeros = torch.zeros(masks.shape).to(masks.device)
            masks = torch.where(masks > self.cfg.training.evaluation.thresholds.value, ones, zeros)
        return masks

    def process(self, gt_masks, pred_masks):
        gt_masks = self._preprocessing(gt_masks)
        for measure in self.cfg.training.evaluation.measures:
            self._results[measure].append(
                self._measures._metrics[measure](gt_masks, pred_masks)
            )

    def best(self, epoch: int, metric: float):
        self._best_epoch = epoch
        self._metric = metric

    @property
    def best_metric(self):
        return self._metric

    @property
    def best_epoch(self):
        return self._best_epoch