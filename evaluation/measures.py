
import numpy as np
from collections import defaultdict
import torch

class Measures:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, cfg):
        self.cfg = cfg
        self._metrics = defaultdict(object)
        self._initialize_metrics(cfg)
        self.thresholds = torch.linspace(0, 1-cfg.training.evaluation.thresholds.smooth, cfg.training.evaluation.thresholds.nums)

    def _initialize_metrics(self, cfg):
        for name in cfg.training.evaluation.measures:
            self._metrics[name] = getattr(self, name)

    def _tp(self, gt_masks, pred_masks):
        return torch.sum(gt_masks * pred_masks).item()

    def _fp(self, gt_masks, pred_masks):
        gt_masks = ~gt_masks.int() + 2
        return torch.sum(gt_masks * pred_masks).item()

    def _fn(self, gt_masks, pred_masks):
        pred_masks = ~pred_masks.int() + 2
        return torch.sum(gt_masks * pred_masks).item()

    def precision(self, gt_masks, pred_masks):
        '''
            :param gt_masks:
            :param pred_masks:

            precision = TP / TP + FP
                FP: real = False but pred = True
        :return:
        '''
        self._precision = list()
        for i in range(self.cfg.training.evaluation.thresholds.nums):
            pred_mask = (pred_masks > self.thresholds[i]).float()
            tp = self._tp(gt_masks, pred_mask)
            fp = self._fp(gt_masks, pred_mask)

            try:
                results = tp/(tp+fp)
            except Exception as ZeroDivisionError:
                results = 0
                pass

            self._precision.append(results)
        return np.mean(self._precision)

    def recall(self, gt_masks, pred_masks):
        '''

        :return:

            recall = TP / TP + FN
                FN : real = True but pred = False

        '''
        self._recall = list()
        for i in range(self.cfg.training.evaluation.thresholds.nums):
            pred_mask = (pred_masks > self.thresholds[i])
            tp = self._tp(gt_masks, pred_mask)
            fn = self._fn(gt_masks, pred_mask)

            try:
                results = tp/(tp+fn)
            except ZeroDivisionError as e:
                results = 0
                pass
            self._recall.append(results)
        return np.mean(self._recall)

    def _f_measure(self, recall, precision):
        try:
            return ((1+self.cfg.training.evaluation.f_measure.beta) * precision * recall) / (self.cfg.training.evaluation.f_measure.beta*precision + recall)
        except ZeroDivisionError as e:
            return 0

    def f_measure(self, gt_masks, pred_masks):
        '''

            (1+beta**2) x precision x recall
            -------------------------------- = F_beta
            beta**2 x precision + recall
        :return:
        '''
        f_scores = []
        for recall, precision in zip(self._recall, self._precision):
            f_scores.append(self._f_measure(recall, precision))

        return np.max(f_scores)

    def mae(self, gt_masks, pred_masks):
        return torch.abs(pred_masks - gt_masks).mean().item()
