
import os

import torch

from engine.checkpointer import Checkpointer
from modeling.meta_arch.build import build_model
from data.transforms.build import build_test_transforms

class DefaultPredictor:
    def __init__(self, cfg):
        self.model = build_model(cfg)
        self.checkpointer = Checkpointer(cfg)
        self.checkpointer.load(path=cfg.demo.weight)
        self.checkpointer.load_model(self.model)
        self.transforms = self.build_test_transforms(cfg)
        self.model.eval()
        self.cfg = cfg

    @classmethod
    def build_test_transforms(cls, cfg):
        return build_test_transforms(cfg)

    @classmethod
    def _squeeze(self, targets: list):
        new = []
        for sample in targets:
            new.append(
                sample.squeeze(0).to('cpu').detach().numpy().transpose(1,2,0)
            )
        if len(targets) == 1:
            new = new[-1]
        return new

    def __call__(self, image, mask=None):
        with torch.no_grad():

            if mask is not None:
                mask = self.transforms(mask).unsqueeze(0).to(self.cfg.training.model.device)
                mask = self._squeeze([mask])

            image = self.transforms(image).unsqueeze(0).to(self.cfg.training.model.device)

            saliency_maps, fuse_map = self.model(image)
            saliency_maps = self._squeeze(saliency_maps)
            fuse_map = self._squeeze([fuse_map])

            image = self._squeeze([image])

            return saliency_maps, fuse_map, image, mask