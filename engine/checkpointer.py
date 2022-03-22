import logging
import os
import torch
from typing import Any, Dict
# from fvcore.common.checkpoint import Checkpointer
from iopath.common.file_io import PathManager as PathManagerBase
from pathlib import Path
from collections import OrderedDict
from utils import comm

class Checkpointer:
    """
        A checkpointer that can save/load model as well as extra checkpointable
        objects.
    """
    def __init__(self, cfg):
        self._save_dir = os.path.join(cfg.training.output_dir, 'models')
        self.logger = logging.getLogger(cfg.logs.name)
        self._checkpoint = {}

    def _save(self, model):
        if comm.get_world_size() > 1:
            return model.module.state_dict()
        else:
            return model.state_dict()

    def save(self,
             name: str,
             model,
             optimizer,
             epoch,
             **kwargs: Any) -> None:
        if not Path(self.save_dir).is_dir():
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self._save(model),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(self.save_dir, f'{name}.pth')
        )
    def _check_model_state_dict(self):
        new_state_dict = OrderedDict()
        # multi-gpus
        if comm.get_world_size() > 1:
            for k, v in self._checkpoint['model_state_dict'].items():
                if 'module' not in k:
                    name = 'module.' + k
                else:
                    name = k
                new_state_dict[name] = v
        #single-gpu
        else:
            for k, v in self._checkpoint['model_state_dict'].items():
                if 'module' in k:
                    name = k[7:]
                else:
                    name = k
                new_state_dict[name] =v
        self._checkpoint['model_state_dict'] = new_state_dict

    def load_model(self, model):
        '''
            Save: only for single-gpu
            but if we train with multi-ple gpu -> we need to start string at 'module'

        '''
        self._check_model_state_dict()
        model.load_state_dict(self._checkpoint['model_state_dict'])

    def load_optimizer(self, optimizer):
        optimizer.load_state_dict(self._checkpoint['optimizer_state_dict'])

    def load(self, path: str) -> None:
        if not path:
            self.logger.info("No checkpoint found. Initializing model from scratch")
        self.logger.info(f"[Checkpointer] Loading from {path} ...")
        if not os.path.isfile(path):
            assert os.path.isfile(path), f"Checkpoint {path} not found!"

        self._checkpoint = self._load_file(path)

    def _load_file(self, path):
        return torch.load(path, map_location=torch.device('cpu'))

    def check_last_epoch(self):
        return self._checkpoint.get('epoch', 0)

    def has_checkpoint(self):
        save_file = os.path.join(self.save_dir, "last_checkpoint.pth")
        return os.path.exists(save_file)

    @property
    def save_dir(self):
        return self._save_dir