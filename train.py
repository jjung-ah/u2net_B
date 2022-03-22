import logging
import os
import torch
import hydra
from hydra import initialize_config_dir, compose
from utils.logger import Logger
from omegaconf import DictConfig
from engine.trainer import Trainer
from engine.launch import launch
from engine.checkpointer import Checkpointer
from utils import comm

def setup(cfg):
    cfg.logs.root_dir = os.path.join(cfg.training.output_dir, 'logs')
    cfg.training.model.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.training.solver.image_per_batch = cfg.training.mp.num_gpus * cfg.training.solver.image_per_batch
    return cfg

def main(cfg):
    cfg = setup(cfg)

    log_storage = Logger(cfg)
    log_storage.start()

    if cfg.demo.evaluation:
        model = Trainer.build_model(cfg)

        checkpointer = Checkpointer(cfg)
        checkpointer.load(path=cfg.demo.weight)
        checkpointer.load_model(model)
        Trainer.test(cfg, model)

        return log_storage.stop()

    trainer = Trainer(cfg)
    trainer.resume_or_load()
    trainer.train()
    return log_storage.stop()


if __name__ == '__main__':
    abs_config_dir = os.path.abspath("./configs")
    with initialize_config_dir(config_dir=abs_config_dir):
        cfg = compose(config_name="config.yaml")
    launch(
        main,
        cfg.training.mp.num_gpus,
        num_machines=cfg.training.mp.num_machines,
        machine_rank=cfg.training.mp.machine_rank,
        dist_url=cfg.training.mp.dist_url,
        cfg = cfg
    )