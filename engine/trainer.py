
import os
import logging
import datetime
from abc import ABC, abstractmethod
import torch
from torch.nn.parallel import DistributedDataParallel
from utils import comm
from modeling.meta_arch import build_model
from solver.build import build_optimizer, build_loss
from solver.losses import split_losses, cal_losses
from data.build import build_segmentation_test_loader, build_segmentation_loader
from data.datasets.hyundai import Datasets
from engine.checkpointer import Checkpointer
from tqdm import tqdm
from evaluation.evaluator import Evaluator
from solver.losses import LossesInfo
from utils.events import EventStorage

class TrainerBase(ABC):
    def __init__(self):
        pass
    @abstractmethod
    def train(self):
        pass

def create_ddp_model(model, *, fp16_compression=False, **kwargs):
    if comm.get_world_size() == 1:
        return model
    ddp = DistributedDataParallel(model, **kwargs)
    return ddp

class Trainer(TrainerBase):
    def __init__(self, cfg) -> None:
        super(Trainer, self).__init__()
        self.cfg = cfg
        model = self.build_model(cfg)
        model.train()
        self.model = create_ddp_model(model, device_ids=[comm.get_local_rank()], broadcast_buffers=False)
        self.optimizer = self.build_optimizer(cfg, self.model)

        self.data_loader = self.build_train_loader(cfg)

        self.criterion = self.build_loss(cfg)
        self.checkpoint = Checkpointer(cfg)

        self.logger = logging.getLogger(cfg.logs.name)
        self.loss_info = LossesInfo()

        self.evaluator = Evaluator(cfg)
        self.event_storage = EventStorage(cfg)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(self):
        last_epoch = self.checkpoint.check_last_epoch()
        start_time = datetime.datetime.now()
        self.logger.info(f'training start!: {start_time}')
        for epoch in range(last_epoch, self.cfg.training.solver.epochs):
            self.loss_info.reset()
            start_iter = 0
            for batch_index, (inputs, labels) in enumerate(self.data_loader):
                self.optimizer.zero_grad()

                inputs = inputs.cuda()
                labels = labels.cuda()

                saliency_maps, fuse_map = self.model(inputs)
                losses = self.criterion(saliency_maps, fuse_map, labels)

                self.loss_info.add(losses)

                losses['total_loss'].backward()
                if start_iter % self.cfg.logs.steps == 0:
                    _losses = self.loss_info.losses

                    self.logger.info(
                        f'epoch: [{epoch+1}]/[{self.cfg.training.solver.epochs}] '
                        f'[train] steps: [{start_iter}]/[{len(self.data_loader)}] '
                        f'[train] loss: {losses.get("total_loss").item()} '
                        f'[train] saliency loss: {losses.get("saliency_loss").item()} '
                        f'[train] fuse_map loss: {losses.get("fuse_loss").item()}'
                    )

                self.optimizer.step()
                start_iter += 1

            self.loss_info.calculate()
            _losses = self.loss_info.losses
            self.logger.info(
                f'epoch: [{epoch+1}]/[{self.cfg.training.solver.epochs}] '
                f'[train] loss: {_losses.get("total_loss")} '
                f'[train] saliency loss: {_losses.get("saliency_loss")} '
                f'[train] fuse_map loss: {_losses.get("fuse_loss")}'
            )
            self.event_storage.add_train_info(_losses, epoch+1)

            if (
                epoch % self.cfg.training.evaluation.steps == 0
            ):
                self.logger.info(f"=========================== do evaluation ===========================")
                self.model.eval()
                self.val(epoch)
                comm.synchronize()
                self.model.train()

                # save checkpoints
                challenge = self.evaluator.results.get(self.cfg.training.evaluation.select)

                if challenge > self.evaluator.best_metric:
                    self.evaluator.best(epoch, challenge)
                    self.logger.info(f'best epoch: {self.evaluator.best_epoch} '
                                     f'best metric[{self.cfg.training.evaluation.select}]: {self.evaluator.best_metric}')
                    self.checkpoint.save('best_model', model=self.model, optimizer=self.optimizer, epoch = epoch)
                self.checkpoint.save('last_checkpoint', model=self.model, optimizer=self.optimizer, epoch = epoch)
                self.checkpoint.save(f'epoch_{epoch}', model=self.model, optimizer=self.optimizer, epoch = epoch)

                self.evaluator.reset()

        end_time = datetime.datetime.now()
        self.logger.info(f'total training time: {end_time - start_time}')
        comm.synchronize()

    def val(self, epoch):
        data_loader = self.build_val_loader(self.cfg)
        self.loss_info.reset()
        start_iter = 0
        with torch.no_grad():
            for batch_index, (inputs, labels) in enumerate(data_loader):

                inputs = inputs.cuda()
                labels = labels.cuda()

                saliency_maps, fuse_map = self.model(inputs)
                losses = self.criterion(saliency_maps, fuse_map, labels)
                self.loss_info.add(losses)
                self.evaluator.process(gt_masks=labels, pred_masks=fuse_map)

                if start_iter % self.cfg.logs.steps == 0:
                    self.logger.info(
                        f'[val] steps: [{start_iter}]/[{len(data_loader)}]'
                        f'[val] loss: {losses.get("total_loss").item()} '
                        f'[val] saliency loss: {losses.get("saliency_loss").item()} '
                        f'[val] fuse_map loss: {losses.get("fuse_loss").item()}'
                    )
                start_iter += 1

        self.loss_info.calculate()
        _losses = self.loss_info.losses

        self.evaluator.calculate()
        _eval = self.evaluator.results

        if _eval is None:
            _eval = {}

        self.logger.info(
            f'[val] loss: {_losses.get("total_loss")}, [val] saliency_loss: {_losses.get("saliency_loss")}, [val] fuse_loss: {_losses.get("fuse_loss")} '
            f'[val] precision: {_eval.get("precision")}, [val] recall: {_eval.get("recall")}, [val] f_measure: {_eval.get("f_measure")}'
        )
        self.event_storage.add_val_info(_losses, _eval, epoch)

    @classmethod
    def test(cls, cfg, model):
        '''
            inference results with ground truth mask
            if cfg.demo.visualize.save is True
                will save inference results and gt in cfg.training.output_dir > results

            :param cfg:
            :param model:
            :param evaluator:
        :return:
        '''
        evaluator = Evaluator(cfg)
        logger = logging.getLogger(cfg.logs.name)
        data_loader = cls.build_test_loader(cfg)

        with torch.no_grad():
            for batch_index, (inputs, labels) in enumerate(data_loader):

                inputs = inputs.cuda()
                labels = labels.cuda()

                saliency_maps, fuse_map = model(inputs)
                evaluator.process(gt_masks=labels, pred_masks=fuse_map)

        evaluator.calculate()
        _eval = evaluator.results

        if _eval is None:
            _eval = {}

        logger.info(
            f'[test] precision: {_eval.get("precision")}, [test] recall: {_eval.get("recall")}, [test] f_measure: {_eval.get("f_measure")}'
        )
        evaluator.reset()

    def resume_or_load(self):
        if self.checkpoint.has_checkpoint():
            model_path = os.path.join(self.checkpoint.save_dir, 'last_checkpoint.pth')
            self.checkpoint.load(model_path)
            self.checkpoint.load_model(self.model)
            self.checkpoint.load_optimizer(self.optimizer)
        else:
            self.logger.info(f'there is no checkpoint file, just start from to scratch')

    @classmethod
    def build_loss(cls, cfg):
        losses = build_loss(cfg)
        return losses

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        return model

    @classmethod
    def build_optimizer(cls, cfg, model):
        return build_optimizer(cfg, model.parameters())

    @classmethod
    def build_train_loader(cls, cfg):
        datasets = Datasets(cfg).get_train
        return build_segmentation_loader(cfg, datasets)

    @classmethod
    def build_val_loader(cls, cfg):
        datasets = Datasets(cfg).get_val
        return build_segmentation_loader(cfg, datasets)

    @classmethod
    def build_test_loader(cls, cfg):
        datasets = Datasets(cfg).get_test
        return build_segmentation_loader(cfg, datasets)