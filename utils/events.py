
from torch.utils.tensorboard import SummaryWriter

# for tensorboard
class EventStorage:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, cfg):
        self.cfg = cfg
        cls = type(self)
        if not hasattr(cls, "_init"):
            self._writer = SummaryWriter(log_dir=cfg.training.output_dir)
        cls._init = True

    def add_train_info(self, losses, epoch):
        for name, loss in losses.items():
            self._writer.add_scalar(f"Loss-{name}/Train", loss, epoch)

    def add_val_info(self, losses, evaluations, epoch):
        for name, loss in losses.items():
            self._writer.add_scalar(f'Loss-{name}/Val', loss, epoch)

        for name, value in evaluations.items():
            self._writer.add_scalar(f'Eval-{name}/Val', value, epoch)