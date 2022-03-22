
from data.transforms.transform import (
    Compose,
    ToPILImage,
    Resize,
    RandomVerticalFlip,
    ToTensor
)
import torchvision.transforms as T

def build_train_val_transforms(cfg):
    return Compose([
        ToPILImage(),
        Resize(cfg.training.transforms.resize.shape, cfg.training.transforms.resize.mode),
        RandomVerticalFlip(cfg.training.transforms.random_vertical_flip.probability),
        ToTensor()
    ])

def build_test_transforms(cfg):
    return T.Compose([
        T.ToPILImage(),
        T.Resize(size=(cfg.training.transforms.resize.shape,cfg.training.transforms.resize.shape), interpolation=cfg.training.transforms.resize.mode),
        T.ToTensor()
    ])

