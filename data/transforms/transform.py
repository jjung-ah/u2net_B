import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms.functional import _interpolation_modes_from_int
import random
import numpy as np
from utils.defaults import resize, binary_mask

class Compose(object):
    """
        call 로 전달받은 image 와 mask에 동시에 같은 augmentation function 적용할 수 있게 변경
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask

class ToTensor(object):
    """
        기본 ToTensor 함수에, call function parameters 만 변경
    """
    def __call__(self, image, mask):
        return F.to_tensor(image), F.to_tensor(mask)

class Resize(object):
    """
        기본 Resize 함수에, call function parameters 만 변경
    """
    def __init__(self, size, interpolation):
        self.size = size
        self.interpolation = _interpolation_modes_from_int(interpolation)

    def __call__(self, image, mask):
        # F.resize(size=(H,W))
        return F.resize(image, (self.size, self.size), self.interpolation), F.resize(mask, (self.size, self.size), self.interpolation)

class ToPILImage(object):
    """
        기본 ToPILIMage 함수에, call function parameters 만 변경
    """
    def __init__(self, mode=None):
        self.mode = mode
    def __call__(self, image, mask):
        return F.to_pil_image(image), F.to_pil_image(mask)

class RandomVerticalFlip(object):
    """
        기본 RandomVerticalFlip 함수에, call function parameters 만 변경
    """
    def __init__(self, ratio: float):
        self.ratio = ratio

    def __call__(self, image, mask):
        if random.random() > self.ratio:
            image = F.vflip(image)
            mask = F.vflip(mask)
        return image, mask

# TODO CHECK!
def inverse_normalize(configs: dict):
    """
        :param configs:
    :return: inverse normalize transforms
    """

    transforms = T.Normalize(
        mean = [-m / s for m, s in zip(configs["TRANSFORMS"]["PIXEL_MEAN"], configs["TRANSFORMS"]["PIXEL_STD"])],
        std = [ 1 / s for s in configs["TRANSFORMS"]["PIXEL_STD"]]
        )

    return transforms

def normalize_masks(masks: list):
    normalized_masks = []
    for mask in masks:
        _min = np.min(mask)
        _max = np.max(mask)
        normalized_masks.append((mask - _min) / (_max - _min))

    if len(normalized_masks) == 1:
        normalized_masks = normalized_masks[-1]
    return normalized_masks


def post_processing_for_demo(cfg, mask):
    mask = normalize_masks([mask])
    mask = binary_mask(mask, cfg.demo.threshold)
    return mask