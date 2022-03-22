

from dataclasses import dataclass
import torch

@dataclass
class DataInfo:
    image_path_lists: list
    image_nums: int

@dataclass
class ValInfo:
    loss: dict
    model_path: str
    model: torch.Tensor

@dataclass
class DataInfoSemantic:
    image_dicts: dict
    mask_dicts: dict
    names: list

@dataclass
class MetaInfoSemantic:
    image_path: str
    mask_path: str


@dataclass
class EncoderStageInfo:
    order: int
    height: int # output size of feature map
    width: int
    features: torch.Tensor


@dataclass
class MetricResults:
    precisions: list
    recalls: list
    f_scores: list


@dataclass
class CompareMetaInfoSemantic:
    image_path: str
    transys_mask_path: str
    hae_mask_path: str