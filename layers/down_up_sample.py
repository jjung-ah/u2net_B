import torch
import torch.nn as nn
import torch.nn.functional as F
from .build import LAYER_REGISTRY

@LAYER_REGISTRY.register()
def interpolate(input,
                scale_factor,
                mode='bilinear',
                align_corners=True,
                recompute_scale_factor=None):
    return F.interpolate(
        input,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners
    )

@LAYER_REGISTRY.register()
def max_pool():
    return nn.MaxPool2d(
        kernel_size=2,
        stride=None,
        padding=0,
        dilation=1,
        return_indices=False,
        ceil_mode=False
    )

@LAYER_REGISTRY.register()
def max_unpool():
    return nn.MaxUnpool2d(
        kernel_size=2,
        stride=None,
        padding=0
    )

@LAYER_REGISTRY.register()
def avg_pool():
    return nn.AvgPool2d(
        kernel_size= ,
        stride=None,
        padding=0,
        ceil_mode=False,
        count_include_pad=True,
        divisor_override=None
    )

@LAYER_REGISTRY.register()
def transpose():
    return nn.ConvTranspose2d(
        in_channels= ,
        out_channels= ,
        kernel_size= ,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode='zeros',
        device=None,
        dtype=None
    )