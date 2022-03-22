
import torch.nn as nn
import torch.nn.functional as F
from solver.build import SOLVER_REGISTRY
from .build import LAYER_REGISTRY
from solver.build import build_activate
from .build import build_downsample, build_upsample

import os
from hydra import initialize_config_dir, compose

abs_config_dir = os.path.abspath("./configs")
with initialize_config_dir(config_dir=abs_config_dir):
    cfg = compose(config_name="config.yaml")
activation = build_activate(cfg)
# down = build_downsample(cfg)
# up = build_upsample(cfg)

def conv_block(in_channel: int, out_channel: int, padding=1, stride=1, kernel_size=3, dilation=1):
    modules = []
    modules.append(nn.Conv2d(
        in_channels=in_channel,
        out_channels=out_channel,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation
    ))
    modules.append(nn.BatchNorm2d(out_channel))
    # modules.append(nn.LeakyReLU())
    modules.append(activation)
    return nn.Sequential(*modules)


def conv_down_block(in_channel: int, out_channel: int, padding=1, stride=1, kernel_size=3, dilation=1, pooling_size=2):
    modules = []
    modules.append(nn.MaxPool2d(pooling_size))
    modules.append(nn.Conv2d(
        in_channels=in_channel,
        out_channels=out_channel,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation
    ))
    modules.append(nn.BatchNorm2d(out_channel))
    # modules.append(nn.LeakyReLU())
    modules.append(activation)
    return nn.Sequential(*modules)


def conv_up_block(in_channel: int, out_channel: int, padding=1, stride=1, kernel_size=3, dilation=1, pooling_size=2):
    modules = []
    modules.append(nn.MaxUnPool2d(pooling_size))
    modules.append(nn.Conv2d(
        in_channels=in_channel,
        out_channels=out_channel,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation
    ))
    modules.append(nn.BatchNorm2d(out_channel))
    # modules.append(nn.LeakyReLU())
    modules.append(activation)
    return nn.Sequential(*modules)


def interpolate(input, scale_factor, mode='bilinear', align_corners=True, recompute_scale_factor=None):
    return F.interpolate(
        input,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners
    )

# def maxpool(
#             input,
#             pooling_size,
#             ):
#     conv = nn.MaxPool2d(pooling_size)
#     return conv(input)
#
# def avgpool(
#             input,
#             pooling_size,
#             ):
#     conv = nn.AvgPool2d(pooling_size)
#     return conv(input)

# it is similar as interpolate function
def upsample(input, size, mode='bilinear', align_corners=True):
    return F.interpolate(
        input,
        size=(size, size),
        mode=mode,
        align_corners=align_corners
    )


def side_block(in_channel: int, out_channel: int, kernel_size: int, stride: int = 1):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel,
                  out_channels=out_channel,
                  kernel_size=kernel_size,
                  stride=stride),
        nn.Sigmoid()
    )

