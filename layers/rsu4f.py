
import torch
import torch.nn as nn
from fvcore.common.registry import Registry
from layers.blocks import conv_block
from layers.encoder import BasicEncoder
from layers.decoder import BasicDecoder
from collections import deque

STAGE_REGISTRY = Registry("STAGE")
STAGE_REGISTRY.__doc__ = " Registry for Blocks "


@STAGE_REGISTRY.register()
class RSU4F(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 mid_channel,
                 dilation,
                 padding=1):
        super(RSU4F, self).__init__()

        # TODO .....
        center = len(dilation) // 2
        self.encoder = BasicEncoder(in_channel, out_channel, mid_channel, dilation[:center])
        self.center = conv_block(mid_channel, mid_channel, padding=padding * dilation[center], dilation=dilation[center])
        self.decoder = BasicDecoder(in_channel, out_channel, mid_channel, dilation[center+1:])


    def forward(self, x):
        features = deque()
        inputs = x
        for i, layer in enumerate(self.encoder.layers):
            outputs = layer(inputs)
            features.append(outputs)
            inputs = outputs

        center = self.center(inputs)
        inputs = torch.cat([inputs, center], dim=1)
        for i, layer in enumerate(self.decoder.layers):
            outputs = layer(inputs)
            inputs = torch.cat([outputs, features.pop()], dim=1)

        outputs = features[-1] + outputs
        return outputs