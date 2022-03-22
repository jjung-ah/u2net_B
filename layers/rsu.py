
import torch
import torch.nn as nn
from fvcore.common.registry import Registry
from layers.blocks import conv_block, interpolate
from layers.encoder import Encoder
from layers.decoder import Decoder
from layers.encoder import BasicEncoder
from layers.decoder import BasicDecoder
from collections import deque

STAGE_REGISTRY = Registry("STAGE")
STAGE_REGISTRY.__doc__ = " Registry for Blocks "

'''
@STAGE_REGISTRY.register()
class RSU(nn.Module):
    def __init__(self,
                 L,
                 in_channel,
                 out_channel,
                 mid_channel,
                 layers,
                 padding=1):
        super(RSU, self).__init__()
        self.init_layer = conv_block(in_channel, out_channel)
        self.encoder = Encoder(L, in_channel, out_channel, mid_channel, layers, padding)
        self.center = conv_block(mid_channel, mid_channel, padding=2, dilation=2)
        self.decoder = Decoder(L, in_channel, out_channel, mid_channel, layers, padding)

    # def __init__(
    #     self,
    #     L: int,
    #     input_channels: int,
    #     middle_channels: int,
    #     output_channels: int,
    #     kernel_size=3,
    #     dilation=1,
    #     padding=1
    # ):
    #     super().__init__()
    #
    #     self.init_layer = conv_block(in_channels=input_channels, out_channels=output_channels)
    #     self.dilation_layer = conv_block(in_channels=middle_channels, out_channels=middle_channels, dilation=2, padding=2)
    #     self.encoders = Encoder(L=L, input_channels=output_channels, output_channels=middle_channels)
    #     self.decoders = Decoder(L=L, input_channels=middle_channels*2, output_channels=output_channels)

    def forward(self, x):
        # 1. an input convolution layer
        init = self.init_layer(x)

        # 2. a U-Net like symmetric encoder-decoder structure with height of L
        encoder_features = []
        # out = copy.deepcopy(init.data)
        out = init.data
        for idx in range(len(self.encoders.x)):
            out = self.encoders.block[idx](out)
            encoder_features.insert(0, out)

        out = self.dilation_layer(out)

        for idx in range(len(self.decoders.x)):
            out = torch.cat((encoder_features[idx], out), dim=1)
            out = self.decoders.x[idx](out)

        # 3. a residual connection
        return init + out
'''

@STAGE_REGISTRY.register()
class RSU(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 mid_channel,
                 layers,
                 dilation,
                 padding=1):
        super(RSU, self).__init__()
        self.encoder = Encoder(in_channel, out_channel, mid_channel, layers, padding)
        self.center = conv_block(mid_channel, mid_channel, padding=2, dilation=2)
        self.decoder = Decoder(in_channel, out_channel, mid_channel, layers, padding)
        # self.decoder = Decoder(out_channel, mid_channel, mid_channel, layers, padding)

    def forward(self, x):
        features = deque()
        inputs = x

        for i, layer in enumerate(self.encoder.layers):
            # out = self.encoders.block[idx](out)
            outputs = layer(inputs)
            features.append(outputs)
            inputs = outputs

        center = self.center(inputs)
        # center = self.center(int(inputs/2))
        inputs = torch.cat([inputs, center], dim=1)
        features.pop()

        for i, layer in enumerate(self.decoder.layers):
            if i > 0:
                inputs = interpolate(inputs, scale_factor=2.0)
                inputs = torch.cat([inputs, features.pop()], dim=1)
            inputs = layer(inputs)

        print('features', features[-1].shape)
        print('input', inputs.shape)
        outputs = features[-1] + inputs  # this inputs is final results of decoder stage
        print('outputs', outputs.shape)
        return outputs




    # def forward(self, x):
    #     features = deque()
    #     inputs = x
    #
    #     for i, layer in enumerate(self.encoder.layers):
    #         outputs = layer(inputs)
    #         features.append(outputs)
    #         inputs = outputs
    #
    #     center = self.center(inputs)
    #     inputs = torch.cat([inputs, center], dim=1)
    #     features.pop()
    #
    #     for i, layer in enumerate(self.decoder.layers):
    #         if i > 0:
    #             inputs = interpolate(inputs, scale_factor=2.0)
    #             inputs = torch.cat([inputs, features.pop()], dim=1)
    #         inputs = layer(inputs)
    #
    #     print('features', features[-1].shape)
    #     print('input', inputs.shape)
    #     outputs = features[-1] + inputs  # this inputs is final results of decoder stage
    #     print('outputs', outputs.shape)
    #     return outputs
    #

@STAGE_REGISTRY.register()
class RSU4F(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 mid_channel,
                 layers,
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