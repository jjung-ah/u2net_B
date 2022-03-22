
from layers.blocks import conv_block
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 mid_channel,
                 depth,
                 padding=1):
        super(Decoder, self).__init__()
        self.layers = self._build_layers(mid_channel, out_channel, depth)

    def _build_layers(self, mid_channel, out_channel, depth):
        """
            :param in_channel: == mid_channel
            :param out_channel:
            :param depth:
        :return: reversed modules
        """
        modules = []
        for i in range(depth-1):
            if i == 0:
                modules.append(conv_block(2*mid_channel, out_channel))
            else:
                modules.append(conv_block(2*mid_channel, mid_channel))
        print(modules)
        return nn.Sequential(*reversed(modules))

    def forward(self, x):
        x = self.layers(x)
        return x

class BasicDecoder(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 mid_channel,
                 dilation,
                 padding=1):
        super(BasicDecoder, self).__init__()
        self.layers = self._build_layers(in_channel, out_channel, mid_channel, dilation, padding)

    def _build_layers(self, in_channel, out_channel, mid_channel, dilation, padding):
        modules = []
        for i, factor in enumerate(dilation):
            _in_channel = 2 * mid_channel
            _out_channel = mid_channel
            modules.append(conv_block(_in_channel, _out_channel, padding=padding * factor, dilation=factor))
        modules.append(conv_block(_in_channel, _in_channel))  # u2net
        # modules.append(conv_block(_in_channel, 2 * _in_channel))  # 20220225 by lee


        return nn.Sequential(*modules)

    def forward(self, x):
        x = self.layers(x)
        return x
