
from layers.blocks import conv_block, conv_down_block
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 mid_channel,
                 layers,
                 padding=1):
        super(Encoder, self).__init__()
        self.layers = self._build_layers(in_channel, out_channel, mid_channel, layers)

    def _build_layers(self, in_channel, out_channel, mid_channel, layers):
        modules = [conv_block(in_channel, out_channel)]
        for i in range(layers - 1):
            if i == 0:
                modules.append(conv_block(out_channel, mid_channel))
            else:
                modules.append(conv_down_block(mid_channel, mid_channel))
        print(modules)
        return nn.Sequential(*modules)

    def forward(self, x):
        x = self.layers(x)
        return x


class BasicEncoder(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 mid_channel,
                 dilation: list,
                 padding=1
                 ):
        super(BasicEncoder, self).__init__()
        self.layers = self._build_layers(in_channel, out_channel, mid_channel, dilation, padding)

    def _build_layers(self, in_channel, out_channel, mid_channel, dilation, padding):
        # modules = [conv_block(in_channel, out_channel)]
        modules = [conv_block(out_channel, mid_channel)]
        for idx, factor in enumerate(dilation):
            modules.append(
                conv_block(mid_channel,mid_channel,dilation=factor, padding=padding * factor,))

        # for i, factor in enumerate(dilation):
        #     if i == 0:
        #         _in_channel = out_channel
        #         _out_channel = mid_channel
        #     else:
        #         _in_channel = mid_channel
        #         _out_channel = mid_channel
        #     modules.append(conv_block(_in_channel, _out_channel, padding=padding * factor, dilation=factor))
        print(modules)
        return nn.Sequential(*modules)

    def forward(self, x):
        x = self.layers(x)
        return x




'''
class Encoder(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 mid_channel,
                 layers,
                 padding=1):
        super(Encoder, self).__init__()
        self.layers = self._build_layers(in_channel, out_channel, mid_channel, layers)

    def _build_layers(self, in_channel, out_channel, mid_channel, layers):
        modules = [conv_block(in_channel, out_channel)]
        for i in range(layers-1):
            if i == 0:
                modules.append(conv_block(out_channel, mid_channel))
            else:
                modules.append(conv_down_block(mid_channel, mid_channel, ))
        return nn.Sequential(*modules)

    def forward(self, x):
        x = self.layers(x)
        return x
        

class BasicEncoder(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 mid_channel,
                 dilation: list,
                 padding=1
                 ):
        super(BasicEncoder, self).__init__()
        self.layers = self._build_layers(in_channel, out_channel, mid_channel, dilation, padding)

    def _build_layers(self, in_channel, out_channel, mid_channel, dilation, padding):
        modules = []
        modules.append(conv_block(in_channel, out_channel))
        for i, factor in enumerate(dilation):
            if i == 0:
                _in_channel = out_channel
                _out_channel = mid_channel
            else:
                _in_channel = mid_channel
                _out_channel = mid_channel
            modules.append(conv_block(_in_channel, _out_channel, padding=padding * factor, dilation=factor))
        return nn.Sequential(*modules)

    def forward(self, x):
        x = self.layers(x)
        return x        
'''