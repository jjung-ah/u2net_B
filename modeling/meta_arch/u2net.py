
import torch
import torch.nn as nn
from modeling.meta_arch.build import META_ARCH_REGISTRY
from layers.weight_init import weight_init_xavier
from layers.rsu import RSU, STAGE_REGISTRY
# from layers.blocks import Block
from layers.blocks import interpolate, side_block, upsample, conv_block
from layers.rsu4f import RSU4F
from collections import deque



import os
from hydra import initialize_config_dir, compose

abs_config_dir = os.path.abspath("./configs")
with initialize_config_dir(config_dir=abs_config_dir):
    cfg = compose(config_name="config.yaml")

'''
def build_stages(cfg):
    arch_stage = cfg.architecture.stages
    layer = cfg.architecture.params.layers
    rsu_depth = cfg.architecture.params.rsu_depth
    rsu4f_depth = cfg.architecture.params.rsu4f_depth
    dilation = cfg.architecture.params.dilation
    encoder_stages, decoder_stages, side_stages = [], [], []
    for stage_info, stage_params in arch_stage.items():
        stage_name = stage_info.split('-')[0]
        stage_part = stage_info.split('-')[-1][0]
'''

''''# origin
def build_encoder_stages(cfg):
    modules = []
    in_channel = 3
    out_channel = cfg.meta_arch.architecture.encoder_stages.out_channel
    layers = cfg.meta_arch.architecture.rsu.layers

    for stage in range(cfg.meta_arch.architecture.rsu.depth):
        mid_channel = cfg.meta_arch.architecture.rsu.mid_channel[stage]
        modules.append(RSU(
            in_channel=in_channel,
            out_channel=out_channel,
            mid_channel=mid_channel,
            layers=layers,
            )
        )
        in_channel = out_channel
        out_channel *= 2
        layers -= 1

        # ##### 20220222 by baek
        # in_channel = out_channel
        # out_channel *= 1
        # layers -= 1

    for stage in range(cfg.meta_arch.architecture.rsu4f.depth):
        modules.append(RSU4F(
            in_channel=in_channel,
            out_channel=in_channel,
            mid_channel=cfg.meta_arch.architecture.rsu4f.mid_channel,
            dilation=cfg.meta_arch.architecture.rsu4f.dilation
        ))
    return nn.Sequential(*modules)

def build_decoder_stages(cfg):
    modules = []

    in_channel = cfg.meta_arch.architecture.decoder_stages.in_channel
    mid_channel = cfg.meta_arch.architecture.decoder_stages.mid_channel

    layers = cfg.meta_arch.architecture.rsu.layers

    for stage in range(cfg.meta_arch.architecture.rsu.depth):
        out_channel = cfg.meta_arch.architecture.decoder_stages.out_channel[stage]
        modules.append(RSU(
            in_channel=in_channel,
            out_channel=out_channel,
            mid_channel=mid_channel,
            layers=layers,
            )
        )

        in_channel *= 2
        mid_channel *= 2
        layers -= 1

        # ##### 20220222 by baek
        # in_channel *= 1
        # mid_channel *= 1
        # layers -= 1

    in_channel = int(in_channel / 2)
    out_channel *= 2
    # in_channel = in_channel  # 20220222 by baek
    # out_channel *= 1  # 20220222 by baek

    modules.append(
        RSU4F(
            in_channel=in_channel,
            out_channel=out_channel,
            mid_channel=mid_channel,
            dilation=cfg.meta_arch.architecture.rsu4f.dilation
        )
    )

    return nn.Sequential(*reversed(modules))
    
def build_side_stages(cfg):
    # TODO related to architecture.decoder_stages.out_channel
    moduels = []
    for in_channel in cfg.meta_arch.architecture.decoder_stages.out_channel:
        moduels.append(
            side_block(
                in_channel,
                out_channel=1,
                kernel_size=3
            )
        )
    # TODO: for encoder_6
    moduels.append(side_block(cfg.meta_arch.architecture.decoder_stages.out_channel[-1], 1, 3))
    # moduels.append(side_block(512, 1, 3))  # u2net
    # moduels.append(side_block(64, 1, 3))  # 20220222 by baek
    return nn.Sequential(*reversed(moduels))
'''

'''
import os
from hydra import initialize_config_dir, compose

abs_config_dir = os.path.abspath("./configs")
with initialize_config_dir(config_dir=abs_config_dir):
    cfg = compose(config_name="config.yaml")


def make_channel_list(cfg, channel, depth):
    dilation = cfg.meta_arch.architecture.rsu4f.dilation
    return [channel*i for i in dilation[:depth]]


def build_channel_list(cfg):
    encoder_out_ch = cfg.meta_arch.architecture.encoder_stages.out_channel
    rsu_mid_ch = cfg.meta_arch.architecture.rsu.mid_channel
    decoder_in_ch = cfg.meta_arch.architecture.decoder_stages.in_channel
    decoder_out_ch = cfg.meta_arch.architecture.decoder_stages.out_channel
    decoder_mid_ch = cfg.meta_arch.architecture.decoder_stages.mid_channel
    depth = cfg.meta_arch.architecture.rsu.depth
    dilation = cfg.meta_arch.architecture.rsu4f.dilation
    if cfg.training.model.meta_arch == 'U2Net':
        encoder_out_channel = make_channel_list(cfg, encoder_out_ch, depth)
        mid_channel_list = [rsu_mid_ch] + [rsu_mid_ch*i for i in dilation[:depth-1]]
        decoder_in_channel = make_channel_list(cfg, decoder_in_ch, depth)
        decoder_mid_channel = make_channel_list(cfg, decoder_mid_ch, depth)
        decoder_out_channel = [decoder_out_ch] + make_channel_list(cfg, decoder_out_ch, depth)
    elif cfg.training.model.meta_arch == 'U2NetPlus':
        encoder_out_channel = [encoder_out_ch] * depth
        mid_channel_list = [rsu_mid_ch] * depth
        decoder_in_channel = [decoder_in_ch] * depth
        decoder_mid_channel = [decoder_mid_ch] * depth
        decoder_out_channel = [decoder_out_ch] * (depth+1)

    return encoder_out_channel, mid_channel_list, decoder_in_channel, decoder_mid_channel, decoder_out_channel

encoder_out_channel, mid_channel_list, decoder_in_channel, decoder_mid_channel, decoder_out_channel = build_channel_list(cfg)

def build_encoder_stages(cfg):
    modules = []
    in_channel = cfg.meta_arch.architecture.encoder_stages.in_channel
    # out_channel = cfg.meta_arch.architecture.encoder_stages.out_channel
    layers = cfg.meta_arch.architecture.rsu.layers
    # encoder_out_channel, mid_channel_list, decoder_in_channel, decoder_mid_channel, decoder_out_channel = build_channel_list(cfg)

    for stage in range(cfg.meta_arch.architecture.rsu.depth):
        mid_channel = mid_channel_list[stage]
        out_channel = encoder_out_channel[stage]
        modules.append(RSU(
            in_channel=in_channel,
            out_channel=out_channel,
            mid_channel=mid_channel,
            layers=layers,
            )
        )
        in_channel = out_channel
        # out_channel *= 2
        layers -= 1

        # ##### 20220222 by baek
        # in_channel = out_channel
        # out_channel *= 1
        # layers -= 1

    for stage in range(cfg.meta_arch.architecture.rsu4f.depth):
        modules.append(RSU4F(
            in_channel=in_channel,
            out_channel=in_channel,
            mid_channel=cfg.meta_arch.architecture.rsu4f.mid_channel,
            dilation=cfg.meta_arch.architecture.rsu4f.dilation
        ))
    return nn.Sequential(*modules)


def build_decoder_stages(cfg):
    modules = []
    # encoder_out_channel, mid_channel_list, decoder_in_channel, decoder_mid_channel, decoder_out_channel = build_channel_list(cfg)
    layers = cfg.meta_arch.architecture.rsu.layers
    in_channel = cfg.meta_arch.architecture.decoder_stages.in_channel
    mid_channel = cfg.meta_arch.architecture.decoder_stages.mid_channel

    for stage in range(cfg.meta_arch.architecture.rsu.depth):
        # in_channel = decoder_in_channel[stage]
        out_channel = decoder_out_channel[stage]
        # mid_channel = decoder_mid_channel[stage]
        # out_channel = cfg.meta_arch.architecture.decoder_stages.out_channel[stage]
        modules.append(RSU(
            in_channel=in_channel,
            out_channel=out_channel,
            mid_channel=mid_channel,
            layers=layers,
            )
        )

        # in_channel *= 2
        # mid_channel *= 2
        layers -= 1

        # ##### 20220222 by baek
        # in_channel *= 1
        # mid_channel *= 1
        # layers -= 1

    if cfg.training.model.meta_arch == 'U2Net':
        in_channel = int(in_channel / 2)
        # in_channel = in_channel
        out_channel *= 2  # 1024
        # out_channel *= 1
    elif cfg.training.model.meta_arch == 'U2NetPlus':
        in_channel = in_channel
        out_channel *= 1

    # in_channel = int(in_channel / 2)
    # out_channel *= 2
    # in_channel = in_channel  # 20220222 by baek
    # out_channel *= 1  # 20220222 by baek

    modules.append(
        RSU4F(
            in_channel=in_channel,
            out_channel=out_channel,
            mid_channel=mid_channel,
            dilation=cfg.meta_arch.architecture.rsu4f.dilation
        )
    )

    return nn.Sequential(*reversed(modules))


def build_side_stages(cfg):
    # TODO related to architecture.decoder_stages.out_channel
    moduels = []
    # encoder_out_channel, mid_channel_list, decoder_in_channel, decoder_mid_channel, decoder_out_channel = build_channel_list(cfg)
    # for in_channel in cfg.meta_arch.architecture.decoder_stages.out_channel:
    for in_channel in decoder_out_channel:
        moduels.append(
            side_block(
                in_channel,
                out_channel=1,
                kernel_size=3
            )
        )
    # TODO: for encoder_6
    moduels.append(side_block(decoder_out_channel[-1], 1, 3))
    # moduels.append(side_block(512, 1, 3))  # u2net
    # moduels.append(side_block(64, 1, 3))  # 20220222 by baek
    return nn.Sequential(*reversed(moduels))



@META_ARCH_REGISTRY.register()
class U2Net(nn.Module):
    def __init__(self, cfg):
        super(U2Net, self).__init__()
        self.encoder_stages = build_encoder_stages(cfg)
        self.decoder_stages = build_decoder_stages(cfg)
        self.side_stages = build_side_stages(cfg)
        self.side_fuse = side_block(in_channel=6, out_channel=1, kernel_size=1)

        self.cfg = cfg
        self._initialize()

    def _initialize(self):
        self.encoder_stages.apply(weight_init_xavier)
        self.decoder_stages.apply(weight_init_xavier)

    def forward(self, x):
        encoder_features = deque()
        inputs = x

        for stage in self.encoder_stages:
            outputs = stage(inputs)
            encoder_features.append(outputs)
            # inputs = interpolate(outputs, scale_factor=0.5)
            inputs = interpolate(outputs, scale_factor=0.5)

        decoder_feature = [encoder_features.pop()]
        for i, stage in enumerate(self.decoder_stages):
            inputs = torch.cat([
                    interpolate(decoder_feature[-1], scale_factor=2.0),
                    encoder_features.pop()
                ], dim=1
            )

            outputs = stage(inputs)
            decoder_feature.append(outputs)

        # side stages
        masks = []
        for feature, side in zip(decoder_feature, self.side_stages):
            outputs = side(feature)
            masks.append(upsample(outputs, size=self.cfg.training.transforms.resize.shape))

        fuse_masks = torch.cat(masks, dim=1)
        outputs = self.side_fuse(fuse_masks)
        return masks, outputs


@META_ARCH_REGISTRY.register()
class U2NetPlus(nn.Module):
    def __init__(self, cfg):
        super(U2NetPlus, self).__init__()
        self.encoder_stages = build_encoder_stages(cfg)
        self.decoder_stages = build_decoder_stages(cfg)
        self.side_stages = build_side_stages(cfg)
        self.side_fuse = side_block(in_channel=6, out_channel=1, kernel_size=1)

        self.cfg = cfg
        self._initialize()

    def _initialize(self):
        self.encoder_stages.apply(weight_init_xavier)
        self.decoder_stages.apply(weight_init_xavier)

    def forward(self, x):
        encoder_features = deque()
        inputs = x

        for stage in self.encoder_stages:
            outputs = stage(inputs)
            encoder_features.append(outputs)
            inputs = interpolate(outputs, scale_factor=0.5)

        decoder_feature = [encoder_features.pop()]
        for i, stage in enumerate(self.decoder_stages):
            inputs = torch.cat([
                    interpolate(decoder_feature[-1], scale_factor=2.0),
                    encoder_features.pop()
                ], dim=1
            )

            outputs = stage(inputs)
            decoder_feature.append(outputs)

        # side stages
        masks = []
        for feature, side in zip(decoder_feature, self.side_stages):
            outputs = side(feature)
            masks.append(upsample(outputs, size=self.cfg.training.transforms.resize.shape))

        fuse_masks = torch.cat(masks, dim=1)
        outputs = self.side_fuse(fuse_masks)
        return masks, outputs
'''

@META_ARCH_REGISTRY.register()
class U2Net(nn.Module):
    def __init__(self, cfg):
        super(U2Net, self).__init__()
        # self.encoder_stages = build_encoder_stages(cfg)
        # self.decoder_stages = build_decoder_stages(cfg)
        # self.side_stages = build_side_stages(cfg)
        # self.side_fuse = side_block(in_channel=6, out_channel=1, kernel_size=1)

        rsu_dict = cfg.meta_arch.architecture.stages
        # img_size = cfg.data.dataloader.transforms.Resize.size

        encoder_blocks, decoder_blocks, side_blocks = self.build_stages(rsu_dict)
        self.encoder_stages = nn.Sequential(*encoder_blocks)
        self.decoder_stages = nn.Sequential(*decoder_blocks)
        self.side_stages = nn.Sequential(*side_blocks)
        self.side_fuse = side_block(in_channel=6, out_channel=1, kernel_size=1)

        self.cfg = cfg
        self._initialize()

    def _initialize(self):
        self.encoder_stages.apply(weight_init_xavier)
        self.decoder_stages.apply(weight_init_xavier)


    def build_stages(self, rsu_dict: dict):
        # Build encoder-blocks and decoder-blocks.
        dilation = cfg.meta_arch.architecture.params.dilation
        encoder_stages, decoder_stages, side_stages = [], [], []
        for stage_info, parameters in rsu_dict.items():
            stage_name = stage_info.split('-')[0].upper()
            stage_part = stage_info.split('-')[-1][:2]
            layers, in_channel, mid_channel, out_channel = parameters

            # build rsu or rsu-4f.
            block = STAGE_REGISTRY.get(stage_name)(in_channel, out_channel, mid_channel, layers, dilation)
            if stage_part == 'en':
                encoder_stages.append(block)
            else:
                decoder_stages.append(block)
                # side_stages.append(side_block(in_channel, 1, 3))
                side_stages.append(side_block(out_channel, 1, 3))  # 20220322
        side_stages.insert(0, side_stages[0])
        # side_stages.append(conv_block(len(side_stages), 1, kernel_size=1, padding=0,))

        return encoder_stages, decoder_stages, side_stages

    def forward(self, x):
        encoder_features = deque()
        inputs = x

        for stage in self.encoder_stages:
            outputs = stage(inputs)
            encoder_features.append(outputs)
            inputs = interpolate(outputs, scale_factor=0.5)

        decoder_feature = [encoder_features.pop()]
        for i, stage in enumerate(self.decoder_stages):
            inputs = torch.cat([
                interpolate(decoder_feature[-1], scale_factor=2.0),
                encoder_features.pop()
            ], dim=1
            )

            outputs = stage(inputs)
            decoder_feature.append(outputs)

        # side stages
        masks = []
        for feature, side in zip(decoder_feature, self.side_stages):
            outputs = side(feature)
            masks.append(upsample(outputs, size=self.cfg.training.transforms.resize.shape))

        fuse_masks = torch.cat(masks, dim=1)
        outputs = self.side_fuse(fuse_masks)
        return masks, outputs


    # def forward(self, x):
    #     encoder_features = []
    #     sups = []
    #     for idx in range(len(self.encoder_blocks)-1):
    #         x = self.encoder_blocks[idx](x)
    #         batch, channel, height, width = x.shape
    #         x = interpolate(x, size=(int(height/2), int(width/2)), mode='bilinear', align_corners=True)
    #         encoder_features.insert(0, x)
    #     x = self.encoder_blocks[-1](x)
    #     sups.append(x)
    #
    #     for idx in range(len(self.decoder_blocks)):
    #         x = torch.cat((encoder_features[idx], x), dim=1)
    #         batch, channel, height, width = x.shape
    #         x = F.interpolate(x, size=(height*2, width*2), mode='bilinear', align_corners=True)
    #         x = self.decoder_blocks[idx](x)
    #         sups.append(x)
    #
    #     # calculate Sups.
    #     masks = []
    #     for idx, sup in enumerate(sups):
    #         mask = self.side_blocks[idx](sup)
    #         masks.append(mask)
    #     fuse_mask = self.side_blocks[-1](torch.cat((masks), dim=1))
    #
    #     return masks, fuse_mask