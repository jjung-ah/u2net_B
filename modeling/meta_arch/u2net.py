
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


@META_ARCH_REGISTRY.register()
class U2Net(nn.Module):
    def __init__(self, cfg):
        super(U2Net, self).__init__()

        rsu_dict = cfg.meta_arch.architecture.stages
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


@META_ARCH_REGISTRY.register()
class U2NetPlus(U2Net):
    def __init__(self, cfg):
        super(U2NetPlus, self).__init__(cfg)

'''
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
