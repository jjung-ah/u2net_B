# # import hydra
# # from omegaconf import DictConfig, OmegaConf
# #
# # @hydra.main(config_path="configs", config_name="config")
# # def main(cfg: DictConfig) -> None:
# #     print(f"OmegaConf.to_yaml(cfg): {OmegaConf.to_yaml(cfg)}") # logger 에서 최종 configs 파일 표시할때 사용
# #     print()
# #     # print(f"model name: {cfg.training.MODEL.BACKBONE.NAME}")
# #     # print(f'resume: {type(cfg.training.SOLVER.RESUME)}')
# #
# # if __name__ == '__main__':
# #     main()
#
#
# # from fvcore.common.registry import Registry
# # TEST_REGISTRY = Registry("TEST")
# #
# # @TEST_REGISTRY.register()
# # class Test:
# #     def __init__(self, cfg):
# #         self.cfg = cfg
# #
# # cfg = {'test': 'test'}
# # test = TEST_REGISTRY.get('Test')(cfg)
# # print()
#
# # from layers.rsu import RSU
# from layers.rsu4f import RSU4F
# #
# import torch
# # configs = {
# #     'in_channel': 3,
# #     'out_channel': 64,
# #     'mid_channel': 32,
# #     'depth': 7,
# #     'padding': 1
# # }
# #
# # layer = RSU(**configs)
# # inputs = torch.rand(1, 3,640,640)
# # outputs = layer(inputs)
#
# RSU4F
# configs = {
#     'in_channel': 512,
#     'out_channel': 512,
#     'mid_channel': 256,
#     'dilation': [1,2,4,8,4,2],
#     'padding': 1
# }
#
# layer = RSU4F(**configs)
# inputs = torch.rand(1, 512, 40, 40)
# outputs = layer(inputs)
# print()
#
#
# from layers.encoder import BasicEncoder
# from layers.decoder import BasicDecoder
#
# # BasicEncoder
# # configs = {
# #     'in_channel': 512,
# #     'out_channel': 512,
# #     'mid_channel': 256,
# #     'dilation': [1,2,4],
# #     'padding': 1
# # }
# #
# #
# # basic_encoder = BasicEncoder(**configs)
# # inputs = torch.rand(1, 512, 40, 40)
# # outputs = basic_encoder(inputs)
# # print()
#
#
# # BasicDecoder
# # configs = {
# #     'in_channel': 512,
# #     'out_channel': 512,
# #     'mid_channel': 256,
# #     'dilation': [4,2],
# #     'padding': 1
# # }
# #
# # basic_decoder = BasicDecoder(**configs)
# # inputs = torch.rand(1, 512, 40, 40)
# # outputs = basic_decoder(inputs)
# # print()
# #
# #
# # import torch
# # import torch.nn as nn
# # from layers.rsu import RSU
# # from layers.rsu4f import RSU4F
# # configs = {
# #     'in_channel': 3,
# #     'out_channel': 64,
# #     'mid_channel': 32,
# #     'depth': 7,
# #     'padding': 1
# # }
# #
# # is_cuda = torch.cuda.is_available()
# # device = torch.device("cuda")
# # model = RSU(**configs)
# # model = nn.DataParallel(model, device_ids=[0,1])
# # model.to(device)
# #
# # print()
# #
# # import os
# # import torch
# # from hydra import initialize_config_dir, compose
# # from modeling.meta_arch.u2net import U2Net, build_encoder_stages, build_decoder_stages, build_side_stages
# # from layers.blocks import interpolate, upsample, side_block
# # from collections import deque
# #
# # abs_config_dir = os.path.abspath("./configs")
# # with initialize_config_dir(config_dir=abs_config_dir):
# #     cfg = compose(config_name="config.yaml")
# #
# # encoder_stage = build_encoder_stages(cfg)
# # inputs = torch.rand(2, 3, 640, 640)
# #
# # encoder_features = deque()
# # for stage in encoder_stage:
# #     outputs = stage(inputs)
# #     encoder_features.append(outputs)
# #     inputs = interpolate(outputs, scale_factor=0.5)
# #     print()
# #
# # decoder_stage = build_decoder_stages(cfg)
# #
# # """
# #     0. En_6
# #     1. De_5 = up(En_6) + En_5
# #     2. De_4 = up(De_5) + En_4
# #     3. De_3 = up(De_4) + En_3
# #     4. De_2 = up(De_3) + En_2
# #     5. De_1 = up(De_2) + En_1
# #
# # """
# #
# # decoder_feature = [encoder_features.pop()]
# # for i, stage in enumerate(decoder_stage):
# #     inputs = torch.cat([
# #             interpolate(decoder_feature[-1], scale_factor=2),
# #             encoder_features.pop()
# #         ], dim=1
# #     )
# #
# #     outputs = stage(inputs)
# #     decoder_feature.append(outputs)
# #
# # side_stages = build_side_stages(cfg)
# # # side stages
# # masks = []
# # for feature, side in zip(decoder_feature, side_stages):
# #     outputs = side(feature)
# #     masks.append(upsample(outputs, size=640))
# #
# # masks = torch.cat(masks, dim=1)
# # fuse_layer = side_block(in_channel=6, out_channel=1, kernel_size=1)
# # outputs = fuse_layer(masks)
# # print()
#
#
# #
# # import torch
# # thlist = torch.linspace(0, 1-0.00001, 10)
# # print()
# #
# # pred_mask = [
# #     [0,1],
# #     [1,0],
# # ]
# #
# # gt = [
# #     [1,1],
# #     [0,0]
# # ]
# #
# # pred_mask = torch.tensor(pred_mask).float()
# # gt = torch.tensor(gt).float()
# #
# # mae = torch.abs(pred_mask - gt).mean()
# # height = 2
# # width = 2
# #
# # a = torch.sum(pred_mask - gt, dim=-1)
# # b = torch.sum(a, dim=-1)
# # mae2 = torch.sum(torch.sum(pred_mask - gt, dim=-1), dim=-1) / height * width
# # print()
# #
#
#
# # import torch
# # import numpy as np
# # gt_masks1 = np.array([
# #     [0,1,0],
# #     [1,1,1],
# #     [1,0,1]
# # ])
# #
# # gt_masks2 = np.array([
# #     [0,0,1],
# #     [0,0,1],
# #     [1,1,1]
# # ])
# #
# #
# # pred_masks1 = np.array([
# #     [0,1,0],
# #     [1,1,0],
# #     [0,1,0]
# # ])
# #
# # pred_masks2 = np.array([
# #     [1,0,1],
# #     [0,1,0],
# #     [1,1,1]
# # ])
# #
# #
# #
# # tp = np.sum(pred_masks * gt_masks)
# #
# # a = ~pred_masks + 2
# # fp = np.sum(a * gt_masks)
# # # fn = np.sum()
# #
# # print()
#
# # label: (batch, channel, shape, shape) => tensor => binary map
# '''
# label:
#     binary map
#         0: background
#         1: foreground
# '''
# # fuse_map: (batch, channel, shape, shape) => tensor => probability map
# '''
# fuse_map:
#     probability map
#
# '''
#
# import torch
# import numpy as np
#
# a = np.array(
#     [
#         [0,1],
#         [1,1]
#     ]
# )
#
# b = np.array(
#     [
#         [1,0],
#         [0,1]
#     ]
# )
#
# a = torch.tensor(a)
#
# # b = np.where((a==0)|(a==1), a^1, a)
# # c = ~a+2
# # print()
#
# #
# # tp = a * b
# #
# # r = torch.sum(tp).item()
# # print()
# #
# #
# # arrays = np.array(
# #     [
# #         [0.5, 0.2, 0.3],
# #         [0.1, 0.2, 0.5]
# #     ]
# # )
# # prop_map = torch.tensor(arrays)
# #
# # p_map = (prop_map > 0.2).float()
# # print()
#
# # import torch
# #
# # gt = torch.rand(2, 1, 5, 5)
# # pred = torch.rand(2,1,5,5)
# # threshold = 0.3
# # print()
# # gt = (gt > threshold).float()
# # bitmap = (pred > threshold).float()
# #
# # tp = gt * bitmap
# #
# # print()
#
# # from collections import defaultdict
# #
# # results = defaultdict(object)
# #
# # recall = [0.1, 0.2, 0.3, 0.4, 0.5]
# # precision = [0.2, 0.3, 0.4, 0.5, 0.6]
# # beta = 0.3
# #
# # results['recall'] = recall
# # results['precision'] = precision
# # results['beta'] = beta
# #
# # print()
# #
#
#
# # def cal(recalls, precisions):
# #     outputs = []
# #     for recall, precision in zip(recalls, precisions):
# #         outputs.append(
# #             ((1+beta)*precision*recall) / beta*precision+recall
# #         )
# #     return np.max(outputs)
# #
# # f_score = cal(recall, precision)
# # print()
#
# # from pathlib import Path
# # root_dir = '/mnt/hdd/results/test'
# #
# # if not Path(root_dir).is_dir():
# #     print('hello world!')
# # else:
# #     print('Fuck')
# #
# # print()
#
# #
# # from modeling.meta_arch.u2net import U2Net
# # import os
# # import torch
# # from hydra import initialize_config_dir, compose
# # from modeling.meta_arch.u2net import U2Net, build_encoder_stages, build_decoder_stages, build_side_stages
# # from layers.blocks import interpolate, upsample, side_block
# # from collections import deque
# #
# # abs_config_dir = os.path.abspath("./configs")
# # with initialize_config_dir(config_dir=abs_config_dir):
# #     cfg = compose(config_name="config.yaml")
# #
# # path = '/mnt/hdd/datasets/transys/datasets-images-transys-2021/05_results/u2net/1st/models/last_checkpoint.pth'
# # model = U2Net(cfg)
# #
# # weights = torch.load(path)
# #
# # state = model.state_dict()
# #
# # model.load_state_dict(weights.get('model_state_dict'))
# #
# # after_state = model.state_dict()
# # print()
#
#
# import torch
# import numpy as np
#
# inputs = torch.rand(2, 1,10,10)
# a = np.array([
#     [0.1,0.2,0.3],
#     [0.3,0.4,1],
#     [1,1,1]
# ])
#
# a_tensor = torch.tensor(a)
# a_tensor_int = a_tensor.int()
# print()
#
#
# b = np.array([
#     [255,255,255,255,255],
#     [255,255,255,255,255],
#     [255,255,255,255,255],
#     [255,255,255,255,255],
#     [255,255,255,255,255]
# ])
#
# b_tensor = torch.tensor(b)
# b_tensor_resize = b_tensor.resize_((3,3))
#
# print()

# import os
# from data.datasets.hyundai import Datasets
# from hydra import initialize_config_dir, compose
#
# abs_config_dir = os.path.abspath("./configs")
# with initialize_config_dir(config_dir=abs_config_dir):
#     cfg = compose(config_name="config.yaml")
#
# root_dir = '/mnt/hdd/datasets/transys/datasets-images-transys-2021/03_run_datasets/semantic/2rd'
# datasets = Datasets(cfg).get_raw_datasets()
# image_lists = datasets.get_test
#


#
# import numpy as np
#
#
# array1 = np.array([
#     [
#         [1,2,3],
#         [4,5,6],
#         [7,8,9]
#     ],
#
# ])
#
# seg_bin = np.array([
#     [
#         [0,0,0],
#         [1,1,1],
#         [2,2,2]
#     ]
# ])
#
#
# mask_image = np.zeros(array1.shape, dtype=np.float32)
# mask_image[seg_bin>0, 2] = 255
# print()

# import numpy as np
# import torch
#
# a = torch.tensor(np.array([[1,2], [3,4]]))
# b = torch.tensor(np.array([[1.0,2.0], [3.0,4.0]]))
#
# c = a*b
# print()


# import os
# import cv2
# from tqdm import tqdm
#
# root_dir = r'D:\datasets\project-hyundai-transys-caseleak\caseleak\03_run_datasets\1st\test\mask'
#
# mask_lists = [os.path.join(root_dir, x) for x in os.listdir(root_dir) if x.endswith('.png')]
#
# for path in tqdm(mask_lists):
#     image_name = os.path.basename(path)
#     mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
#     mask[mask > 0] = 255
#
#     cv2.imwrite(os.path.join(root_dir, image_name), mask)
#
#

#
# ####################### to mask contour from binary mask
# import cv2
# import os
#
# root_dir = r'D:\datasets\project-hyundai-transys-caseleak\caseleak\03_run_datasets\2nd\train'
# file_name = 'P09J1030915066054G120_L_01-Ch1,14_Rotation_6000%_1'
# # image_name = ''
# # mask_image_name = 'P09J1030915066054G120_L_01-Ch1,14_Rotation_6000%_1.png'
#
# original_image = cv2.imread(os.path.join(root_dir, 'image', f"{file_name}.jpg"))
# mask_image = cv2.imread(os.path.join(root_dir, 'mask', f"{file_name}.png"), cv2.COLOR_BGR2GRAY)
# res, thr = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)
# contours,_ = cv2.findContours(mask_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#
# for contour in contours:
#     cv2.drawContours(original_image, [contour], -1, (0, 0, 255), 10)
#
# cv2.imshow('contour', original_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print()


# import os
# from pathlib import Path
# image_path = r'D:\datasets\project-hyundai-transys-caseleak\caseleak\01_raw_datasets\caseleak_1st_model\test_ensemble\sample_test_ensemble_220107\ch1_14\image\P10L1150812045064G120_L_01-Ch1,14_Rotation_6000%_1.jpg'
#
#
# image_path1 = Path(image_path)
# print(f'parent: {os.path.abspath(image_path1.parent)}')
#
# image_path2 = os.path.abspath(os.path.join(image_path, os.pardir))
# image_path3 = os.path.dirname(os.path.dirname(image_path))
# # dir, file = os.path.split(image_path)
# # up_dir = os.path.abspath(dir)
# print()

###################################################################################

# 202203
import torch
from modeling.meta_arch.u2net import U2Net
import os
from hydra import initialize_config_dir, compose

abs_config_dir = os.path.abspath("./configs")
with initialize_config_dir(config_dir=abs_config_dir):
    cfg = compose(config_name="config.yaml")

print(cfg.training.model.meta_arch)
model = U2Net(cfg)
# model = U2NetPlus(cfg)

# inputs = torch.randn((1,3,640,640)).cuda()
inputs = torch.randn((1, 3, 640, 640))
outputs = model(inputs)
print(outputs.shape)




# # 20220311
# in_channel = 128  # will *2
# mid_channel = 16  # will *2
# out_channel = 64  # [64, 64, 128, 256, 512]  # TODO
# en_out_channel = 64
# rsu_mid_chanel = 32
# depth = 4
# layers = 7
# dilation = [1, 2, 4, 8, 4, 2]
#
# def make_channel_list(channel, depth):
#     return [channel*i for i in dilation[:depth]]
#
# de_in_ch = make_channel_list(in_channel, depth)
# de_mid_ch = make_channel_list(mid_channel, depth)
# de_out_ch = [out_channel] + make_channel_list(out_channel, depth)
# rsu_mid_ch = [rsu_mid_chanel] + [rsu_mid_chanel*i for i in dilation[:depth-1]]
# print('de_in_ch:', de_in_ch)
# print('de_mid_ch:', de_mid_ch)
# print('de_out_ch:', de_out_ch)
# print('rsu_mid_ch:', rsu_mid_ch)
#
# en_out_ch = make_channel_list(en_out_channel, depth)
# print('en_out_ch:', en_out_ch)
# print([16]*4)

'''
from layers.rsu import RSU
# from layers.rsu4f import RSU4F

import torch
configs = {
    'in_channel': 3,
    'out_channel': 64,
    'mid_channel': 32,
    'layers': 7,
    'padding': 1
}

layer = RSU(**configs)
inputs = torch.rand(1, 3,640,640)
outputs = layer(inputs)
print(outputs.shape)
'''


#
# RSU4F
# configs = {
#     'in_channel': 512,
#     'out_channel': 512,
#     'mid_channel': 256,
#     'dilation': [1,2,4,8,4,2],
#     'padding': 1
# }
#
# layer = RSU4F(**configs)
# inputs = torch.rand(1, 512, 40, 40)
# outputs = layer(inputs)
# print()
#
#

import torch
from layers.encoder import Encoder, BasicEncoder
from layers.decoder import Decoder, BasicDecoder

# # Encoder
# configs = {
#     'in_channel': 128,  # 512
#     'out_channel': 256,  # 512
#     'mid_channel': 64,  # 256
#     'layers': 7,   # Encoder
#     'padding': 1,
# #    'dilation': [1,2,4]   # BasicEncoder
# }
#
#
# encoder = Encoder(**configs)
# inputs = torch.rand(3, 128, 80, 80)  # Encoder
# outputs = encoder(inputs)
# # basic_encoder = BasicEncoder(**configs)
# # inputs = torch.rand(1, 512, 40, 40)  # BasicEncoder
# # outputs = basic_encoder(inputs)
# print(outputs.shape)



'''
# Decoder
configs = {
    'in_channel': 512,  # 512
    'out_channel': 128,  # 512
    'mid_channel': 64,  # 256
    'depth': 7,
    'padding': 1
}

decoder = Decoder(**configs)
inputs = torch.rand(3, 64, 80, 80)
outputs = decoder(inputs)
print(outputs.shape)
'''

# # BasicDecoder
# configs = {
#     'in_channel': 512,
#     'out_channel': 512,
#     'mid_channel': 256,
#     'dilation': [4,2],
#     'padding': 1
# }
#
# basic_decoder = BasicDecoder(**configs)
# inputs = torch.rand(1, 512, 40, 40)
# outputs = basic_decoder(inputs)
# print(outputs.shape)


# #
# #
# # import torch
# # import torch.nn as nn
# # from layers.rsu import RSU
# # from layers.rsu4f import RSU4F
# # configs = {
# #     'in_channel': 3,
# #     'out_channel': 64,
# #     'mid_channel': 32,
# #     'depth': 7,
# #     'padding': 1
# # }
# #
# # is_cuda = torch.cuda.is_available()
# # device = torch.device("cuda")
# # model = RSU(**configs)
# # model = nn.DataParallel(model, device_ids=[0,1])
# # model.to(device)
# #
# # print()

