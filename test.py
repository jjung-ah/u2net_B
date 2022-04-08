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
print()




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




import os
from glob import glob
from pathlib import Path
import random
from collections import defaultdict

root_path = 'D:\\datasets\\project-hyundai-transys-caseleak-datasets-v2\\01_raw\\'
sub_folders = ['Right', 'Rotation', 'Top', 'Top1', 'Front', 'Front_6']
num = 100
ratio = 0.7

IMAGE_EXTENSION = ['.jpg', '.jpeg', '.png']

def is_image(paths: list):
    images = []
    for item in paths:
        name, ext = os.path.splitext(item)
        if ext in IMAGE_EXTENSION:
            images.append(item)
    return images


# img_list = []
# f = open('D:\\datasets\\project-hyundai-transys-caseleak-datasets-v2\\01_raw\\test.txt', 'a')
# for (path, dirs, files) in os.walk(root_path):
#     if set(dirs) - set(sub_folders) == set() and dirs != []:
#         for d in dirs:
#             dir_path = path + '\\' + d
#             print(dir_path)
#             image_list = glob(dir_path + '\\*')
#             # print(len(image_list[:num]))
#             # print(image_list[:num])
#             # print(len(image_list[num:]))
#             # print(image_list[num:])
#             img_list = is_image(image_list)
#             # for i in image_list:
#             #     name, ext = os.path.splitext(i)
#             #     print(name, ext)
#             #     if ext in IMAGE_EXTENSION:
#             #         img_list.append(i)
#             print(img_list)
#             for i in image_list:
#                 data = i + '\n'
#                 f.write(data)


# l = [1, 2, 3]
# print(l[0], l[1:])


def make_image_list(root_path: str):
    for (path, dirs, files) in os.walk(root_path):
        if set(dirs) - set(sub_folders) == set() and dirs != []:
            for d in dirs:
                dir_path = path + f'{os.sep}' + d
                # print(dir_path)
                image_list = glob(dir_path + f'{os.sep}' + '*.jpg')
                # image_path_list = glob(dir_path + f'{os.sep}' + '.*')
                # img_file_list = is_image(image_path_list)
    return image_list

def split_test_list(image_list: list, num: int):
    train_val_list = image_list[:num]
    test_list = image_list[num:]
    return train_val_list, test_list

def split_train_val(train_val_list: list, ratio: float, shuffle: False):
    if shuffle == True:
        random.shuffle(train_val_list)
    length = len(train_val_list)
    split_num = int(length * ratio)
    train_set = train_val_list[:split_num]
    val_set = train_val_list[split_num:]
    return train_set, val_set

def split_shuffle_test_list(root_path: str, num: int):
    train_val_dict = defaultdict()
    test_dict = defaultdict()
    test_list = []
    for (path, dirs, files) in os.walk(root_path):
        if set(dirs) - set(sub_folders) == set() and dirs != []:
            dir_path = path + f'{os.sep}' + dirs[0]
            # print(dir_path)
            # image_path_list = glob(dir_path + f'{os.sep}' + '.*')
            # img_file_list = is_image(image_path_list)
            img_file_list = glob(dir_path + f'{os.sep}' + '*.jpg')
            # print(img_file_list)
            train_val_list = random.sample(img_file_list, num)  # 이걸로 해야하나.. 아니면 shuffle 후에 100개를 잘라야하나..?
            # test_list = list(set(image_path_list) - set(train_val_list))
            test_list += list(set(img_file_list) - set(train_val_list))
            for i in train_val_list:
                file_name = os.path.basename(i)
                train_val_dict[file_name] = [path, dirs]
            for j in test_list:
                file_name = os.path.basename(j)
                test_dict[file_name] = [path, dirs]
    file_key = list(test_dict.keys())
    test_datasets = make_dict_path(file_key, test_dict)
    return train_val_dict, test_datasets

def make_dataset_dict(dataset_list: list):
    pass

def split_train_val_dict(train_val_dict: dict, ratio: float, shuffle: False):
    file_key = list(train_val_dict.keys())
    if shuffle == True:
        random.shuffle(file_key)
    length = len(train_val_dict)
    split_num = int(length * ratio)
    train_set = file_key[:split_num]
    val_set = file_key[split_num:]
    train_datasets = make_dict_path(train_set, train_val_dict)
    val_datasets = make_dict_path(val_set, train_val_dict)
    return train_datasets, val_datasets

def make_dict_path(key_name_set: list, path_dict: dict):
    path_list = []
    for i in key_name_set:
        for j in path_dict[i][1]:
            f_name = i.split('_')
            file_name = i.replace(f_name[-2], j)
            value = path_dict[i][0] + f'{os.sep}' + j + f'{os.sep}' + file_name
            path_list.append(value)
    return path_list



def is_label():
    pass

def save_text(image_list: list, out_dir: str):
    if is_label == False:
        f = open(out_dir, 'a', encoding='utf-8')
        for i in image_list:
            data = i + '\n'
            f.write(data)
    else:
        f = open(out_dir, 'a', encoding='utf-8')
        for i in image_list:
            data = i + '\n'
            f.write(data)
    f.close()


def check_text_file():
    pass


# @hydra.main(config_path='configs', config_name='config.yaml')
# def main(cfg):
#     # cfg = setup(cfg)
#     if cfg.datasets.parsing:
#         parser = Parser(cfg)
#
#
# if __name__ == '__main__':
#     main()


# train_val_datasets, test_datasets = split_shuffle_test_list(root_path, 100)
# # # print(train_val_datasets)
# # # print(len(train_val_datasets))
# train_datasets, val_datasets = split_train_val_dict(train_val_datasets, 0.7, False)
# # # print(train_datasets)
# # save_text(test_datasets, 'D:\\datasets\\project-hyundai-transys-caseleak-datasets-v2\\01_raw\\test.txt')
# # save_text(train_datasets, 'D:\\datasets\\project-hyundai-transys-caseleak-datasets-v2\\01_raw\\train.txt')
# # save_text(val_datasets, 'D:\\datasets\\project-hyundai-transys-caseleak-datasets-v2\\01_raw\\val.txt')


# dict = {'1': ['one', ['10', '11']], '2': ['two', ['20', '22']], '3': ['three', ['30', '33']]}
# for i in list(dict.keys())[:2]:
#     print(dict[i])
#     print(dict[i][1])


def check_test_file(self):
    text_list = []
    f = open('D:\\datasets\\project-hyundai-transys-caseleak-datasets-v2\\01_raw\\test.txt', 'r', encoding='utf-8')
    text = f.readlines()
    train_intersection = set(train_datasets).intersection(set(text))
    val_intersection = set(val_datasets).intersection(set(text))
    pass

# f = open('D:\\datasets\\project-hyundai-transys-caseleak-datasets-v2\\01_raw\\test.txt', 'r', encoding='utf-8')  # encoding='cp949'
# text = f.readlines()
# text_list = [i.split('\n')[0] for i in text]
# print(text_list)
# train_intersection = set(train_datasets).intersection(set(text))
# val_intersection = set(val_datasets).intersection(set(text))
# # print(val_datasets)
# print(train_intersection)
# print(val_intersection)

# a = set([1, 2, 3])
# b = set([1, 2, 5])
# print(a.intersection(b))


root_dir = 'D:\\datasets\\project-hyundai-transys-caseleak-datasets-v2\\01_raw\\'

def get_subfolder(dirname):
    subfolders = [f.path for f in os.scandir(dirname) if f.is_dir()]
    for dirname in list(subfolders):
        subfolders.extend(get_subfolder(dirname))
    return subfolders

# subfolders = get_subfolder(root_dir)
# print(subfolders)
# sub = [os.path.dirname(i) for i in subfolders if i.find('image') != -1]
# print(sub)

def get_path_subfolders(root_dir):
    sub_folders = ['Right', 'Rotation', 'Top', 'Top1', 'Front', 'Front_6']
    path_dirs_list = []
    for (path, dirs, files) in os.walk(root_dir):
        if set(dirs) - set(sub_folders) == set() and dirs != []:
            path_dirs_list.append([path, dirs])
    return path_dirs_list

path_dirs_list = get_path_subfolders(root_dir)
print(path_dirs_list)

def get_datasets(path_list):
    datasets = []
    for path in path_list:
        datasets += glob(os.path.join(path, 'image', '*'))
    return datasets

# image_datasets = get_datasets(subfolders)
# print(image_datasets)

def make_datasets_dict(datasets_list):
    dataset_dict = dict()
    for i in datasets_list:
        path_dirs = os.path.split(i)[0].split(os.path.sep)
        file_name = os.path.split(i)[1]
        roi_path, view_dir, last_dir = i.split(os.path.join(path_dirs[-2], path_dirs[-1]))[0], path_dirs[-2], path_dirs[-1]
        dataset_dict[file_name] = [roi_path, view_dir, last_dir]
    return dataset_dict

# image_dict = make_datasets_dict(image_datasets)
# print(image_dict)

def make_pairwise_file(dataset_dict):

    pass




def split_shuffle_test_list(root_path: str, num: int):
    test_dict = defaultdict()
    test_list = []
    for (path, dirs, files) in os.walk(root_path):
        if set(dirs) - set(sub_folders) == set() and dirs != []:
            dir_path = path + f'{os.sep}' + dirs[0]
            # print(dir_path)
            # image_path_list = glob(dir_path + f'{os.sep}' + '.*')
            # img_file_list = is_image(image_path_list)
            img_file_list = glob(dir_path + f'{os.sep}' + '*.jpg')
            # print(img_file_list)
            train_val_list = random.sample(img_file_list, num)  # 이걸로 해야하나.. 아니면 shuffle 후에 100개를 잘라야하나..?
            # test_list = list(set(image_path_list) - set(train_val_list))
            test_list += list(set(img_file_list) - set(train_val_list))
            for i in train_val_list:
                file_name = os.path.basename(i)
                train_val_dict[file_name] = [path, dirs]
            for j in test_list:
                file_name = os.path.basename(j)
                test_dict[file_name] = [path, dirs]
    file_key = list(test_dict.keys())
    test_datasets = make_dict_path(file_key, test_dict)
    return train_val_dict, test_datasets





# import cv2
#
# # image1 = cv2.imread("D:\\datasets\\project-hyundai-transys-caseleak-datasets-v2\\01_raw\\test\\P06L7010853049013D900_L_01-Rear Cover_Top_4000%.jpg")
# # image2 = cv2.imread("D:\\datasets\project-hyundai-transys-caseleak-datasets-v2\\01_raw\\test\\P06L7011827106013D900_L_01-Rear Cover_Top_4000%.jpg")
# # image3 = cv2.imread("D:\\datasets\\project-hyundai-transys-caseleak-datasets-v2\\01_raw\\test\\P06L7020903053013D900_L_01-Rear Cover_Top_4000%.jpg")
# image1 = cv2.imread("D:\\datasets\\project-hyundai-transys-caseleak-datasets-v2\\01_raw\\test\\P10L3041130044064G420_L_01_P7_Top_8000%.jpg")
# image2 = cv2.imread("D:\\datasets\project-hyundai-transys-caseleak-datasets-v2\\01_raw\\test\\P11L1302205132074G420_L_01_P7_Top_8000%.jpg")
# image3 = cv2.imread("D:\\datasets\\project-hyundai-transys-caseleak-datasets-v2\\01_raw\\test\\P11L2060047287064G420_L_01_P7_Top_8000%.jpg")
#
#
# # image = cv2.addWeighted(image1, image2, image3)
# # image = (image1 + image2 + image3) / 3
# image = cv2.add(image1, image2, image3)
# # image = image1 - image
#
# # cv2.imshow("AVG", image1)
# cv2.imshow("AVG", image)
# cv2.waitKey()
# cv2.destroyAllWindows()

