import os
from glob import glob
import shutil

'''
root_dir = '/mnt/hdd/datasets/project-hyundai-transys-caseleak/caseleak/04_results/4th/figures/test/binary_mask/'
# output_dir = '/mnt/hdd/datasets/project-hyundai-transys-caseleak/caseleak/04_results/4th/figures/test_/binary_mask/'

# dir_list = os.listdir(path)
# print(dir_list)

# for root, dirs, files in os.walk(root_dir):
#     print(root, dirs)

file_path = glob(root_dir + '*')
# dir_name = ['binary_mask', 'defaults']
for i in file_path:
    # basename = os.path.basename(i)
    # dir_path = i.split(basename)[0]
    # print(dir_path, basename)
    output_path = i.replace('test', 'test_folder')
    print(output_path)
    shutil.copy(i, output_path)
'''

# stage = 'rsu-en1'
# stage_name = stage.split('-')[0]
# stage_part = stage.split('-')[-1][:2]
# print(stage_name)
# print(stage_part)


impimport torch
import torch.nn as nn
import torch.nn.functional as F








