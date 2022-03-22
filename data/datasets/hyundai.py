
import torch
import os
from utils.defaults import load_image
from data.datasets.build import DATASET_REGISTRY
from data.datasets.utils import SemanticFolder
from omegaconf import DictConfig
from collections import defaultdict

@DATASET_REGISTRY.register()
class TransysSemantic:
    def __init__(self, cfg:DictConfig, datasets, transforms):
        self.cfg = cfg
        self.datasets = datasets
        self.transforms = transforms

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, i):
        info = self.datasets[i]
        image = load_image(info.image_path)
        mask = load_image(info.mask_path, type='L')
        if self.transforms is not None:
            image, mask = self.transforms(image, mask)
        # return image.to(self.device), mask.to(self.device) # RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
        return image, mask

class Datasets:
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, cfg):
        self.cfg = cfg
        cls = type(self)
        if not hasattr(cls, "_init"):
            self._datasets = self.get_raw_datasets()
        cls._init = True

    @property
    def get_train(self):
        return self._datasets['train']

    @property
    def get_val(self):
        return self._datasets['val']

    @property
    def get_test(self):
        return self._datasets['test']

    def _get_datasets(self, name):
        return self._datasets[name]

    def get_raw_datasets(self):
        root_dir = self.cfg.training.datasets.root_dir
        root_folder = SemanticFolder()

        datasets = defaultdict(list)

        sub_folder_lists = [x for x in os.listdir(root_dir)]

        for sub_folder_name in sub_folder_lists:
            sub_folder = root_folder.get_sub_folder(sub_folder_name)
            sub_folder.add_meta(os.path.join(root_dir, sub_folder_name))

        # dataset_type: 각 train / val 의 하위 폴더에 저장된 모든 폴더의 정보를 병합
        for dataset_type, sub_folder in root_folder.sub_folder.items():
            # meta_info: image_dicts / mask_dicts / names

            for image_name in sub_folder.meta_folder.names:
                datasets[dataset_type].append(
                    sub_folder.add_path(
                        image_path=sub_folder.meta_folder.image_dicts[image_name],
                        mask_path=sub_folder.meta_folder.mask_dicts[image_name]
                    )
                )

        return datasets