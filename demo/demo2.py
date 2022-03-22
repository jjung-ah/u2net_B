

'''
임시.... ㅋㅋ
'''



import hydra
from engine.default import DefaultPredictor
from data.datasets.hyundai import Datasets
from tqdm import tqdm
from utils.defaults import load_image
from utils.logger import Logger
import os
import datetime
import torch
from utils.visualizer import Visualizer
from data.datasets.utils import get_file_dicts
from collections import defaultdict
from data.datasets.meta_classes import CompareMetaInfoSemantic

def get_all_datasets(path: str) -> defaultdict:
    '''
        this function use for compare.py (temporarly code)
        to compare between hae results and transys results whose model is the best
    Args:
        path:

    Returns:

    '''

    datasets = defaultdict(list)

    for root, dirs, files in os.walk(path):
        for file in files:
            name, ext = os.path.splitext(os.path.basename(file))
            if "_result2" in name:
                name = name.replace("_result2", '')
            elif "_result1" in name:
                name = name.replace("_result1", '')
            datasets[name].append(os.path.join(root, file))
    return datasets

def split_datasets_types(datasets: dict) -> defaultdict:
    new_datasets = defaultdict()

    for name, paths in datasets.items():
        image_path= ''
        transys_mask_path = ''
        hae_mask_path = ''

        for path in paths:
            if 'desktop.ini' in path:
                continue
            if 'image' in path:
                image_path = path

        new_datasets[name] = CompareMetaInfoSemantic(image_path=image_path, transys_mask_path=transys_mask_path, hae_mask_path=hae_mask_path)


    return new_datasets

def setup(cfg):
    cfg.logs.root_dir = os.path.join(cfg.training.output_dir, 'logs', 'demo')
    cfg.training.model.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

@hydra.main(config_path="../configs", config_name="config")
def demo(cfg):
    cfg = setup(cfg)

    log_storage = Logger(cfg)
    log_storage.start()
    logger = log_storage.logger

    model = DefaultPredictor(cfg)

    start_time = datetime.datetime.now()
    visualizer = Visualizer(cfg)

    # image list

    image_dict = get_all_datasets(cfg.demo.root_dir)
    image_dict = split_datasets_types(image_dict)

    for key, path in tqdm(image_dict.items(), desc='demo'):
        save_dir = os.path.dirname(os.path.dirname(path.image_path))
        if 'desktop' in key or path.image_path == '':
            continue
        image = load_image(path.image_path)
        image_name = os.path.basename(path.image_path)
        height, width, channel = image.shape

        saliency_maps, fuse_map, image, mask = model(image)

        visualizer.draws(image, saliency_maps, fuse_map, mask)
        visualizer.save_prediction_mask_for_demo2(image_name, height, width, os.path.join(save_dir,'hae-mask'))


    elapsed_time = datetime.datetime.now() - start_time
    logger.info(f'total inference time : {elapsed_time}')

    log_storage.stop()


if __name__ == '__main__':
    demo()