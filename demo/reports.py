import logging

import hydra
from engine.default import DefaultPredictor
from data.datasets.utils import make_file_lists
from data.datasets.hyundai import Datasets
from tqdm import tqdm
from utils.defaults import load_image
from utils.logger import get_logger, setup_logger, Logger
import os
import datetime
import torch
from utils import comm
from utils.visualizer import Visualizer
import cv2

'''
    folder inference
        there are original images and masks(binary)
'''
def setup(cfg):
    cfg.logs.root_dir = os.path.join(cfg.training.output_dir, 'logs', 'demo')
    cfg.training.model.device = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

@hydra.main(config_path="../configs", config_name="config")
def reports(cfg):
    cfg = setup(cfg)

    log_storage = Logger(cfg)
    log_storage.start()
    logger = log_storage.logger

    model = DefaultPredictor(cfg)
    image_lists = Datasets(cfg).get_test

    start_time = datetime.datetime.now()
    logger.info(f'total images: {len(image_lists)}')
    visualizer = Visualizer(cfg)
    for image_meta in tqdm(image_lists, desc='reports'):
        image = load_image(image_meta.image_path)
        height, width, channel = image.shape
        image_name = os.path.basename(image_meta.image_path)

        mask = load_image(image_meta.mask_path, type='L')

        saliency_maps, fuse_map, image, mask = model(image, mask)
        visualizer.draws(image, saliency_maps, fuse_map, mask)
        visualizer.save_default_sample(image_name)
        visualizer.save_saliency_maps(image_name)
        visualizer.save_prediction_mask(image_name, height, width)

    elapsed_time = datetime.datetime.now()-start_time
    logger.info(f'total inference time : {elapsed_time}')

    log_storage.stop()


if __name__ == '__main__':
    reports()