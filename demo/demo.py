

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
    image_dict = get_file_dicts(cfg.demo.image_dir, cfg.demo.image_ext)
    folder_name = os.path.basename(cfg.demo.image_dir)
    for _, path in tqdm(image_dict.items(), desc='demo'):
        image = load_image(path)
        image_name = os.path.basename(path)
        height, width, channel = image.shape

        saliency_maps, fuse_map, image, mask = model(image)

        visualizer.draws(image, saliency_maps, fuse_map, mask)
        visualizer.save_default_sample(image_name, folder_name)
        visualizer.save_saliency_maps(image_name, folder_name)
        visualizer.save_prediction_mask(image_name, height, width, folder_name)


    elapsed_time = datetime.datetime.now() - start_time
    logger.info(f'total inference time : {elapsed_time}')

    log_storage.stop()


if __name__ == '__main__':
    demo()