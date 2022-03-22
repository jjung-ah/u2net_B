
import os
import cv2
from collections import defaultdict
import numpy as np
from data.datasets.meta_classes import CompareMetaInfoSemantic
import copy
from pathlib import Path
from tqdm import tqdm

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
            if 'result1' in name:
                name = name.replace('_result1', '')
            datasets[name].append(os.path.join(root, file))
    return datasets


def split_datasets_types(datasets: dict) -> defaultdict:
    new_datasets = defaultdict()

    for name, paths in datasets.items():
        image_path= ''
        transys_mask_path = ''
        hae_mask_path = ''

        for path in paths:
            if 'image' in path:
                image_path = path
            elif 'transys-mask' in path:
                transys_mask_path = path
            elif 'hae-mask' in path:
                hae_mask_path = path

        new_datasets[name] = CompareMetaInfoSemantic(image_path=image_path, transys_mask_path=transys_mask_path, hae_mask_path=hae_mask_path)


    return new_datasets

root_dir = '/mnt/hdd/datasets/project-hyundai-transys-caseleak/caseleak/01_raw_datasets/caseleak_1st_model/test_ensemble/sample_test_ensemble_211122_ver1'


def blend_on_image(image, mask):
    _, seg_bin = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    mask_image = np.zeros(image.shape, dtype=np.float32)
    mask_image[mask > 0, 0] = 255
    blended_image = cv2.addWeighted(image, 0.5, mask_image, 0.5, 0)

    return blended_image

def hstack_image(images):
    return np.hstack(images)

def load_image(path: str, type='RGB'):
    image = cv2.imread(path)
    if type == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif type == 'L':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def resize(image, height, width):
    return cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_LINEAR
    )

def save_image(image, path):
    image = cv2.convertScaleAbs(image, alpha=(255.0))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, image)


def visualize(datasets: defaultdict) -> None:

    for name, meta in tqdm(datasets.items()):
        if 'desktop' in name:
            continue
        try:
            image = load_image(meta.image_path)

            image = image.astype(np.float32) / 255.
            image = resize(image, 640, 640)
            hae_mask = load_image(meta.hae_mask_path, 'L')
            hae_mask = resize(hae_mask, 640, 640)
            hae_results = blend_on_image(copy.deepcopy(image), hae_mask)

            transys_mask = load_image(meta.transys_mask_path, 'L')
            transys_mask = resize(transys_mask, 640, 640)
            transys_results = blend_on_image(copy.deepcopy(image), transys_mask)

            results = hstack_image([image, hae_results, transys_results])
        except Exception as e:
            print(f'error: {e}')
            continue


        save_dir = os.path.join(os.path.dirname(os.path.dirname(meta.image_path)), 'compare')
        Path(save_dir).mkdir(exist_ok=True, parents=True)

        save_image(results, os.path.join(save_dir, f'{name}.jpg'))




def compare():
    datasets = get_all_datasets(root_dir)
    datasets = split_datasets_types(datasets)
    visualize(datasets)

if __name__ == '__main__':
    compare()