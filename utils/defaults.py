
import os
import cv2
import numpy as np
from collections import defaultdict
from data.datasets.meta_classes import CompareMetaInfoSemantic

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

def binary_mask(mask, threshold=0.3):
    mask[mask < threshold] = 0
    mask[mask > threshold] = 255
    return mask

def save_image(image, path):
    # if not image.dtype == 'uint8':
    #     image = image.astype(np.uint8)
    image = cv2.convertScaleAbs(image, alpha=(255.0))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(path, image)

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