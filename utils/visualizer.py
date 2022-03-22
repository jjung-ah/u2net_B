
import cv2
import os
from pathlib import Path
from data.transforms.transform import post_processing_for_demo
from copy import deepcopy
import numpy as np
from utils.defaults import save_image, resize
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, cfg):
        self.cfg = cfg
        self._images = {}

    @property
    def images(self):
        return self._images

    # TODO: classmethods?
    def blend_on_image(self, image, mask):

        _, seg_bin = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
        mask_image = np.zeros(image.shape, dtype=np.float32)
        mask_image[seg_bin >0, 0] = 255
        blended_image = cv2.addWeighted(image, self.cfg.demo.visualize.blended_weights, mask_image, self.cfg.demo.visualize.blended_weights, 0)

        return blended_image

    def saliency_maps(self, masks):
        return post_processing_for_demo(self.cfg, masks)

    def draws(self, image, saliency_maps, fuse_map, mask=None):
        if mask is not None:
            mask = post_processing_for_demo(self.cfg, mask)
            self._images['gt'] = self.blend_on_image(deepcopy(image), mask)

        fuse_map = post_processing_for_demo(self.cfg, fuse_map)
        self._images['ori'] = deepcopy(image)
        self._images['pred'] = self.blend_on_image(deepcopy(image), fuse_map)
        self._images['fuse_map'] = fuse_map
        if self.cfg.demo.visualize.saliency_map:
            saliency_maps.reverse()
            self._images['saliency_map'] = self.saliency_maps(saliency_maps)

    def _hstack_images(self, images):
        return np.concatenate(images, axis=1)

    def save_saliency_maps(self, image_name, folder_name='test'):
        save_dir = os.path.join(self.cfg.training.output_dir, 'figures', folder_name, 'reports','saliency_maps')
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        hstack_image = self._hstack_images(self._images['saliency_map'])
        save_image(hstack_image, os.path.join(save_dir, image_name))

    def save_prediction_mask(self, image_name, height, width, folder_name='test'):
        save_dir = os.path.join(self.cfg.training.output_dir, 'figures', folder_name, 'binary_mask')
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        fuse_map = resize(self._images['fuse_map'], height, width)
        save_image(fuse_map, os.path.join(save_dir, image_name))

    def save_default_sample(self, image_name, folder_name='test'):
        save_dir = os.path.join(self.cfg.training.output_dir, 'figures', folder_name, 'reports','defaults')
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        images = [self._images[name] for name in self.cfg.demo.visualize.default_samples]
        hstack_image = self._hstack_images(images)
        save_image(hstack_image, os.path.join(save_dir, image_name))

    def save_prediction_mask_for_demo2(self, image_name, height, width, save_dir):
        Path(save_dir).mkdir(exist_ok=True, parents=True)
        fuse_map = resize(self._images['fuse_map'], height, width)
        save_image(fuse_map, os.path.join(save_dir, image_name))


    # def save_each_images(self, image_name: str, height, width, original_size=True):
    #     for name, image in self._images.items():
    #         save_dir = os.path.join(self.cfg.training.output_dir, 'figures', f'{name}')
    #         Path(save_dir).mkdir(exist_ok=True, parents=True)
    #         if len(image.shape) != 3:
    #             image = self._hstack_images(image)
    #             save_image(image, os.path.join(save_dir, image_name))
    #             continue
    #         if original_size:
    #             image = resize(image, height, width)
    #         save_image(image, os.path.join(save_dir, image_name))
    #
    # def _cal_figure_size(self):
    #     row = 1
    #     col = 3
    #     if self.cfg.demo.visualize.saliency_map:
    #         row = 2
    #         col = int(len(self._images['saliency_map']))
    #
    #     return row, col
    #
    # def save_report_images(self, image_name):
    #     row, col = self._cal_figure_size()
    #     fig, axes = plt.subplots(row, col)
    #
    #     if row > 1:
    #         for rows in axes:
    #             for row in rows:
    #                 row.axes.xaxis.set_visible(False)
    #                 row.axes.yaxis.set_visible(False)
    #     else:
    #         for row in axes:
    #             row.axes.xaxis.set_visible(False)
    #             row.axes.yaxis.set_visible(False)
    #
    #     t = -1
    #     for i, name in enumerate(self.cfg.demo.visualize.default_sample):
    #         image = self._images[name]
    #         axes[0][i*2].imshow(image)
    #         axes[0][i*2].title.set_text(name)
    #         t += 2
    #         axes[0][t].axis('off')
    #
    #     if self.cfg.demo.visualize.saliency_map:
    #         for i in range(col):
    #             axes[1][i].imshow(self._images['saliency_map'][i])
    #             axes[1][i].title.set_text(f'stage{i+1}')
    #
    #     plt.tight_layout()
    #     save_dir = os.path.join(self.cfg.training.output_dir, 'figures', 'reports')
    #     Path(save_dir).mkdir(exist_ok=True, parents=True)
    #     plt.savefig(os.path.join(save_dir, image_name))
    #     plt.clf()
    #     plt.close('all')

    def save(self, image_name: str):
        for folder_name, image in self._images.items():
            save_dir = os.path.join(self.cfg.training.output_dir, 'results', folder_name)
            Path(save_dir).mkdir(exist_ok=True, parents=True)
            cv2.imwrite(os.path.join(save_dir, image_name), image)


