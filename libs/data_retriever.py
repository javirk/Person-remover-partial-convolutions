from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import h5py
import logging
import random
import skimage.transform
import torchvision.transforms as transforms
from libs.mask import MaskGenerator

logger = logging.getLogger(__name__)


class Resize:
    def __init__(self, size):
        from collections.abc import Iterable
        assert isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)
        if isinstance(size, int):
            self._size = (size, size)
        else:
            self._size = size

    def __call__(self, img: np.ndarray):
        resize_image = skimage.transform.resize(img, self._size)
        # the resize will return a float64 array
        return skimage.util.img_as_ubyte(resize_image)


class OCTInpaintDataset(Dataset):
    """OCT HDF5 dataset."""

    def __init__(self, hdf5_file, input_channels=3, image_set="data/slices", label_set="data/markers", transform_image=None,
                 mask='irregular'):
        '''
        labels: [Healthy, SRF, IRF, HF, Drusen, RPD, ERM, GA, ORA, FPED]
        '''
        self.dataset = None
        self.hdf5_file = hdf5_file
        self.image_set_name = image_set
        self.label_set_name = label_set
        self.input_channels = input_channels

        self.image_set = h5py.File(self.hdf5_file, 'r')[self.image_set_name]  # JGT. It was None before
        self.label_set = h5py.File(self.hdf5_file, 'r')[self.label_set_name]  # JGT. It was None before
        with h5py.File(self.hdf5_file, 'r') as file:
            self.dataset_len = file[image_set].shape[0]

        self.transform_image = transform_image
        self.mask_type = mask
        if self.mask_type == 'irregular':
            self.mask_generator = MaskGenerator()

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if self.image_set is None:
            self.image_set = h5py.File(self.hdf5_file, 'r')[self.image_set_name]

        if self.label_set is None:
            self.label_set = h5py.File(self.hdf5_file, 'r')[self.label_set_name]

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # image = self.image_set[idx] / 256 # Don't need to convert to float if we are converting to PIL later.
        image = self.image_set[idx]
        label = self.label_set[idx].astype(np.float32)

        seed = torch.randint(0, 2 ** 32, size=(1,))[0]
        if self.transform_image:
            random.seed(seed)
            image = self.transform_image(image)
        if self.mask_type == 'random':
            ini_x = random.randint(0, image.shape[1] - 1)
            ini_y = random.randint(0, image.shape[2] - 1)
            mask = self.random_walk(image.shape[1:], (ini_x, ini_y), 50000)
        elif self.mask_type == 'irregular':
            height = image.shape[1]
            width = image.shape[2]
            mask = self.mask_generator.sample(height=height, width=width)
        else:
            mask = np.ones(image.shape[1:])

        if self.input_channels == 3:
            image = np.concatenate([image, image, image], axis=-1)
            mask = np.concatenate([mask, mask, mask], axis=0)

        sample = {'images': image, 'labels': label, 'mask': mask}
        return sample

    @staticmethod
    def random_walk(img_shape, initial_point, length):
        '''
        Get a mask based on a random walk. Maybe better as a transformation for dataloader?
        :param img_shape: set, list. Shape of the image. Channels last.
        :param initial_point: set, list. Place to start the random walk.
        :param length: int. Length of the random walk.
        :return: np.array. Shape = img_shape. 1 in the non-masked places, 0 in the masked ones.
        '''
        # assert img_shape[-1] <= 3, 'Image shape must be channels last'
        action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        x, y = initial_point
        canvas = np.ones(img_shape)
        canvas = np.expand_dims(canvas, axis=0)
        img_size_x = img_shape[0]
        img_size_y = img_shape[1]

        x_list = []
        y_list = []

        for i in range(length):
            r = random.randint(0, len(action_list) - 1)
            x = np.clip(x + action_list[r][0], a_min=0, a_max=img_size_x - 1)
            y = np.clip(y + action_list[r][1], a_min=0, a_max=img_size_y - 1)
            x_list.append(x)
            y_list.append(y)
        canvas[0, np.array(x_list), np.array(y_list)] = 0
        return canvas


if __name__ == '__main__':
    transform = transforms.Compose(
        [Resize(224), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    d = OCTInpaintDataset('../../../Datasets/data_healthy.hdf5', transform_image=transform, mask='irregular')
