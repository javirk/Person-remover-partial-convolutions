from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision.datasets.folder import DatasetFolder, has_file_allowed_extension
import random
import torchvision.transforms as transforms
from libs.mask import MaskGenerator
from PIL import Image
from pathlib import Path
import os

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class InpaintDataset(Dataset):
    def __init__(self, folder, transform_image=None):
        super(InpaintDataset, self).__init__()
        assert os.path.isdir(folder), f'{folder} is not a valid path'
        self.folderpath = Path(folder)
        self.transform_image = transform_image
        self.mask_generator = MaskGenerator()

        self.image_set = os.listdir(self.folderpath)
        self.dataset_len = len(self.image_set)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, item):
        image = Image.open(self.folderpath.joinpath(self.image_set[item]))

        seed = torch.randint(0, 2 ** 32, size=(1,))[0]
        if self.transform_image:
            random.seed(seed)
            image = self.transform_image(image)

        height = image.shape[1]
        width = image.shape[2]
        mask = self.mask_generator.sample(height=height, width=width)
        mask = np.concatenate([mask, mask, mask], axis=0) # Because it must have as many channels as the image

        return {'image': image, 'mask': mask}

class InpaintImageFolder(Dataset):
    def __init__(self, folder, image_size, transform_image=None):
        super(Dataset, self).__init__()
        self.folderpath = Path(folder)
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        if transform_image is None:
            normalization_mean = [0.485, 0.456, 0.406]
            normalization_std = [0.229, 0.224, 0.225]
            self.transform_image = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(),
                                                 transforms.Normalize(normalization_mean, normalization_std)])
        else:
            self.transform_image = transform_image

        self.loader = self.make_dataset(self.folderpath)

    def __getitem__(self, item):
        image = Image.open(self.loader[item])
        width, height = image.size
        if self.transform_image:
            image = self.transform_image(image)

        return {'image': image, 'dimensions': (width, height)}

    def __len__(self):
        return len(self.loader)

    @staticmethod
    def make_dataset(folder):
        images = []
        for file in os.listdir(folder):
            if has_file_allowed_extension(file, IMG_EXTENSIONS):
                path = os.path.join(folder, file)
                images.append(path)
        return images



if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    d = InpaintDataset('../Datasets/Paris/paris_eval/', transform_image=transform)