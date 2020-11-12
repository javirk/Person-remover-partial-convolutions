from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import random
import torchvision.transforms as transforms
from libs.mask import MaskGenerator
from PIL import Image
from pathlib import Path


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
        mask = mask.float()

        return {'image': image, 'mask': mask}


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    d = InpaintDataset('../Datasets/Paris/paris_eval/', transform_image=transform)