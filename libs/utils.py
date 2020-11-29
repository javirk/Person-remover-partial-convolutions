import torch
import argparse
import yaml
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os

def change_range(image, output_min, output_max):
    if type(image) == torch.Tensor:
        input_min = torch.min(image)
        input_max = torch.max(image)
        output_image = ((image - input_min) / (input_max - input_min)) * (output_max - output_min) + output_min
        if output_max > 1:
            output_image = output_image.int()
    else:
        input_min = np.min(image)
        input_max = np.max(image)
        output_image = ((image - input_min) / (input_max - input_min)) * (output_max - output_min) + output_min
        if output_max > 1:
            output_image = output_image.astype(np.uint8)
    return output_image

def write_loss_tb(writer, name, loss_dict, lambda_dict, n_iter):
    for key, coef in lambda_dict.items():
        value = coef * loss_dict[key]
        writer.add_scalar(f'{name}/loss_{key}', value.item(), n_iter)

def read_config(config_path):
    with open(str(config_path)) as file:
        config = yaml.full_load(file)

    return config

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def channels_to_last(arr):
    if type(arr) == torch.Tensor:
        raise ValueError('Not possible to swap dimensions in torch Tensor as of yet.')
    else:
        return np.moveaxis(arr, 0, -1)

def read_image(file):
    image = Image.open(file)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    batch = image.unsqueeze(0)
    return batch

def save_batch(batch, filenames, path):
    for im, file in zip(batch, filenames):
        file = os.path.join(path, file)
        im = channels_to_last(change_range(im, 0, 1))
        plt.imsave(file, im)

def crop_center(img, cropx, cropy, channels_first=True):
    if channels_first:
        _, _, y, x = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        img = img[:, :, starty:starty + cropy, startx:startx + cropx]
    else:
        y, x, _ = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        img = img [:, starty: starty + cropy, startx: startx + cropx, :]

    return img

