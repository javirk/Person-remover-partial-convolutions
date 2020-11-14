import numpy as np
import torch

def change_range(image, output_min, output_max):
    input_min = torch.min(image)
    input_max = torch.max(image)
    output_image = ((image - input_min) / (input_max - input_min)) * (output_max - output_min) + output_min
    if output_max > 1:
        output_image = output_image.int()
    return output_image

def write_loss_tb(writer, name, loss_dict, lambda_dict, n_iter):
    for key, coef in lambda_dict.items():
        value = coef * loss_dict[key]
        writer.add_scalar(f'{name}/loss_{key}', value.item(), n_iter)
