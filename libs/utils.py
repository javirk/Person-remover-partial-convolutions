import torch
import argparse
import yaml

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