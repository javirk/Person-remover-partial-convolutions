from detector.model import Detector
from inpainter.model import Inpainter
from libs.data_retriever import IMG_EXTENSIONS
import torch
from libs.utils import save_batch, crop_center, read_image
import matplotlib.pyplot as plt
from skimage.transform import resize
import os

input_folder = 'Datasets/remove_people/'
batch_size = 8
objects = ['person']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# files = InpaintImageFolder(input_folder, 256)
# loader, _ = Inpainter.make_dataloader(files, batch_size)

# Prepare models
detector = Detector(objects)
inpainter = Inpainter(mode='try', checkpoint_dir='inpainter/weights/')

for file in os.listdir(input_folder):
    if file.lower().endswith(IMG_EXTENSIONS):
        input_file = input_folder + file
        image = read_image(input_file)
        mask, _ = detector(image)
        image_inpaint = image * mask
        output_inpaint = inpainter(image_inpaint, mask)
        output_inpaint = output_inpaint.detach().cpu().numpy()
        output_inpaint = crop_center(output_inpaint, image.shape[-1], image.shape[-2])
        final_image = image_inpaint + (1 - mask) * output_inpaint
        save_batch(final_image.numpy(), [file], 'output/')