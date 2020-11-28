from detector.model import Detector
from inpainter.model import Inpainter
from libs.data_retriever import InpaintImageFolder
import torch
from libs.utils import save_batch, channels_to_last
import matplotlib.pyplot as plt
from skimage.transform import resize

input_folder = 'Datasets/remove_people/'
batch_size = 8
objects = ['person']
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

files = InpaintImageFolder(input_folder, 256)
loader, _ = Inpainter.make_dataloader(files, batch_size)

# Prepare models
detector = Detector(objects)
inpainter = Inpainter(mode='try', checkpoint_dir='inpainter/weights/')

for i, data in enumerate(loader):
    input_images, filenames, dimensions = data['image'], data['filename'], data['dimensions']
    mask, _ = detector(input_images)
    image = input_images * mask
    output = inpainter(image, mask)
    output_numpy = output.detach().cpu().numpy()
    output = resize(channels_to_last(output_numpy), dimensions)
    output_com = image + (1 - mask) * output.detach()
    save_batch(output, filenames, 'output/')