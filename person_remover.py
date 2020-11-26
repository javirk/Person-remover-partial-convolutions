from detector.model import Detector
from inpainter.model import Inpainter
from libs.data_retriever import InpaintImageFolder
import torch

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
    input_images = data['image']
    mask, segmented_image = detector(input_images)
    print('hola')
