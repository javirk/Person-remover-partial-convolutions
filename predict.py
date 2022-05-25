""" Performs object removal through Replicate/Cog"""
import argparse
import os
import tempfile

import cv2
import numpy as np
import torch
import torchvision
from cog import BasePredictor, Input, Path
from PIL import Image
from tqdm import tqdm

from detector.model_deeplab import AVAILABLE_MODELS, Detector
from inpainter.model import Inpainter
from libs.data_retriever import IMG_EXTENSIONS
from libs.utils import crop_center, read_image, save_batch


# Maps object to index
OBJECTS_LIST = ['__background__', 'aeroplane', 'bicycle', 'bird', \
'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',\
'diningtable', 'dog', 'horse', 'motorbike', 'person',\
'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# Instantiate Cog Predictor
class Predictor(BasePredictor):
    def setup(self):

        # Select torch device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, 
        image_path: Path = Input(description="Input image"),
        objects_to_remove: str = Input(description="Object(s) to remove (separate with comma, e.g. car,cat,bird). See full list of names at https://github.com/javirk/Person-remover-partial-convolutions/blob/master/detector/deeplab.names", default='person,car'),
        ) -> Path:

        # parse objects to remove
        objects_to_remove = objects_to_remove.split(',') 

        image_path = str(image_path)  # convert to string
        # Prepare models
        detector_model = "deeplab"
        encoder = "resnet50dilated"
        decoder = "ppm_deepsup"
        detector = Detector(
            detector_model, encoder=encoder, decoder=decoder, object_names=objects_to_remove
        )
        inpainter = Inpainter(mode="try", checkpoint_dir="inpainter/weights/")

        if image_path.lower().endswith(IMG_EXTENSIONS):

            # preprocess image
            image = Image.open(image_path).convert("RGB")
            transform = torchvision.transforms.ToTensor()
            image = transform(image).unsqueeze(0).to(self.device)

            # run model
            mask = detector(image)
            torch.cuda.empty_cache()
            image_inpaint = image * mask
            output_inpaint = inpainter(image_inpaint, mask)
            output_inpaint = crop_center(
                output_inpaint, image.shape[-1], image.shape[-2]
            )
            final_image = image_inpaint + (1 - mask) * output_inpaint

            # save torch tensor to output file path
            output_path = Path(tempfile.mkdtemp()) / "output.png"
            torchvision.utils.save_image(final_image, output_path)
            print(output_path)
            return output_path
