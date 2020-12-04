import torch
import torchvision
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from mit_semseg.config import cfg
from mit_semseg.dataset import TestDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from pathlib import Path

class Detector:
    def __init__(self, name, object_names=['person'], threshold=0.5):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        if name == 'deeplab':
            self.name = 'deeplab'
            names_path = Path(__file__).parents[0].joinpath('coco.names')
            self.class_names = [c.strip() for c in open(names_path).readlines()]
            self.objects = [self.class_names.index(x) for x in object_names if x in self.class_names]
            print(f'Deeplab model. {"-, ".join(object_names)} will be extracted')
            self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        else:
            self.name = 'resnet50'
            names_path = Path(__file__).parents[0].joinpath('resnet50.names')
            self.class_names = [c.strip() for c in open(names_path).readlines()]
            self.objects = [self.class_names.index(x) for x in object_names if x in self.class_names] # TODO: list is longer. Make it capable to ingest all names
            print(f'Resnet50 model. {"-, ".join(object_names)} will be extracted')
            net_encoder = ModelBuilder.build_encoder(
                arch='resnet50dilated',
                fc_dim=2048,
                weights=str(Path(__file__).parents[0].joinpath('weights', 'encoder_epoch_20.pth')))
            net_decoder = ModelBuilder.build_decoder(
                arch='ppm_deepsup',
                fc_dim=2048,
                num_class=150,
                weights=str(Path(__file__).parents[0].joinpath('weights', 'decoder_epoch_20.pth')),
                use_softmax=True)
            crit = torch.nn.NLLLoss(ignore_index=-1)
            self.model = SegmentationModule(net_encoder, net_decoder, crit)

        self.model.to(self.device)
        self.model.eval()

    def __call__(self, input_batch):
        if self.name == 'deeplab':
            return self.run_deeplab(input_batch)
        else:
            return self.run_unet(input_batch)

    def run_deeplab(self, input_batch):
        input_batch = input_batch.to(self.device)

        # Run the model
        with torch.no_grad():
            output = self.model(input_batch)['out']

        scores = torch.nn.functional.softmax(output, dim=1)
        scores = scores > self.threshold
        scores = torch.sum(scores[:, self.objects, ...], dim=1)
        scores = (scores < 1)
        mask = scores.unsqueeze(1).repeat(1, 3, 1, 1).float()
        return mask

    def run_unet(self, input_batch):
        singleton_batch = {'img_data': input_batch.to(self.device)}
        output_size = input_batch.shape[2:]
        with torch.no_grad():
            scores = self.model(singleton_batch, segSize=output_size)

        # Output values in this case are already probabilities.
        scores = scores > self.threshold
        scores = torch.sum(scores[:, self.objects, ...], dim=1)
        scores = (scores < 1)
        mask = scores.unsqueeze(1).repeat(1, 3, 1, 1).float()

        return mask

    def mask_objects(self, segmented_image):
        # There has to be a better way to do this. Everything in pytorch, but I don't know.
        return torch.tensor((~np.isin(segmented_image, self.objects)), dtype=torch.float)


if __name__ == '__main__':
    from PIL import Image
    from segmentation_models_pytorch.encoders import get_preprocessing_fn
    model = Detector('otro')

    image = Image.open('../Datasets/remove_people/2.png')
    dim = image.size
    image = image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    batch = image.unsqueeze(0)

    res = model(batch)

