import torch
import torchvision
from torchvision import transforms
import numpy as np
from libs.utils import read_image


class Detector:
    def __init__(self, objects, names_path='detector/coco.names'):
        self.class_names = [c.strip() for c in open(names_path).readlines()]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        self.objects = [self.class_names.index(x) for x in objects if x in self.class_names]

    def __call__(self, input_batch):
        input_batch.to(self.device)

        # Run the model
        with torch.no_grad():
            output = self.model(input_batch)['out']

        segmented_image = output.argmax(1)
        mask = self.mask_objects(segmented_image)

        return mask, segmented_image

    def mask_objects(self, segmented_image):
        return (~np.isin(segmented_image, self.objects)).astype(int)


if __name__ == '__main__':
    from PIL import Image
    i = read_image('../Datasets/remove_people/dog.jpg')
    model = Detector(['dog'], './coco.names')
    mask, segmented_image = model(i)
