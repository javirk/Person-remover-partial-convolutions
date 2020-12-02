import torch
import torchvision
from torchvision import transforms
import numpy as np
import torch.nn.functional as F

class Detector:
    def __init__(self, name, objects=['person'], names_path='detector/coco.names', threshold=0.5):
        self.class_names = [c.strip() for c in open(names_path).readlines()]
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        if name == 'deeplab':
            self.name = 'deeplab'
            self.objects = [self.class_names.index(x) for x in objects if x in self.class_names]
            object_names = [self.class_names[x] for x in self.objects]
            print(f'Deeplab model. {"-, ".join(object_names)} will be extracted')
            self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        else:
            print('Unet model. Only people will be extracted.')
            pass

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
        dimensions = input_batch.shape[-2:]
        input_batch = input_batch.to(self.device)
        input_batch = F.interpolate(input_batch, size=(640, 640), mode='bilinear')
        with torch.no_grad():
            output = self.model(input_batch)

        mask = F.interpolate(output, size=dimensions, mode='bilinear', align_corners=True)
        mask = F.softmax(mask, dim=1)
        mask = mask[:, 0, ...]
        mask = mask.repeat(1, 3, 1, 1).float()
        return mask

    def mask_objects(self, segmented_image):
        # There has to be a better way to do this. Everything in pytorch, but I don't know.
        return torch.tensor((~np.isin(segmented_image, self.objects)), dtype=torch.float)


if __name__ == '__main__':
    from PIL import Image
    from segmentation_models_pytorch.encoders import get_preprocessing_fn
    model = Detector('otro', names_path='./coco.names')

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

