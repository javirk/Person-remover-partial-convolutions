import torch
import torchvision
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
from pathlib import Path
from libs.utils import crop_center

AVAILABLE_MODELS = ['mitcsail', 'deeplab','ssd']

class Detector:
    def __init__(self, name, object_names=['person'], threshold=0.5, encoder=None, decoder=None, encoder_path=None, decoder_path=None):
        assert name.lower() in AVAILABLE_MODELS, 'Model has to be one of ' + ', '.join(AVAILABLE_MODELS)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        if name.lower() == 'mitcsail':
            try:
                # TODO: this is ugly but functional
                assert encoder is not None and decoder is not None, 'Encoder and decoder names must be given to run mitcsail model'

                self.name = 'mitcsail'
                names_path = Path(__file__).parents[0].joinpath('mitcsail.names')
                self.class_names = [c.strip() for c in open(names_path).readlines()]
                self.objects = [self.class_names.index(x) for x in object_names if
                                x in self.class_names]  # TODO: list is longer. Make it capable to ingest all names
                self.model = self._load_mistcsail_model(encoder, decoder, encoder_path, decoder_path)
                print(f'MITCSAIL model loaded. {"-, ".join(object_names)} will be extracted')
                self.run_model = self.run_mitcsail

            except ModuleNotFoundError:
                name = 'deeplab'
                print('MITCSAIL import model not found. Install it with "pip install git+https://github.com/CSAILVision/semantic-segmentation-pytorch.git@master".')
                print('Using deeplab for the moment')

        if name.lower() == 'ssd':
            self.name = 'ssd'
            names_path = Path(__file__).parents[0].joinpath('ssd.names')
            self.class_names = [c.strip() for c in open(names_path).readlines()]
            self.objects = [self.class_names.index(x) for x in object_names if x in self.class_names]
            print(f'SSD model. {"-, ".join(object_names)} will be extracted')

            self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math='fp32')
            self.ssd_utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')
            self.run_model = self.run_ssd

        if name.lower() == 'deeplab':
            self.name = 'deeplab'
            names_path = Path(__file__).parents[0].joinpath('deeplab.names')
            self.class_names = [c.strip() for c in open(names_path).readlines()]
            self.objects = [self.class_names.index(x) for x in object_names if x in self.class_names]
            print(f'Deeplab model. {"-, ".join(object_names)} will be extracted')
            self.model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
            self.run_model = self.run_deeplab

        try:
            self.model = self.model.to(self.device)
        except AttributeError:
            print(f'{name} is not a valid model')
        self.model.eval()

    def __call__(self, input_batch, factor_augment=None):
        return self.run_model(input_batch, factor_augment)

    @staticmethod
    def _load_mistcsail_model(encoder, decoder, encoder_path=None, decoder_path=None):
        from mit_semseg.models import ModelBuilder, SegmentationModule
        from mit_semseg.config import cfg
        if encoder_path is None:
            encoder_path = str(Path(__file__).parents[0].joinpath('weights', f'encoder_{encoder}.pth'))
        if decoder_path is None:
            decoder_path = str(Path(__file__).parents[0].joinpath('weights', f'decoder_{decoder}.pth'))

        config_file = str(Path(__file__).parents[0].joinpath('config', f'ade20k-{encoder}-{decoder}.yaml'))

        cfg.merge_from_file(config_file)

        net_encoder = ModelBuilder.build_encoder(
            arch=encoder,
            fc_dim=cfg.MODEL.fc_dim,
            weights=encoder_path)
        net_decoder = ModelBuilder.build_decoder(
            arch=decoder,
            fc_dim=cfg.MODEL.fc_dim,
            num_class=cfg.DATASET.num_class,
            weights=decoder_path,
            use_softmax=True)

        crit = torch.nn.NLLLoss(ignore_index=-1)
        model = SegmentationModule(net_encoder, net_decoder, crit)
        return model

    def run_ssd(self, input_batch, factor_augment=None):
        new_dim = 400
        original_dims = input_batch.shape[-2:]
        input_batch = F.interpolate(input_batch, (new_dim, new_dim))
        with torch.no_grad():
            detections_batch = self.model(input_batch)

        results_per_input = self.ssd_utils.decode_results(detections_batch)
        best_results_per_input = [self.ssd_utils.pick_best(results, 0.40) for results in results_per_input]

        bboxes, classes, confidences = best_results_per_input[0]
        mask = torch.ones_like(input_batch)
        for bbox, cl, confidence in zip(bboxes, classes, confidences):
            if cl - 1 in self.objects:
                left, bot, right, top = bbox
                x1, y1, x2, y2 = [int(val * 300) for val in [left, bot, right, top]]
                mask[..., y1:y2, x1:x2] = 0

        mask = F.interpolate(mask, original_dims).float()

        return mask

    def run_deeplab(self, input_batch, factor_augment=None):
        # It seems that the segmentation model always leaves the right parts unsegmented, so we are duplicating
        # the batch and flipping
        flipped_batch = torch.flip(input_batch, [-1])
        batch = torch.cat((input_batch, flipped_batch), dim=0)

        # Run the model
        with torch.no_grad():
            output = self.model(batch)['out']

        scores = torch.nn.functional.softmax(output, dim=1)
        scores = scores > self.threshold
        scores = torch.sum(scores[:, self.objects, ...], dim=1)
        scores = (~(scores < 1)).float()

        scores = scores[:scores.shape[0] // 2] + torch.flip(scores[scores.shape[0] // 2:], [-1])
        scores = (scores == 0).float()

        mask = scores.unsqueeze(1).repeat(1, 3, 1, 1).float()

        return mask

    def run_mitcsail(self, input_batch, factor_augment=None):
        # It seems that the segmentation model always leaves the right parts unsegmented, so we are duplicating
        # the batch and flipping

        flipped_batch = torch.flip(input_batch, [-1])
        batch = torch.cat((input_batch, flipped_batch), dim=0)
        singleton_batch = {'img_data': batch}
        output_size = input_batch.shape[2:]

        with torch.no_grad():
            scores = self.model(singleton_batch, segSize=output_size)

        del singleton_batch, batch, flipped_batch

        # Output values in this case are already probabilities.
        scores = scores > self.threshold
        scores = torch.sum(scores[:, self.objects, ...], dim=1)
        scores = (~(scores < 1)).float()

        scores = scores[:scores.shape[0] // 2] + torch.flip(scores[scores.shape[0] // 2:], [-1])
        scores = (scores == 0).float()

        mask = scores.unsqueeze(1).repeat(1, 3, 1, 1).float()
        # TODO: Method to expand the mask by offsetting pixels
        # if factor_augment is not None:
        #     assert factor_augment > 1, 'Augmentation factor must be greater than 1'
        #     initial_dim = np.array(mask.shape[2:])
        #     mask_augment = F.interpolate(mask, scale_factor=factor_augment, mode='nearest', align_corners=True)
        #     mask_augment = crop_center(mask_augment, initial_dim[1], initial_dim[0])
        # else:
        #     mask_augment = mask

        return mask

    def mask_objects(self, segmented_image):
        # There has to be a better way to do this. Everything in pytorch, but I don't know.
        return torch.tensor((~np.isin(segmented_image, self.objects)), dtype=torch.float)


if __name__ == '__main__':
    from PIL import Image
    import matplotlib.pyplot as plt
    from time import time
    model = Detector('ssd', encoder='mobilenetv2dilated', decoder='c1_deepsup')

    image = Image.open('../Datasets/remove_people/2.png')
    dim = image.size
    image = image.convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    batch = image.unsqueeze(0)
    batch = batch.to(model.device)

    # flipped_batch = torch.flip(batch, [-1])
    # batch = torch.cat((batch, flipped_batch), dim=0)
    # singleton_batch = {'img_data': batch.to(model.device)}
    # output_size = batch.shape[2:]
    # start = time()
    # with torch.no_grad():
    #     scores = model.model(singleton_batch, segSize=output_size)
    # print(time() - start)

    start = time()
    res = model(batch)
    print(time() - start)

