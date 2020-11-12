import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path
from libs.data_retriever import OCTInpaintDataset
import matplotlib.pyplot as plt
from models.loss import InpaintingLoss
from models.partialconv2d import PConvUNet, VGG16FeatureExtractor
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torchvision
import argparse
from time import time
from libs.utils import change_range

def main(BATCH_SIZE, EPOCHS, LR, writing_per_epoch, input_channels):
    if input_channels == 3:
        normalization_mean = [0.485, 0.456, 0.406]
        normalization_std = [0.229, 0.224, 0.225]
    elif input_channels == 1:
        normalization_mean = [0.5]
        normalization_std = [0.5]
    else:
        raise ValueError('Input channels must be either 1 or 3')

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    LAMBDA_DICT = {'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.0}
    data_path = Path(__file__).parents[0].joinpath('Datasets', 'data_healthy.hdf5')
    tb_path = Path(__file__).resolve().parents[0].joinpath('runs/TL_{}'.format(current_time))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(tb_path)

    transform = transforms.Compose(
        [transforms.ToPILImage(), transforms.Resize([256, 256]), transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(2, fill=1), transforms.ToTensor(),
         transforms.Normalize(normalization_mean, normalization_std)])

    totalset = OCTInpaintDataset(data_path, input_channels=input_channels, transform_image=transform)
    set_size = len(totalset)
    trainloader = torch.utils.data.DataLoader(totalset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    writing_freq = set_size // (writing_per_epoch * BATCH_SIZE)

    model = PConvUNet(input_channels=input_channels)
    model.train()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    model.to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device)

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch}')
        start = time()
        for i, data in enumerate(trainloader):
            image_gt, mask = data['images'].to(device), data['mask'].to(device)
            mask = mask.float()
            image = image_gt * mask

            output, _ = model(image, mask)

            loss_dict = criterion(image, mask, output, image_gt)

            loss = 0.0
            for key, coef in LAMBDA_DICT.items():
                value = coef * loss_dict[key]
                loss += value

            if (i + 1) % writing_freq == 0:
                n_iter = epoch * set_size // BATCH_SIZE + i + 1
                output_com = image + (1 - mask) * output
                writer.add_image('original', change_range(torchvision.utils.make_grid(image_gt), 0, 1), n_iter)
                writer.add_image('output', change_range(torchvision.utils.make_grid(output_com), 0, 1), n_iter)

                print(f'Iteration {n_iter}')
                for key, coef in LAMBDA_DICT.items():
                    value = coef * loss_dict[key]
                    writer.add_scalar('loss_{:s}'.format(key), value.item(), n_iter)
                    print(f'\t{key}: {value}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch} finished. Time: {time() - start}')
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), Path(__file__).parents[0].joinpath('weights', f'model_{epoch + 1}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch-size',
                        default=32,
                        type=int,
                        help='Size of the batch')

    parser.add_argument('-e', '--epochs',
                        default=100,
                        type=int,
                        help='Number of epochs')

    parser.add_argument('-lr', '--learning-rate',
                        default=2e-4,
                        type=float,
                        help='Initial learning rate')

    parser.add_argument('-w', '--write-per-epoch',
                        default=10,
                        type=int,
                        help='Times to write per epoch')

    parser.add_argument('-i', '--input-channels',
                        default=1,
                        type=int,
                        help='Number of channels in the images')

    FLAGS, unparsed = parser.parse_known_args()

    main(FLAGS.batch_size, FLAGS.epochs, FLAGS.learning_rate, FLAGS.write_per_epoch, FLAGS.input_channels)
