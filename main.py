import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pathlib import Path
from libs.data_retriever import InpaintDataset
import matplotlib.pyplot as plt
from models.loss import InpaintingLoss
from models.partialconv2d import PConvUNet, VGG16FeatureExtractor
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torchvision
import argparse
from time import time
from libs.utils import change_range, write_loss_tb


def main():
    normalization_mean = [0.485, 0.456, 0.406]
    normalization_std = [0.229, 0.224, 0.225]

    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    LAMBDA_DICT = {'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.0}

    train_data_path = Path(__file__).parents[0].joinpath('Datasets', 'Paris', 'paris_train_original')
    test_data_path = Path(__file__).parents[0].joinpath('Datasets', 'Paris', 'paris_eval_gt')
    tb_path = Path(__file__).resolve().parents[0].joinpath('runs/TL_{}'.format(current_time))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(tb_path)

    transform_train = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop((500, 500)), transforms.Resize(256),
         transforms.ToTensor(), transforms.Normalize(normalization_mean, normalization_std)])
    transform_test = transforms.Compose([transforms.Resize(256), transforms.ToTensor(),
                                         transforms.Normalize(normalization_mean, normalization_std)])

    trainset = InpaintDataset(train_data_path, transform_image=transform_train)
    testset = InpaintDataset(test_data_path, transform_image=transform_test)
    set_size = len(trainset)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    writing_freq = set_size // (writing_per_epoch * BATCH_SIZE)

    model = PConvUNet(input_channels=input_channels)
    model.train()
    print('We will use', torch.cuda.device_count(), 'GPUs')
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model.to(device)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = InpaintingLoss(VGG16FeatureExtractor()).to(device)

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch}')
        start = time()
        for i, data in enumerate(trainloader):
            image_gt, mask = data['image'].to(device), data['mask'].to(device)
            mask = mask.float()
            image = image_gt * mask

            output, _ = model(image, mask)

            loss_dict = criterion(image, mask, output, image_gt)
            loss = InpaintingLoss.calculate_total_loss(loss_dict, LAMBDA_DICT)

            if (i + 1) % writing_freq == 0:
                n_iter = epoch * set_size // BATCH_SIZE + i + 1
                write_loss_tb(writer, 'train', loss_dict, LAMBDA_DICT, n_iter)

                if (i + 1) % (writing_freq * 2) == 0:
                    output_com = image + (1 - mask) * output.detach()
                    output_com = output_com.cpu()
                    # writer.add_image('train/original', change_range(torchvision.utils.make_grid(image_gt), 0, 1),
                    #                  n_iter)
                    writer.add_image('train/output', change_range(torchvision.utils.make_grid(output_com), 0, 1),
                                     n_iter)

                    testing_images = iter(testloader).next()
                    test_image_gt = testing_images['image'].to(device)
                    test_mask = testing_images['mask'].to(device).float()

                    test_image = test_image_gt * test_mask
                    test_output, _ = model(test_image, test_mask)
                    test_output_com = test_image + (1 - test_mask) * test_output.detach()
                    test_output_com = test_output_com.cpu()
                    # writer.add_image('test/original', change_range(torchvision.utils.make_grid(test_image_gt), 0, 1),
                    #                  n_iter)
                    writer.add_image('test/output', change_range(torchvision.utils.make_grid(test_output_com), 0, 1),
                                     n_iter)

                print(f'Iteration {n_iter}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch {epoch} finished. Time: {time() - start}')
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), Path(__file__).parents[0].joinpath('weights', f'model_{epoch + 1}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-b', '--batch-size',
                        default=8,
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
                        default=1000,
                        type=int,
                        help='Times to write per epoch')

    parser.add_argument('-i', '--input-channels',
                        default=3,
                        type=int,
                        help='Number of channels in the images')

    FLAGS, unparsed = parser.parse_known_args()

    BATCH_SIZE = FLAGS.batch_size
    EPOCHS = FLAGS.epochs
    LR = FLAGS.learning_rate
    writing_per_epoch = FLAGS.write_per_epoch
    input_channels = FLAGS.input_channels

    main()
