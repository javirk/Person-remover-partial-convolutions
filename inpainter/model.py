import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from pathlib import Path
import os
from datetime import datetime
from time import time
from inpainter.loss import InpaintingLoss
from inpainter.partialconv2d import PConvUNet, VGG16FeatureExtractor
import libs.utils as u
import matplotlib.pyplot as plt


class Inpainter:
    def __init__(self, mode, train_dataset=False, test_dataset=False, checkpoint_dir='', restore_parameters=False,
                 epochs=100, lr=2e-4, batch_size=8, initial_epoch=None, writing_per_epoch=10, freeze_bn=False,
                 config_path='config.yml'):
        self.mode = mode
        self.input_channels = 3
        self.epochs = epochs
        self.lr = lr
        self.checkpoint_dir = checkpoint_dir
        self.restore_parameters = restore_parameters
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.freeze_bn = freeze_bn

        config = u.read_config(config_path)

        self.model = self._prepare_model()
        self.batch_size = batch_size

        if self.mode == 'train':
            assert train_dataset, 'No training dataset supplied for train mode.'
            self.model.train()
            self.train_loader, self.train_size = self.make_dataloader(train_dataset, self.batch_size)
            self.test_loader, self.test_size = self.make_dataloader(test_dataset, self.batch_size)

            self.writing_freq = self.train_size // (writing_per_epoch * self.batch_size)

            self.lambda_dict = self._create_lambda_dict(config)

            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
            self.criterion = InpaintingLoss(VGG16FeatureExtractor()).to(self.device)
            if self.restore_parameters:
                print(f'The model will be trained for {self.epochs} epochs and will restore last saved parameters')
                try:
                    checkpoint = torch.load(self._retrieve_last_model(), map_location=self.device)
                    self.initial_epoch = checkpoint['epoch']
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as e:
                    print('Error while restoring a checkpoint: ', e)

                    self.initial_epoch = initial_epoch if initial_epoch is not None else 0
                    print(
                        f'The model will be trained for {self.epochs} epochs and will NOT restore last saved parameters')
            else:
                self.initial_epoch = initial_epoch if initial_epoch is not None else 0
                print(f'The model will be trained for {self.epochs} epochs and will NOT restore last saved parameters')

            self.train_summary_writer = self.writers_tensorboard()
            self._prepare_dirs()
        elif self.mode == 'test':
            self.test_loader, self.test_size = self.make_dataloader(test_dataset, self.batch_size)
            self.model.eval()
            checkpoint = torch.load(self._retrieve_last_model(), map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.eval()
            checkpoint = torch.load(self._retrieve_last_model(), map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])

    def __call__(self, input_batch, mask_batch):
        input_batch.to(self.device)
        mask_batch.to(self.device)
        with torch.no_grad():
            output, _ = self.model(input_batch, mask_batch)
        return output

    def _save_parameters(self, epoch):
        path_checkpoint = os.path.join(self.checkpoint_dir, f'model_{epoch}.tar')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path_checkpoint)

    def _load_pretrained(self, model):
        assert os.path.isdir(self.checkpoint_dir), 'Checkpoint_dir must be a directory.'
        pretrained_model = os.path.join(self.checkpoint_dir, 'pretrained.pth')
        model.load_state_dict(torch.load(pretrained_model, map_location=self.device)['model'])
        return model

    def _retrieve_last_model(self):
        assert os.path.isdir(self.checkpoint_dir), 'Checkpoint_dir must be a directory.'
        files = os.listdir(self.checkpoint_dir)
        files = [int(x[6:-4]) for x in files if 'model' in x]  # This only leaves the number. TODO: regular expression
        last_model_path = os.path.join(self.checkpoint_dir, f'model_{max(files)}.tar')
        return last_model_path

    def _prepare_model(self):
        model = PConvUNet(input_channels=3, freeze_enc_bn=self.freeze_bn)
        model = self._load_pretrained(model)
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        model.to(self.device)
        return model

    @staticmethod
    def make_dataloader(data, batch_size):
        if data:
            if isinstance(data, torch.utils.data.dataloader.DataLoader):
                return data, len(data.dataset)
            else:
                return torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=0), len(data)
        else:
            return None, None

    @staticmethod
    def _create_lambda_dict(config):
        LAMBDA_DICT = dict()
        losses = ['valid', 'hole', 'tv', 'prc', 'style']
        for key in losses:
            LAMBDA_DICT[key] = config['lambda'][key]
        return LAMBDA_DICT

    def fit(self):
        assert self.mode == 'train', 'Must be in train mode to fit.'
        for epoch in range(self.initial_epoch, self.initial_epoch + self.epochs):
            print(f'Epoch {epoch}')
            start = time()
            for i, data in enumerate(self.train_loader):
                image_gt, mask = data['image'].to(self.device), data['mask'].to(self.device)
                mask = mask.float()
                image = image_gt * mask

                output, _ = self.model(image, mask)

                loss_dict = self.criterion(image, mask, output, image_gt)
                loss = InpaintingLoss.calculate_total_loss(loss_dict, self.lambda_dict)

                if (i + 1) % self.writing_freq == 0:
                    n_iter = epoch * self.train_size // self.batch_size + i + 1
                    u.write_loss_tb(self.train_summary_writer, 'train', loss_dict, self.lambda_dict, n_iter)

                    if (i + 1) % (self.writing_freq * 2) == 0:
                        output = u.crop_center(output, mask.shape[-2], mask.shape[-1])
                        output_com = image + (1 - mask) * output.detach()
                        output_com = output_com.cpu()
                        # writer.add_image('train/original', change_range(torchvision.utils.make_grid(image_gt), 0, 1),
                        #                  n_iter)
                        self.train_summary_writer.add_image('train/output',
                                                            u.change_range(torchvision.utils.make_grid(output_com), 0,
                                                                           1), n_iter)

                        if self.test_loader is not None:
                            testing_images = iter(self.test_loader).next()
                            test_image_gt = testing_images['image'].to(self.device)
                            test_mask = testing_images['mask'].to(self.device).float()

                            test_image = test_image_gt * test_mask
                            test_output, _ = self.model(test_image, test_mask)
                            test_output = u.crop_center(test_output, test_mask.shape[-2], test_mask.shape[-1])
                            test_output_com = test_image + (1 - test_mask) * test_output.detach()
                            test_output_com = test_output_com.cpu()
                            # writer.add_image('test/original', change_range(torchvision.utils.make_grid(test_image_gt), 0, 1),
                            #                  n_iter)
                            self.train_summary_writer.add_image('test/output', u.change_range(
                                torchvision.utils.make_grid(test_output_com), 0, 1), n_iter)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f'Epoch {epoch} finished. Time: {time() - start}')
            if (epoch + 1) % 2 == 0:
                self._save_parameters(epoch)

    @staticmethod
    def writers_tensorboard():
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'inpainter/logs/' + current_time
        train_log_dir = Path(__file__).parents[1].joinpath(train_log_dir)
        train_summary_writer = SummaryWriter(train_log_dir)

        return train_summary_writer

    @staticmethod
    def _prepare_dirs():
        os.makedirs('inpainter/weights', exist_ok=True)

    def test_model(self):
        for i, data in enumerate(self.test_loader):
            images_gt, masks = data['image'].to(self.device), data['mask'].to(self.device)
            masks = masks.float()
            images = images_gt * masks
            with torch.no_grad():
                output, _ = self.model(images, masks)

            self._save_images(output, masks, images_gt, 0, i * images_gt.shape[0])

    def _save_images(self, images, masks, images_gt, epoch, ex):
        try:
            masks = masks.detach().cpu().numpy()
            images_gt = images_gt.detach().cpu().numpy()
            images = images.detach().cpu().numpy()
        except:
            pass

        input_im = images_gt * masks
        output_im = input_im + (1 - masks) * images

        title = ['Input image', 'Ground truth', 'Predicted Image']
        for i_im in range(images.shape[0]):
            plt.figure(figsize=(15, 15))
            display_list = [input_im[i_im], images_gt[i_im], output_im[i_im]]
            for i in range(len(title)):
                plt.subplot(1, 3, i + 1)
                plt.title(title[i])
                plt.imshow(u.channels_to_last(u.change_range(display_list[i], 0, 1)))
                plt.axis('off')

            plt.savefig(f'inpainter/output/{self.mode}/salida {epoch}_{ex + i_im}.png')
            plt.close()
