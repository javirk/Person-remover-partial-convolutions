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


class Inpainter:
    def __init__(self, mode, train_dataset=False, test_dataset=False, checkpoint_dir='', restore_parameters=False,
                 epochs=100, lr=2e-4, batch_size=8, initial_epoch=None, writing_per_epoch=10, config_path='config.yml'):
        self.mode = mode
        self.input_channels = 3
        self.epochs = epochs
        self.lr = lr
        self.checkpoint_dir = checkpoint_dir
        self.restore_parameters = restore_parameters
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        config = u.read_config(config_path)

        self.model = self._prepare_model()

        if self.mode == 'train':
            assert train_dataset, 'No training dataset supplied for train mode.'
            self.model.train()
            self.batch_size = batch_size
            self.train_loader, self.train_size = self._make_dataloader(train_dataset, self.batch_size)
            self.test_loader, self.test_size = self._make_dataloader(test_dataset, self.batch_size)

            self.writing_freq = self.train_size // (writing_per_epoch * self.batch_size)

            self.lambda_dict = self._create_lambda_dict(config)

            self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr)
            self.criterion = InpaintingLoss(VGG16FeatureExtractor()).to(self.device)
            if self.restore_parameters:
                print(f'The model will be trained for {self.epochs} epochs and will restore last saved parameters')
                try:
                    checkpoint = torch.load(self._retrieve_last_model())
                    self.initial_epoch = checkpoint['epoch'] if initial_epoch is None else initial_epoch
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as e:
                    print('Error while restoring a checkpoint: ', e)

                    self.initial_epoch = initial_epoch if initial_epoch is not None else 0
                    print(f'The model will be trained for {self.epochs} epochs and will NOT restore last saved parameters')
            else:
                self.initial_epoch = initial_epoch if initial_epoch is not None else 0
                print(f'The model will be trained for {self.epochs} epochs and will NOT restore last saved parameters')

            self.train_summary_writer = self.writers_tensorboard()
        else:
            self.model.eval()
            checkpoint = torch.load(self._retrieve_last_model())
            self.model.load_state_dict(checkpoint['model_state_dict'])

    def _save_parameters(self, epoch):
        path_checkpoint = os.path.join(self.checkpoint_dir, f'model_{epoch}.tar')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path_checkpoint)

    def _retrieve_last_model(self):
        assert os.path.isdir(self.checkpoint_dir), 'Checkpoint_dir must be a directory.'
        files = os.listdir(self.checkpoint_dir)
        files = [int(x[6:-4])  for x in files if 'model' in x]  # This only leaves the number. TODO: regular expression
        last_model_path = os.path.join(self.checkpoint_dir, f'model_{max(files)}.tar')
        return last_model_path

    def _prepare_model(self):
        model = PConvUNet(input_channels=3)
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        model.to(self.device)
        return model

    @staticmethod
    def _make_dataloader(data, batch_size):
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
                        output_com = image + (1 - mask) * output.detach()
                        output_com = output_com.cpu()
                        # writer.add_image('train/original', change_range(torchvision.utils.make_grid(image_gt), 0, 1),
                        #                  n_iter)
                        self.train_summary_writer.add_image('train/output', u.change_range(torchvision.utils.make_grid(output_com), 0, 1),
                                                            n_iter)

                        if self.test_loader is not None:
                            testing_images = iter(self.test_loader).next()
                            test_image_gt = testing_images['image'].to(self.device)
                            test_mask = testing_images['mask'].to(self.device).float()

                            test_image = test_image_gt * test_mask
                            test_output, _ = self.model(test_image, test_mask)
                            test_output_com = test_image + (1 - test_mask) * test_output.detach()
                            test_output_com = test_output_com.cpu()
                            # writer.add_image('test/original', change_range(torchvision.utils.make_grid(test_image_gt), 0, 1),
                            #                  n_iter)
                            self.train_summary_writer.add_image('test/output', u.change_range(torchvision.utils.make_grid(test_output_com), 0, 1), n_iter)

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
