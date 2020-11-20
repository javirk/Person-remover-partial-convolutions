from inpainter.model import Inpainter
import torchvision.transforms as transforms
from pathlib import Path
import argparse
from libs.data_retriever import InpaintDataset
from libs.utils import str2bool

parser = argparse.ArgumentParser()

parser.add_argument('-b', '--batch-size', default=8, type=int, help='Size of the batch')
parser.add_argument('-e', '--epochs', default=100, type=int, help='Number of epochs')
parser.add_argument('-lr', '--learning-rate', default=2e-4, type=float, help='Initial learning rate')
parser.add_argument('-w', '--write-per-epoch', default=10, type=int, help='Times to write per epoch')
parser.add_argument('-train', '--training-dir', default='Datasets/Paris/paris_train_original', type=str, help='Path for training samples')
parser.add_argument('-test', '--testing-dir', default='Datasets/Paris/paris_eval_gt', type=str, help='Path for testing samples')
parser.add_argument('-test-samples', '--test-samples', default=2, type=int, help='Number of generated samples for testing')
parser.add_argument('-s', '--image-size', default=256, type=int, help='Size of the images (squared)')
parser.add_argument('-r', '--restore-check', default=True, type=str2bool, help='Restore last checkpoint in folder --checkpoint')
parser.add_argument('-c', '--checkpoint-dir', default='inpainter/weights', help='Checkpoint directory')
parser.add_argument('-m', '--mode', default='test', help='Mode: train or test')
parser.add_argument('-ie', '--initial-epoch', default=0, type=int, help='Initial epoch')
parser.add_argument('-config', '--config_file', default='config.yml', type=str, help='Path for config file')
parser.add_argument('-f', '--freeze-bn', type=str2bool, nargs='?', const=True, default=False, help='Freeze BN while training')

def main(FLAGS):
    print('Parameters\n')
    print(f'Image = {FLAGS.image_size}x{FLAGS.image_size}\n')
    normalization_mean = [0.485, 0.456, 0.406]
    normalization_std = [0.229, 0.224, 0.225]

    if FLAGS.mode == 'train':
        train_data_path = Path(__file__).parents[0].joinpath(FLAGS.training_dir)
        test_data_path = Path(__file__).parents[0].joinpath(FLAGS.testing_dir)

        transform_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomCrop((500, 500)),
                                              transforms.Resize(FLAGS.image_size), transforms.ToTensor(),
                                              transforms.Normalize(normalization_mean, normalization_std)])

        transform_test = transforms.Compose([transforms.Resize(FLAGS.image_size), transforms.ToTensor(),
                                             transforms.Normalize(normalization_mean, normalization_std)])

        trainset = InpaintDataset(train_data_path, transform_image=transform_train)
        testset = InpaintDataset(test_data_path, transform_image=transform_test)

        inpainter = Inpainter(FLAGS.mode, trainset, testset, checkpoint_dir=FLAGS.checkpoint_dir, restore_parameters=FLAGS.restore_check,
                              epochs=FLAGS.epochs, lr=FLAGS.learning_rate, batch_size=FLAGS.batch_size, initial_epoch=FLAGS.initial_epoch,
                              writing_per_epoch=FLAGS.write_per_epoch, freeze_bn=FLAGS.freeze_bn, config_path=FLAGS.config_file)
        inpainter.fit()
    else:
        print(f'{FLAGS.mode} mode not implemented yet.')


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)