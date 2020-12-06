from detector.model_deeplab import Detector, AVAILABLE_MODELS
from inpainter.model import Inpainter
from libs.data_retriever import IMG_EXTENSIONS
import torch
from libs.utils import save_batch, crop_center, read_image
import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-i', '--image-path',
                    default='Datasets/remove_people/',
                    type=str,
                    help='The path to the directory where images are saved')

parser.add_argument('-io', '--image-output-path',
                    type=str,
                    default='./output/',
                    help='The path of the output photos')

parser.add_argument('-dm', '--detector-model',
                    type=str,
                    default='deeplab',
                    help=F'Detector model name. It has to been one of {", ".join(AVAILABLE_MODELS)}')

parser.add_argument('-e', '--encoder',
                    type=str,
                    default='resnet50dilated',
                    help='Encoder name. Only valid when detector model is MITCSAIL (default)')

parser.add_argument('-d', '--decoder',
                    type=str,
                    default='ppm_deepsup',
                    help='Decoder name. Only valid when detector model is MITCSAIL (default)')

parser.add_argument('-ob', '--objects', nargs='+', type=str, default=['person'])


def main(FLAGS):
    if FLAGS.image_path == FLAGS.image_output_path:
        raise Exception('Input and output directories cannot be the same')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Prepare models
    detector = Detector(FLAGS.detector_model, encoder=FLAGS.encoder, decoder=FLAGS.decoder, object_names=FLAGS.objects)
    inpainter = Inpainter(mode='try', checkpoint_dir='inpainter/weights/')

    for file in os.listdir(FLAGS.image_path):
        if file.lower().endswith(IMG_EXTENSIONS):
            input_file = FLAGS.image_path + file
            image = read_image(input_file, device)
            mask = detector(image)
            torch.cuda.empty_cache()

            image_inpaint = image * mask
            output_inpaint = inpainter(image_inpaint, mask)
            output_inpaint = crop_center(output_inpaint, image.shape[-1], image.shape[-2])

            final_image = image_inpaint + (1 - mask) * output_inpaint
            save_batch(final_image.detach().cpu().numpy(), [file], FLAGS.image_output_path)
            del image_inpaint, image, mask, final_image

if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    main(FLAGS)
