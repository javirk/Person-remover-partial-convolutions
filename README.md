# Person Remover v2

Would you like to travel to a touristic spot and yet appear alone in the photos? 

_Person remover_ is a project that uses partial convolutions to remove people or other objects from
photos. For partial convolutions, the code by [naoto0804](https://github.com/naoto0804/pytorch-inpainting-with-partial-conv) has been adapted,
whereas for segmentation models, either torch hub models or the code by [MIT CSAIL Computer Vision](https://github.com/CSAILVision/semantic-segmentation-pytorch),
have been used.

This project is capable of removing objects in images and video.

Python 3.7 and Pytorch 1.7.0 a have been used in this project.


## How does it work?

A model with partial convolutions has been trained to fill holes in images. These instructions will you train a model in
your local machine. However, the training dataset that has been used for the model are not 
[publicly available](http://graphics.cs.cmu.edu/projects/whatMakesParis/). This dataset consists of 14900,
256x256x3 images. The code handles the creation of a hole in the center of the images and learns how to fill it with the
surrounding data.

### Requisites

In order to use the program Python 3.7 and the libraries specified in  `requirements.txt` should be installed.

### Installation

Clone the repository
```
git clone https://github.com/javirk/Person-remover-partial-convolutions.git
```

##### Using MIT CSAIL models for segmentation
If you want to use MIT CSAIL models, first pip install the library by
```
pip install git+https://github.com/CSAILVision/semantic-segmentation-pytorch.git@master
```
Then, download and save appropriate weights and config. Pretrained weights are available [here](http://sceneparsing.csail.mit.edu/model/pytorch/).
They should be renamed as "encoder/decoder_"+model name. For example, to use ppm_deepsup as decoder and resnet50dilated as 
encoder, the following files should be present under `./detector/weights` folder:
```
encoder_resnet50dilated.pth
decoder_ppm_deepsup.pth
```
Configuration files are available in [their github repo](https://github.com/CSAILVision/semantic-segmentation-pytorch) and
must be placed in detector/config folder. For the example above, the file `ade20k-resnet50dilated-ppm_deepsup.yaml` must be
in said folder.

Download the weights for partial convolutions from [Google Drive](https://drive.google.com/file/d/12Y9OzZjw6yTPPLqhBMEnBn1r4I_83UIC/view?usp=sharing)
and put them in `./inpainter/weights/`.

To get results of images, run `person_remover.py`:
```
python person_remover.py -i /dir/of/input/images
``` 
This will use deeplab model by default. One can change the segmentation model with:
```
python person_remover.py -i /dir/of/input/images -dm mitcsail -e resnet50dilated -d ppm_deepsup
``` 

It is also possible to specify the type of object to remove (only people are chosen by default):
```
python person_remover.py -i /dir/to/input/images -ob person bicycle car
``` 
Which will remove people, bycicles and cars. Take into account that the objects to remove depend on the segmentation model,
and are defined in their respective `.names` files in `./detector/` folder.

### Training

Segmentation models are taken pretrained. For partial convolutions networks, the training has spanned 47 epochs or fine tuning
with batch normalization layers in the decoder set to not trainable in a dataset of 14900 training and 100 test images 
using the default parameters. It is worth noticing that the training process is extremely sensitive, so the best results
might not come in the first run.

Training with the default parameters (check `image_inpainting.py` for reference) is performed as follows:
```
python image_inpainting.py -train /dir/of/training/images -test /dir/of/test/images -mode train
```
This will restart training from the last saved model in `inpainter/weights` folder. To train from scratch you should use:
```
python image_inpainting.py -train /dir/of/training/images -test /dir/of/test/images -mode train -r False
```

## Image removal

![p2p_fill_3](https://github.com/javirk/Person-remover-partial-convolutions/blob/master/images_readme/Imagen1.png)
![p2p_fill_4](https://github.com/javirk/Person-remover-partial-convolutions/blob/master/images_readme/Imagen2.png)
![p2p_fill_5](https://github.com/javirk/Person-remover-partial-convolutions/blob/master/images_readme/Imagen3.png)
![p2p_fill_6](https://github.com/javirk/Person-remover-partial-convolutions/blob/master/images_readme/Imagen4.png)
![p2p_fill_7](https://github.com/javirk/Person-remover-partial-convolutions/blob/master/images_readme/Imagen5.png)
![p2p_fill_8](https://github.com/javirk/Person-remover-partial-convolutions/blob/master/images_readme/Imagen6.png)
![p2p_fill_9](https://github.com/javirk/Person-remover-partial-convolutions/blob/master/images_readme/Imagen7.png)

## Next steps

The quality of the filling heavily relies on the segmentator, a bigger mask that covers the whole person
is better than leaving out small parts such as one hand or one foot because the network tends to use these pixels to 
inpaint the rest. Working on a way to expand the mask by pixel offsetting would very likely lead to better results.

Partial convolutions at times modify the lightning conditions of the image, which then results in big borders around the
filled areas. A smoother mix between inpainted and original images would be a quantitative step towards visual reality.

With the available weights, big artifacts are present in some cases due to the training methodology. A retraining will 
be carried out to tackle this issue.

## Author

* **Javier Gamazo** - *Initial work* - [Github](https://github.com/javirk). [LinkedIn](https://www.linkedin.com/in/javier-gamazo-tejero/)

## License

This project is under Apache license. See [LICENSE.md](LICENSE.md) for more details.

## Acknowledgments

* [naoto0804](https://github.com/naoto0804/pytorch-inpainting-with-partial-conv) for partial convolutions initial code
* [CSAIL Vision](https://github.com/CSAILVision/semantic-segmentation-pytorch) for segmentation models
