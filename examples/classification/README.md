To run the classification using pretrained model download the pretrained weights file
1. [VGG19 Trained on ImageNet](https://drive.google.com/file/d/0B9ScQjaDDiwpRnVqZV9JQmh4ZE0/view?usp=sharing)
2. [Inception_Resnet_v2 trained on ImageNet](https://drive.google.com/file/d/0B9ScQjaDDiwpTk1kNDBqT1lKRUU/view?usp=sharing)

```
python examples/classification/imagenet_inception_classify.py --weights_from /path/to/trainedweights --image_filename /path/to/image
```
