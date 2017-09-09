[![Build Status](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://openagi.github.io/tefla/)
[![Build Status](https://travis-ci.org/openAGI/tefla.svg?branch=master)](https://travis-ci.org/openAGI/tefla)
[![PyPI version](https://badge.fury.io/py/tefla.svg)](https://badge.fury.io/py/tefla)
[![Build Status](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/openagi/tefla/blob/master/LICENSE)
# Tefla: Deep Learning library, a Higher level API for TensorFlow

Tefla is built on top of Tensorflow. It provides higher level access to tensorflow's features. Inerface, Easy to build complex models. 

Tefla features:

        . Support for data-sets, data-augmentation

        . easy to define complex deep models

        . single and multi GPU training

        . various prediction fnctions including ensembling of models

        . different metrics for performance measurement\

        . custom losses

        . learning rate schedules, polynomial, step, validation_loss based



## TensorFlow Installation

Tefla requires Tensorflow(version >=r1.0)
```Shell
pip install tensorflow-gpu
```
## Additional Requirements
```Shell
sudo apt-get install -y cython
sudo apt-get install python-opencv
pip install scipy
pip install Cython
pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

## Tefla Installation version 
# Tefla 1.2.0 released
For the latest stable version:
```python
pip install tefla
```

for current version installation:
```python
pip install git+https://github.com/n3011/tefla.git
```

For Developer / TO Work with source and modifying source code:
```Shell
git clone https://github.com/n3011/tefla.git
cd tefla
pip install -r requirements.txt
export PYTHONPATH=.
```
## Documentation

[Tefla Docs](https://n3011.github.io/tefla/)


## Tefla Models
Recent deep convolutional models are easy to implement using TEFLA
 
1. [Recent Models](https://github.com/openagi/models/nets)

## Getting Started

1. Its as easy as
```python
>>>from tefla.core.layers import conv2d
>>>convolved = conv2d(input, 48, False, None)

```
# 2 a. Data Directory structure for using normal images
```Shell
|-- Data_Dir
|   |-- training_image_size (eg. training_256, for 256 image size)
|   |-- validation_image_size (eg. validation_256, for 256 image size)
|   |-- training_labels.csv
|   |-- validation_labels.csv
```
# 2 b. TFRecords support available using tefla/dataset class
    1. [Train v2](https://github.com/n3011/tefla/blob/master/tefla/trainv2.py)

# Run training:
  ```Shell
python tefla/train.py --model models/alexnet.py --training_cnf models/multiclass_cnf.py --data_dir /path/to/data/dir (as per instructions 2.a)
  ```
3. Mnist example gives a overview about Tefla usages
 
```python
image_size =(32, 32)
crop_size = (28, 28)
def model(is_training, reuse):
    common_args = common_layer_args(is_training, reuse)
    conv_args = make_args(batch_norm=True, activation=prelu, **common_args)
    fc_args = make_args(activation=prelu, **common_args)
    logit_args = make_args(activation=None, **common_args)

    x = input((None, height, width, 1), **common_args)
    x = conv2d(x, 32, name='conv1_1', **conv_args)
    x = conv2d(x, 32, name='conv1_2', **conv_args)
    x = max_pool(x, name='pool1', **common_args)
    x = dropout(x, drop_p=0.25, name='dropout1', **common_args)
    x = fully_connected(x, n_output=128, name='fc1', **fc_args)
    x = dropout(x, drop_p=0.5, name='dropout2', **common_args)
    logits = fully_connected(x, n_output=10, name="logits", **logit_args)
    predictions = softmax(logits, name='predictions', **common_args)

    return end_points(is_training)

training_cnf = {
    'classification': True,
    'validation_scores': [('validation accuracy', util.accuracy_wrapper), ('validation kappa', util.kappa_wrapper)],
    'num_epochs': 50,
    'lr_policy': StepDecayPolicy(
        schedule={
            0: 0.01,
            30: 0.001,
        }
    )
}
util.init_logging('train.log', file_log_level=logging.INFO, console_log_level=logging.INFO)

trainer = SupervisedTrainer(model, training_cnf, classification=training_cnf['classification'])
trainer.fit(data_set, weights_from=None, start_epoch=1, verbose=1, summary_every=10)
```



## Contributions

Welcome to the first release of Tefla, if you find any bug, please report it in the GitHub issues section.

Improvements and requests for new features are more than welcome! Do not hesitate to twist and tweak Tefla, and send pull-requests.


## License

[MIT License](https://n3011.github.io/tefla/license/)
